"""LLM 기반 보고서 생성을 위한 서비스 모듈."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

import httpx
from dotenv import load_dotenv
import base64
import requests


load_dotenv()


class LLMReportService:
    """LLM을 활용해 주유소 보고서를 생성하는 서비스."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        auth_scheme: Optional[str] = None,
    ) -> None:
        raw_api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.api_key = raw_api_key.strip() if raw_api_key else None
        self.base_url = base_url or os.getenv(
            "LLM_API_URL", "https://api.openai.com/v1/chat/completions"
        )
        self.model = model or os.getenv("LLM_MODEL", "gpt-5-mini")
        self.auth_scheme = (auth_scheme or os.getenv("LLM_AUTH_SCHEME") or "Bearer").strip()
        try:
            default_timeout = float(os.getenv("LLM_TIMEOUT", "30"))
        except ValueError:
            default_timeout = 30.0
        self.timeout = timeout or default_timeout
        self.force_json_response = os.getenv("LLM_FORCE_JSON", "true").lower() != "false"
        try:
            self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        except ValueError:
            self.temperature = 0.3
        self.routing_table = self._load_routing_table()
        self.google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()


    def _load_routing_table(self) -> Dict[str, Dict[str, Any]]:
        raw_table = os.getenv("LLM_ROUTING_TABLE")
        data: Optional[Dict[str, Any]] = None

        if raw_table:
            try:
                parsed = json.loads(raw_table)
                if isinstance(parsed, dict):
                    data = parsed
            except json.JSONDecodeError as exc:
                print(f"LLM 라우팅 테이블 파싱 실패: {exc}")

        if data is None:
            routing_file = os.getenv("LLM_ROUTING_FILE")
            if routing_file:
                try:
                    with Path(routing_file).expanduser().open("r", encoding="utf-8") as fp:
                        parsed = json.load(fp)
                    if isinstance(parsed, dict):
                        data = parsed
                except Exception as exc:
                    print(f"LLM 라우팅 파일 로드 실패: {exc}")

        if not data:
            return {}

        table: Dict[str, Dict[str, Any]] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                table[str(key)] = value

        return table

    def _resolve_route(self, station_id: Optional[int]) -> Dict[str, Any]:
        defaults = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "force_json": self.force_json_response,
            "temperature": self.temperature,
            "auth_scheme": self.auth_scheme,
        }

        if not self.routing_table:
            return defaults

        candidate: Optional[Dict[str, Any]] = None
        if station_id is not None:
            candidate = self.routing_table.get(str(station_id))

        if candidate is None:
            candidate = (
                self.routing_table.get("*")
                or self.routing_table.get("default")
                or self.routing_table.get("DEFAULT")
            )

        if not candidate:
            return defaults

        merged = defaults.copy()
        for key, value in candidate.items():
            if key in {"timeout", "temperature"}:
                try:
                    merged[key] = float(value)
                except (TypeError, ValueError):
                    continue
            elif key == "force_json":
                merged[key] = self._normalise_bool(value)
            else:
                merged[key] = value

        return merged

    @staticmethod
    def _normalise_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).lower() not in {"false", "0", "no"}

    async def generate_report(
        self,
        station: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        parcel_summary: Optional[Dict[str, Any]] = None,
        station_id: Optional[int] = None,
        map_images: Optional[Dict[str, str]] = None,
        stats_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """보고서에 포함할 요약/인사이트/실행항목을 반환한다."""

        route_config = self._resolve_route(station_id)
        llm_response = await self._request_llm(
            station,
            recommendations,
            parcel_summary,
            route_config,
            station_id,
            map_images,
            stats_payload,
        )
        if llm_response:
            parsed = self._parse_llm_response(llm_response)
            if parsed:
                return parsed

        return self._fallback_report(station, recommendations, parcel_summary)

    async def _request_llm(
        self,
        station: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        parcel_summary: Optional[Dict[str, Any]],
        route_config: Dict[str, Any],
        station_id: Optional[int],
        map_images: Optional[Dict[str, str]],
        stats_payload: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """LLM API 호출. 실패 시 None."""

        api_key = (route_config.get("api_key") or "").strip()
        if not api_key:
            return None

        station_summary = self._summarise_station(station)
        recommendation_summary = self._summarise_recommendations(recommendations)
        parcel_context = self._format_parcel_summary(parcel_summary)
        station_ref = station.get("상호") or station.get("name") or "해당 주유소"
        station_identifier = f"ID {station_id} - {station_ref}" if station_id is not None else station_ref

        visual_context = self._build_visual_prompt_section(map_images or {})
        stats_context = self._summarise_stats_for_prompt(stats_payload)

        user_prompt = (
            "당신은 도시 재생 및 부동산 활용 전략을 제시하는 컨설턴트입니다. 아래 주유소 정보를 분석하여 "
            "입지 특성 요약(2~3문장), 조사 현황(로드뷰·분석 지표 기반 3줄 내외 불릿), 권장 실행 항목 3개, "
            "그리고 세부 추천 활용안을 JSON으로만 응답하세요.\n"
            "summary는 위성사진과 주유소 위치를 근거로 주변환경을 묘사하고, investigation은 로드뷰와 "
            "분석 지표(비중 높게 반영)를 근거로 현장 상태를 설명하는 불릿 텍스트로 작성합니다.\n"
            "**중요**: investigation은 반드시 줄바꿈(\\n)으로 구분된 3개의 불릿 항목으로 작성하세요. "
            "각 불릿 항목은 한 문장으로 구성하고, 불릿 기호(•, -)는 포함하지 마세요. 줄바꿈만으로 구분합니다.\n"
            "JSON 키는 summary(문장), investigation(멀티라인 문자열), actions(문장 리스트), usage_programs(리스트)입니다.\n"
            "\n"
            "**usage_programs 작성 규칙**:\n"
            "- 반드시 아래 [추천 활용 방안]에 제공된 순서와 명칭(1~3순위)을 그대로 사용하세요. 임의로 변경하거나 새로운 순위를 만들지 마세요.\n"
            "- 각 항목은 {\"usage\":\"[추천 용도]\", \"rank\":순번, \"programs\":[{\"name\":\"프로그램명\", \"reason\":\"선정 이유 2~3문장\"}, ...]} 형식의 JSON 객체로 작성하세요.\n"
            "- 각 순위별로 프로그램은 정확히 3개를 작성하고, 선정 이유는 2~3문장으로 서술하되 불릿/개행 없이 문장만 포함합니다.\n"
            "- usage에는 제공된 추천 용도명을 그대로 입력하고, programs 배열 외의 여분 텍스트나 마크다운은 추가하지 마세요.\n"
            "\n"
            "모든 문장은 한국어 비즈니스 보고서 어투로 작성하고, JSON 이외의 다른 설명이나 마크다운은 포함하지 마세요.\n"
            "추천 활용안은 위성사진, 로드뷰, 분석 지표 내용을 모두 참고하되 분석 지표에 가장 높은 비중을 두세요.\n\n"
            f"[대상 주유소] {station_identifier}\n"
            f"[주유소 정보]\n{station_summary}\n\n"
            f"[추천 활용 방안]\n{recommendation_summary}\n"
            f"[반경 300m 필지 통계]\n{parcel_context}\n"
        )

        if visual_context:
            user_prompt += f"[이미지 데이터]\n{visual_context}\n\n"
        if stats_context:
            user_prompt += f"[입지 분석 지표]\n{stats_context}\n"

        messages = [
            {
                "role": "system",
                "content": "도시 입지 분석을 수행하는 한국어 컨설턴트입니다.",
            },
            {"role": "user", "content": user_prompt},
        ]

        headers = self._build_headers(api_key, route_config)
        payload = {
            "model": route_config.get("model", self.model),
            "messages": messages,
            "temperature": route_config.get("temperature", self.temperature),
        }
        if route_config.get("force_json", self.force_json_response):
            payload["response_format"] = {"type": "json_object"}

        try:
            timeout_value = route_config.get("timeout", self.timeout)
            async with httpx.AsyncClient(timeout=timeout_value) as client:
                response = await client.post(
                    route_config.get("base_url", self.base_url),
                    headers=headers,
                    json=payload,
                )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return None
            content = choices[0].get("message", {}).get("content", "")
            return content.strip() or None
        except Exception as exc:  # pragma: no cover - 네트워크 예외 처리
            print(f"LLM 보고서 생성 실패: {exc}")
            return None


    def prepare_map_images(
        self, lat: Optional[Any], lng: Optional[Any], *, width: int = 600, height: int = 450
    ) -> Dict[str, str]:
        """LLM 및 보고서 공용으로 사용할 지도/로드뷰 이미지 Base64 묶음."""

        images: Dict[str, str] = {}
        if not lat or not lng or not self.google_maps_api_key:
            return images

        try:
            sat_b64 = self._fetch_satellite_image_b64(
                float(lat), float(lng), width=width, height=height, zoom=18
            )
            if sat_b64:
                images["satellite"] = sat_b64
        except Exception as exc:
            print(f"[Satellite] 이미지 생성 실패: {exc}")

        try:
            rv1_b64 = self._fetch_streetview_image_b64(
                float(lat),
                float(lng),
                heading=0,
                pitch=0,
                width=width,
                height=height,
                fov=90,
            )
            if rv1_b64:
                images["streetview1"] = rv1_b64
        except Exception as exc:
            print(f"[StreetView1] 이미지 생성 실패: {exc}")

        try:
            rv2_b64 = self._fetch_streetview_image_b64(
                float(lat),
                float(lng),
                heading=180,
                pitch=0,
                width=width,
                height=height,
                fov=90,
            )
            if rv2_b64:
                images["streetview2"] = rv2_b64
        except Exception as exc:
            print(f"[StreetView2] 이미지 생성 실패: {exc}")

        return images


    def _build_headers(self, api_key: str, route_config: Dict[str, Any]) -> Dict[str, str]:
        """인증 스킴에 맞게 헤더를 구성한다."""

        auth_scheme = (route_config.get("auth_scheme") or self.auth_scheme or "Bearer").strip()
        headers = {"Content-Type": "application/json"}

        if auth_scheme.lower() == "basic":
            headers["Authorization"] = f"Basic {api_key}"
        else:
            headers["Authorization"] = f"{auth_scheme} {api_key}".strip()

        return headers

    def _parse_llm_response(self, content: str) -> Optional[Dict[str, Any]]:
        """LLM 응답을 JSON으로 파싱."""

        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").lstrip("json").strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        summary = str(data.get("summary", "")).strip()
        insights = [str(item).strip() for item in data.get("insights", []) if str(item).strip()]
        actions = [str(item).strip() for item in data.get("actions", []) if str(item).strip()]
        investigation = str(data.get("investigation", "") or data.get("investigation_text", "")).strip()
        detailed_usage = str(data.get("detailed_usage", "") or data.get("recommendations_text", "")).strip()
        usage_programs = data.get("usage_programs") or data.get("programs") or data.get("usage_details")

        if not summary and not insights and not actions and not detailed_usage and not investigation and not usage_programs:
            return None

        return {
            "summary": summary,
            "insights": insights,
            "actions": actions,
            "investigation": investigation,
            "detailed_usage": detailed_usage,
            "usage_programs": usage_programs,
        }


    def _format_parcel_summary(self, summary: Optional[Dict[str, Any]]) -> str:
        if not summary:
            return "반경 내 필지 데이터가 충분하지 않습니다."

        bucket_counts = summary.get("bucket_counts", {})
        bucket_line = ", ".join(
            f"{label} {bucket_counts.get(label, 0)}개"
            for label in ["소형", "중형", "대형", "초대형"]
            if bucket_counts.get(label)
        )

        lines = [
            f"총 {summary.get('total_count', 0)}개 필지, 평균 면적 약 {summary.get('average_area', 0):.0f}㎡",
        ]
        if bucket_line:
            lines.append(f"면적 분포: {bucket_line}")

        top_land_uses = summary.get("top_land_uses") or []
        if top_land_uses:
            uses_text = ", ".join(
                f"{item.get('use')} {item.get('count')}개"
                for item in top_land_uses
                if item.get("use")
            )
            if uses_text:
                lines.append(f"주요 지목: {uses_text}")

        closest = summary.get("closest") or {}
        distance = closest.get("distance_m")
        if distance:
            label = closest.get("label") or "가장 인접 필지"
            lines.append(f"지도 중심과 {distance:.0f}m 거리의 {label}")

        return "\n".join(lines)

    def _fallback_report(
        self,
        station: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        parcel_summary: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """LLM 호출이 실패했을 때의 기본 보고서."""

        name = station.get("상호") or station.get("name") or "해당 주유소"
        address = station.get("주소") or station.get("address") or "-"
        land_use = station.get("용도지역") or station.get("토지용도") or station.get("지목") or "정보 없음"
        area = station.get("대지면적") or station.get("면적") or station.get("AREA")

        summary_parts = [
            f"{name}({address}) 부지에 대한 기초 입지 진단입니다.",
            f"주요 용도지역은 '{land_use}'로 파악되며 주변 토지이용과의 연계를 고려해야 합니다.",
        ]
        if area:
            summary_parts.append(f"확인된 대지 면적 정보: {area}.")

        insights: List[str] = []
        if recommendations:
            top_type = (
                recommendations[0].get("type")
                or recommendations[0].get("usage_type")
                or recommendations[0].get("category")
            )
            if top_type:
                insights.append(f"추천 데이터 상 우선 검토가 필요한 용도는 '{top_type}' 유형입니다.")
        insights.append("주변 상권 밀도와 교통 접근성을 정량 분석해 수요 포착 범위를 확장할 필요가 있습니다.")
        insights.append("지자체 개발계획 및 도시재생 사업과의 연계를 검토해 정책 수혜 가능성을 확보해야 합니다.")
        insights.append("기존 주유소 설비 전환 시 공사 기간·안전관리·환경영향을 체계적으로 관리할 필요가 있습니다.")

        actions = [
            "현장 실사를 통해 용도지역·지구단위계획 등 인허가 요건을 세부 확인합니다.",
            "추천 활용 방안 대비 수익성·투자비·수요를 시나리오별로 비교 분석합니다.",
            "지자체 및 인근 이해관계자와의 협력 방안을 마련해 추진 동력을 확보합니다.",
        ]

        return {
            "summary": " ".join(summary_parts),
            "insights": insights,
            "actions": actions,
        }

        if not parcel_summary:
            return None

        total = parcel_summary.get("total_count")
        if not total:
            return None

        average_area = parcel_summary.get("average_area") or 0
        bucket_counts = parcel_summary.get("bucket_counts", {})
        small = bucket_counts.get("소형", 0)
        medium = bucket_counts.get("중형", 0)
        large = bucket_counts.get("대형", 0)
        xlarge = bucket_counts.get("초대형", 0)

        phrases = [
            f"반경 300m 내 필지 {total}개, 평균 면적 약 {average_area:.0f}㎡가 확인됩니다."
        ]

        distribution_bits = []
        if small:
            distribution_bits.append(f"소형 {small}개")
        if medium:
            distribution_bits.append(f"중형 {medium}개")
        if large:
            distribution_bits.append(f"대형 {large}개")
        if xlarge:
            distribution_bits.append(f"초대형 {xlarge}개")
        if distribution_bits:
            phrases.append("면적 분포는 " + ", ".join(distribution_bits) + " 수준입니다.")

        top_land_uses = parcel_summary.get("top_land_uses") or []
        if top_land_uses:
            lead = top_land_uses[0]
            if lead.get("use"):
                phrases.append(
                    f"주요 지목은 '{lead['use']}' 계열이 두드러집니다."
                )

        closest = parcel_summary.get("closest") or {}
        distance = closest.get("distance_m")
        label = closest.get("label")
        if distance:
            phrases.append(
                f"지도 중심과 약 {distance:.0f}m 거리에 위치한 {label or '인접 필지'}가 핵심 앵커로 활용될 수 있습니다."
            )

        return " ".join(phrases)

    def _summarise_station(self, station: Dict[str, Any]) -> str:
        """보고서 프롬프트에 활용할 핵심 정보 정리."""

        keys_of_interest = [
            "상호",
            "주소",
            "지번주소",
            "용도지역",
            "지목",
            "대지면적",
            "연면적",
            "주용도",
            "준공일자",
            "폐업일자",
        ]

        parts = []
        for key in keys_of_interest:
            value = station.get(key)
            if value:
                parts.append(f"{key}: {value}")

        lat = station.get("위도")
        lng = station.get("경도")
        if lat and lng:
            parts.append(f"위치: 위도 {lat}, 경도 {lng}")

        if not parts:
            return "제공된 세부 정보가 거의 없습니다."

        return " | ".join(parts)

    def _summarise_recommendations(self, recommendations: List[Dict[str, Any]]) -> str:
        if not recommendations:
            return "추천 결과 없음"

        lines = []
        for item in recommendations:
            usage = item.get("type") or item.get("usage_type") or item.get("category") or "미정"
            score = item.get("score") or item.get("similarity") or item.get("rank") or item.get("probability")
            description = item.get("description")
            line = usage
            if score is not None:
                try:
                    line += f" (점수: {float(score):.3f})"
                except (TypeError, ValueError):
                    line += f" (점수: {score})"
            if description:
                line += f" - {description}"
            lines.append(line)

        return "\n".join(lines[:5])
    def _build_visual_prompt_section(self, map_images: Dict[str, str]) -> str:
        """
        LLM 프롬프트에 포함할 이미지 컨텍스트를 문자열로 만든다.
        현재는 토큰 절약을 위해 위성사진 + 로드뷰1만 사용한다.
        """
        lines: List[str] = []
        if map_images.get("satellite"):
            lines.append("[위성사진_B64]")
            lines.append(map_images["satellite"])
        if map_images.get("streetview1"):
            lines.append("[로드뷰1_B64]")
            lines.append(map_images["streetview1"])
        return "\n".join(lines)

    def _summarise_stats_for_prompt(self, payload: Optional[Dict[str, Any]]) -> str:
        if not payload:
            return ""
        metrics = (payload or {}).get("metrics") or {}
        relative = (payload or {}).get("relative") or {}

        label_map = {
            "traffic": "일교통량(AADT)",
            "tourism": "관광지수(행정동)",
            "population": "인구수(행정동)",
            "commercial_density": "상권지수",
            "parcel_300m": "반경 300m 필지수",
            "parcel_500m": "반경 500m 필지수",
        }

        def _fmt_value(val: Any) -> str:
            if val is None:
                return "-"
            try:
                num = float(val)
            except (TypeError, ValueError):
                return str(val)
            if abs(num) < 1:
                return f"{num:.3f}"
            if abs(num) < 1000:
                return f"{num:,.0f}"
            return f"{num:,.0f}"

        lines: List[str] = []
        for key, label in label_map.items():
            value_text = _fmt_value(metrics.get(key))
            rel_val = relative.get(key)
            rel_text = "-"
            if self._is_number(rel_val):
                rel_text = f"{float(rel_val):+.1f}%"
            lines.append(f"- {label}: {value_text} (상대값: {rel_text})")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 보고서 HTML 빌더
    # ------------------------------------------------------------------
    def build_report_html(
        self,
        *,
        station: Dict[str, Any],
        report_date: Optional[datetime] = None,
        map_html: str,
        terrain_html: str,
        llm_report: Optional[Dict[str, Any]],
        recommendations: List[Dict[str, Any]],
        stats_payload: Optional[Dict[str, Any]] = None,
        parcel_summary: Optional[Dict[str, Any]] = None,
        land_payload: Optional[Dict[str, Any]] = None,
        nearby_parcels_available: bool = False,
        map_images: Optional[Dict[str, str]] = None,
    ) -> str:
        """주어진 데이터로 폐·휴업 주유소 실태조사 보고서 HTML을 만든다."""

        report_date = report_date or datetime.now()
        station_name = station.get("상호") or station.get("name") or "주유소"
        address = station.get("주소") or station.get("지번주소") or "주소 정보 없음"
        lat = station.get("위도")
        lng = station.get("경도")

        summary_text = ""
        insights: List[str] = []
        actions: List[str] = []
        detailed_usage_text = ""
        investigation_raw = ""

        if isinstance(llm_report, dict):
            summary_text = llm_report.get("summary") or ""
            insights = llm_report.get("insights") or []
            actions = llm_report.get("actions") or []
            investigation_raw = llm_report.get("investigation") or ""
            detailed_usage_text = (
                llm_report.get("detailed_usage")
                or llm_report.get("recommendations_text")
                or ""
            )
            usage_programs = llm_report.get("usage_programs")
        else:
            usage_programs = None

        environment_text = summary_text or "LLM 분석 결과를 불러오지 못했습니다. 기본 현황을 참고하세요."
        investigation_text = investigation_raw or self._compose_investigation_section(insights, actions)
        investigation_text = self._format_investigation_text(investigation_text)

        recommendation_html = self._render_rank_cards_structured(
            recommendations,
            usage_programs,
            fallback_reason_text=(detailed_usage_text or None),
        )


        stats_section = self._compose_stats_section(stats_payload)

        land_price = (land_payload or {}).get("land_price") or {}
        announce_date = land_price.get("announce_date") or "-"
        land_price_text = land_price.get("price_str") or "-"

        land_use_raw = ((land_payload or {}).get("land_use") or {}).get("raw") or []
        land_use_names: List[str] = []
        for item in land_use_raw:
            land_use_name = str(item.get("name", "")).strip()
            if land_use_name and land_use_name not in land_use_names:
                land_use_names.append(land_use_name)
            if len(land_use_names) >= 5:
                break
        land_use_text = ", ".join(land_use_names) if land_use_names else "지목 정보 없음"

        coords_text = f"위도 {lat}, 경도 {lng}" if lat and lng else "좌표 정보 없음"

        map_images = map_images or {}

        # Google Maps API를 서버에서 호출하여 위성/로드뷰 이미지를 Base64로 생성
        satellite_img = ""
        streetview1_img = ""
        streetview2_img = ""

        if map_images.get("satellite"):
            satellite_img = (
                f'<img src="data:image/jpeg;base64,{map_images["satellite"]}" '
                f'alt="위성사진" style="width:100%; height:100%; '
                f'object-fit:cover; border-radius:10px;">'
            )
        if map_images.get("streetview1"):
            streetview1_img = (
                f'<img src="data:image/jpeg;base64,{map_images["streetview1"]}" '
                f'alt="현장사진(로드뷰1)" style="width:100%; height:100%; '
                f'object-fit:cover; border-radius:10px;">'
            )
        if map_images.get("streetview2"):
            streetview2_img = (
                f'<img src="data:image/jpeg;base64,{map_images["streetview2"]}" '
                f'alt="현장사진(로드뷰2)" style="width:100%; height:100%; '
                f'object-fit:cover; border-radius:10px;">'
            )

        if lat and lng and self.google_maps_api_key:
            try:
                if not satellite_img:
                    sat_b64 = self._fetch_satellite_image_b64(float(lat), float(lng), width=600, height=450, zoom=18)
                    if sat_b64:
                        satellite_img = (
                            f'<img src="data:image/jpeg;base64,{sat_b64}" '
                            f'alt="위성사진" style="width:100%; height:100%; '
                            f'object-fit:cover; border-radius:10px;">'
                        )
            except Exception as e:
                print(f"[Satellite] 이미지 생성 실패: {e}")

            try:
                if not streetview1_img:
                    rv1_b64 = self._fetch_streetview_image_b64(float(lat), float(lng),
                                                               heading=0, pitch=0,
                                                               width=600, height=450, fov=90)
                    if rv1_b64:
                        streetview1_img = (
                            f'<img src="data:image/jpeg;base64,{rv1_b64}" '
                            f'alt="현장사진(로드뷰1)" style="width:100%; height:100%; '
                            f'object-fit:cover; border-radius:10px;">'
                        )
            except Exception as e:
                print(f"[StreetView1] 이미지 생성 실패: {e}")

            try:
                if not streetview2_img:
                    rv2_b64 = self._fetch_streetview_image_b64(float(lat), float(lng),
                                                               heading=180, pitch=0,
                                                               width=600, height=450, fov=90)
                    if rv2_b64:
                        streetview2_img = (
                            f'<img src="data:image/jpeg;base64,{rv2_b64}" '
                            f'alt="현장사진(로드뷰2)" style="width:100%; height:100%; '
                            f'object-fit:cover; border-radius:10px;">'
                        )
            except Exception as e:
                print(f"[StreetView2] 이미지 생성 실패: {e}")

        # 기본값으로 terrain_html 사용 (위성사진이 없을 경우)
        if not satellite_img:
            satellite_img = terrain_html if terrain_html else '<div class="placeholder">위성사진</div>'

        # 로드뷰가 없을 경우 placeholder 사용
        if not streetview1_img:
            streetview1_img = '<div class="placeholder">현장사진(로드뷰1)</div>'
        if not streetview2_img:
            streetview2_img = '<div class="placeholder">현장사진(로드뷰2)</div>'


        return f"""
        <!DOCTYPE html>
        <html lang=\"ko\">
        <head>
            <meta charset=\"utf-8\">
            <title>폐·휴업주유소실태조사보고서 - {station_name}</title>
            <style>
                * {{ box-sizing: border-box; }}
                body {{
                    margin: 0;
                    padding: 24px;
                    font-family: 'Noto Sans KR', 'Pretendard', Arial, sans-serif;
                    background: #f5f6fa;
                    color: #1f2937;
                }}
                .report {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: #fff;
                    padding: 80px 80px 48px;      
                    padding-left: 100px;
                    padding-right: 100px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.08);
                    border-radius: 16px;
                    border: 1px solid #e5e7eb;
                }}
                .title {{
                    text-align: center;
                    margin-bottom: 12px;
                    font-size: 28px;
                    letter-spacing: -0.02em;
                    font-weight: 800;
                }}
                .date {{ text-align: center; color: #6b7280; margin-bottom: 24px; }}
                .section {{ margin-top: 22px; }}
                .section h2 {{
                    font-size: 22px;         
                    font-weight: 800;        /* 굵게 */
                    margin: 28px 0 14px 0;
                    padding-bottom: 6px;
                    display: inline-block;
                }}

                .basic-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 14px;
                }}
                .basic-table th,
                .basic-table td {{
                    border: 1px solid #111827;
                    padding: 8px 10px;
                    vertical-align: top;
                }}
                .basic-table th.label {{
                    background: #f7f7f7;
                    width: 120px;
                    text-align: center;
                    vertical-align: middle;
                }}
                /* 칼럼 폭 고정 (지도 / 라벨 / 값) */
                .basic-table col.col-location {{ width: 260px; }}
                .basic-table col.col-label    {{ width: 120px; }}
                .basic-table col.col-value    {{ width: auto; }}
                /* 위치도: 정사각형 빈칸 */
                .basic-table .location-box {{
                    padding: 0;
                    background: #f0f2f5;
                    border-radius: 12px;
                    border: 1px solid #d1d5db;
                    position: relative;
                }}
                .basic-table .location-box::before {{
                    content: "";
                    display: block;
                    padding-top: 100%;  /* 1:1 비율 유지 */
                }}

                .placeholder {{
                    height: 220px;
                    border-radius: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: #9ca3af;
                    font-size: 14px;
                }}

                .info-item {{
                    border: 1px solid #e5e7eb;
                    border-radius: 10px;
                    padding: 12px 14px;
                    background: #fafafa;
                }}
                .label {{ font-size: 13px; color: #6b7280; }}
                .value {{ font-size: 15px; margin-top: 6px; font-weight: 600; }}
                .text-box {{
                    border: 1px solid #e5e7eb;
                    border-radius: 10px;
                    padding: 16px 18px;
                    background: #fdfdfd;
                    line-height: 1.75;
                    white-space: pre-line;
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
                }}
                .text-box + .text-box {{
                    margin-top: 12px;
                }}
                .rank-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: 16px;
                }}
                .rank-card {{
                    border: 1px solid #e5e7eb;
                    border-radius: 14px;
                    background: linear-gradient(180deg, #ffffff 0%, #f9fafb 100%);
                    box-shadow: 0 8px 20px rgba(0,0,0,0.04);
                    display: flex;
                    flex-direction: column;
                    padding: 18px;
                    min-height: 200px;
                    gap: 10px;
                }}
                .rank-card-header {{
                    font-size: 15px;
                    font-weight: 800;
                    color: #0f172a;
                    padding-bottom: 6px;
                    border-bottom: 1px solid #e5e7eb;
                }}
                .rank-card-body {{
                    color: #111827;
                    font-size: 14px;
                    line-height: 1.7;
                    display: grid;
                    gap: 12px;
                }}
                .usage-line {{
                    display: flex;
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 6px;
                }}
                .usage-line > div:last-child {{
                    white-space: pre-line;
                    width: 100%;
                }}
                .reason-label {{
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    padding: 4px 10px;
                    border-radius: 999px;
                    background: #487d6a;
                    color: #ffffff;
                    font-weight: 800;
                    font-size: 12px;
                    min-width: 86px;
                }}
                /* 추천 활용안에서 1순위/2순위/3순위 제목 */
                .rank-title {{
                    display: inline-block;
                    font-size: 16px;
                    font-weight: 800;
                    color: #111827;
                }}
                .text-box .rank-title {{
                    display: block;
                    margin-top: 24px;
                    margin-bottom: 4px;
                    font-size: 16px;
                    font-weight: 700;
                    color: #111827;
                }}

                /* 첫 번째 1순위 제목은 윗 여백 0 */
                .text-box .rank-title:first-of-type {{
                    margin-top: 0;
                }}
                .stats-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
                .stats-table th, .stats-table td {{
                    border: 1px solid #e5e7eb;
                    padding: 8px 10px;
                    text-align: left;
                }}
                .stats-table th {{ background: #f3f4f6; }}
                                /* 리포트용 요약 지표 박스 */
                .metrics-section {{
                    border: 1px solid #e5e7eb;
                    border-radius: 12px;
                    padding: 18px 18px 16px;
                    background: #f9fafb;
                    display: grid;
                    gap: 14px;
                }}
                .metrics-header {{
                    display: grid;
                    gap: 8px;
                }}
                .metrics-title {{
                    margin: 0;
                    font-size: 17px;
                    font-weight: 700;
                    color: #111827;
                }}
                .metrics-hint {{
                    margin: 0;
                    color: #6b7280;
                    font-size: 13px;
                }}
                .metrics-body {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 14px;
                    align-items: stretch;
                }}
                .metrics-column {{
                    display: flex;
                    flex-direction: column;
                    height: 100%;
                }}
                .metrics-list {{
                    display: grid;
                    gap: 8px;
                }}
                .metric-row {{
                    display: flex;
                    justify-content: space-between;
                    padding: 8px 10px;
                    background: #fff;
                    border: 1px solid #e5e7eb;
                    border-radius: 8px;
                    font-size: 13px;
                }}
                .metric-label {{ font-weight: 600; color: #111827; }}
                .metric-value {{ color: #374151; }}
                .radar-card {{
                    width: 100%;
                    border: 1px dashed #d1d5db;
                    background: #fff;
                    border-radius: 10px;
                    padding: 10px;
                    display: grid;
                    gap: 6px;
                    flex: 1;
                }}
                .radar-title {{
                    font-size: 14px;
                    font-weight: 700;
                    color: #111827;
                    text-align: center;
                }}
                .radar-chart {{
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 6px;
                }}
                .radar-chart svg {{ width: 100%; max-width: 210px; height: auto; overflow: visible; }}
                .radar-bg-line {{
                    fill: none;
                    stroke: #e5e7eb;
                    stroke-width: 1;
                }}
                .radar-axis {{
                    stroke: #e5e7eb;
                    stroke-width: 1;
                }}
                .radar-area {{
                    fill: rgba(37, 99, 235, 0.25);
                    stroke: #2563eb;
                    stroke-width: 2;
                }}
                .radar-label {{
                    font-size: 11px;
                    fill: #4b5563;
                    text-anchor: middle;
                }}
                .metric-chart {{
                    border: 1px dashed #d1d5db;
                    background: #fff;
                    border-radius: 10px;
                    padding: 12px 12px 10px;
                }}
                .bar-chart {{
                    display: grid;
                    grid-template-columns: repeat(6, minmax(0, 1fr));
                    gap: 12px;
                    align-items: stretch;
                    padding: 6px 4px 2px;
                }}
                .bar-wrapper {{
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    font-size: 12px;
                    color: #374151;
                    gap: 4px;
                    min-width: 0;
                }}
                .bar-area {{
                    position: relative;
                    width: 100%;
                    max-width: 42px;
                    height: 190px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .bar-axis {{
                    position: absolute;
                    left: 0;
                    right: 0;
                    top: 50%;
                    height: 1px;
                }}
                .bar {{
                    position: absolute;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 100%;
                    max-width: 30px;
                    border-radius: 8px 8px 4px 4px;
                    background: linear-gradient(180deg, #60a5fa, #2563eb);
                    bottom: 50%;
                    max-height: 90px;
                    transition: height 0.2s ease;
                }}
                .bar.negative {{
                    background: linear-gradient(180deg, #fca5a5, #ef4444);
                    top: 50%;
                    bottom: auto;
                    border-radius: 4px 4px 8px 8px;
                }}
                .bar-label {{
                    text-align: center;
                    line-height: 1.3;
                    word-break: keep-all;
                }}
                .bar-value {{
                    font-weight: 700;
                    color: #111827;
                    text-align: center;
                }}
                .bar-value.positive {{ color: #1d4ed8; }}
                .bar-value.negative {{ color: #dc2626; }}
                .subtle {{ color: #9ca3af; font-size: 13px; }}
                .photo-grid {{
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);  /* 3등분 */
                    gap: 10px;
                    margin-top: 12px;
                }}

                .photo-item {{
                    border: 1px solid #e5e7eb;
                    border-radius: 10px;
                    background: #fafafa;
                    height: 260px;                /* 고정 높이 */
                    overflow: hidden;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}

                .photo-caption {{
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    text-align: center;
                    font-size: 14px;
                    margin-top: 6px;
                    color: #6b7280;
                }}
                .print-button {{
                    position: fixed;
                    top: 16px;
                    right: 16px;
                    z-index: 1000;
                    padding: 8px 14px;
                    border-radius: 999px;
                    border: 1px solid #d1d5db;
                    background: #ffffff;
                    cursor: pointer;
                    font-size: 13px;
                    font-weight: 600;
                    color: #374151;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.12);
                }}
                .print-button:hover {{
                    background: #f3f4f6;
                }}
                /* 👉 인쇄(PDF)용 스타일 */
                @media print {{
                    .bar,
                    .bar.negative {{
                        -webkit-print-color-adjust: exact;
                        print-color-adjust: exact;
                    }}
                    .print-button {{
                        display: none;
                    }}
                    body {{
                        margin: 0;
                        padding: 0;
                        background: #ffffff;
                    }}
                    .report {{
                        box-shadow: none;
                        border: none;
                        margin: 0;
                        border-radius: 0;
                        padding: 40px 48px;
                    }}
                }}
                .basic-table .location-map-inner {{
                    position: absolute;
                    inset: 8px;
                    border-radius: 10px;
                    overflow: hidden;
                }}

                .basic-table .location-map-inner #report-map {{
                    width: 100%;
                    height: 100%;
                }}
                #report-map a[href^="http://map.kakao.com"],
                #report-map img[src*="m_bi_b.png"] {{
                    display: none !important;
                }}
            </style>
        </head>
        <body>
            <button class="print-button" onclick="window.print()">PDF 출력</button>

            <article class=\"report\">
                <header>
                    <div class=\"title\">폐·휴업주유소실태조사보고서</div>
                    <div class=\"date\">작성일시: {report_date.strftime('%Y-%m-%d %H:%M')}</div>
                </header>

                <section class="section">
                    <h2>1. 기본 정보</h2>
                    <table class="basic-table">
                        <colgroup>
                            <col class="col-location">
                            <col class="col-label">
                            <col class="col-value">
                            <col class="col-label">
                            <col class="col-value">
                        </colgroup>
                        <tr>
                            <!-- 위치도 들어갈 정사각형 칸 -->
                            <td class="location-box" rowspan="6">
                                <div class="location-map-inner">
                                    {map_html}
                                </div>
                            </td>
                            <th class="label">주유소 이름</th>
                            <td colspan="3">{station_name}</td>
                        </tr>
                        <tr>
                            <th class="label">상태</th>
                            <td colspan="3">폐업</td>
                        </tr>
                        <tr>
                            <th class="label">소재지</th>
                            <td colspan="3">{address}</td>
                        </tr>
                        <tr>
                            <th class="label">공시일자</th>
                            <td>{announce_date}</td>
                            <th class="label">공시지가</th>
                            <td>{land_price_text}</td>
                        </tr>
                        <tr>
                            <th class="label">지목</th>
                            <td colspan="3">{land_use_text}</td>
                        </tr>
                        <tr>
                            <th class="label">주변환경</th>
                            <td colspan="3">{environment_text}</td>
                        </tr>
                    </table>
                    <p class="subtle">좌표: {coords_text}</p>
                </section>


                <section class=\"section\">
                    <h2>2. 조사 현황</h2>
                    <div class=\"text-box\">{investigation_text}</div>
                </section>

                <section class="section">

                    <div class="photo-grid">
                        <div class="photo-item">{satellite_img}</div>
                        <div class="photo-item">{streetview1_img}</div>
                        <div class="photo-item">{streetview2_img}</div>
                    </div>

                    <div class="photo-caption">
                        <div>위성사진</div>
                        <div>현장사진(로드뷰1)</div>
                        <div>현장사진(로드뷰2)</div>
                    </div>
                </section>

                <section class=\"section\">
                    <h2>3. 분석 지표</h2>
                    {stats_section}
                </section>

                <section class=\"section\"> 
                    <h2>4. 추천 활용안</h2>
                    {recommendation_html}
                </section>

                <section class=\"section\">
                    <h2>5. 지적도</h2>
                    <div class=\"info-grid\">
                        <div class=\"info-item\"><div class=\"placeholder\">지적도 이미지</div></div>
                    </div>
                </section>
            </article>
        </body>
        </html>
        """
        """
        '1순위: XXX' 같은 텍스트를
        <span class="rank-title">1순위: XXX</span> 으로 변환해주는 후처리 유틸 함수.
        """
        pattern = r"(\d+순위\s*:\s*[^\n]+)"
        return re.sub(pattern, r'<span class="rank-title">\1</span>', text)


    def _compose_investigation_section(self, insights: List[str], actions: List[str]) -> str:
        bullets: List[str] = []
        if insights:
            bullets.extend(insights)
        if actions:
            bullets.extend(actions)
        if not bullets:
            return "LLM 조사 결과를 수집하지 못했습니다. 현장 확인 후 업데이트하세요."
        return "\n".join(bullets)

    def _format_investigation_text(self, text: str) -> str:
        """조사 현황 텍스트를 불릿 포인트로 포맷팅한다."""
        if not text:
            return "LLM 조사 결과를 수집하지 못했습니다. 현장 확인 후 업데이트하세요."
        
        normalized = re.sub(r"\r\n?", "\n", str(text)).strip()
        lines = [line.strip() for line in normalized.split("\n") if line.strip()]
        
        # 줄바꿈이 없으면 문장 단위로 분리
        if len(lines) == 1:
            # 마침표 기준으로 문장 분리
            sentences = re.split(r'(?<=[.!?])\s+(?=[가-힣A-Z])', lines[0])
            lines = [s.strip() for s in sentences if s.strip()]
        
        if not lines:
            return "LLM 조사 결과를 수집하지 못했습니다. 현장 확인 후 업데이트하세요."
        
        bullet_prefix = re.compile(r"^[•\-\u2022●]\s*")
        formatted_lines = []
        for line in lines:
            cleaned = bullet_prefix.sub("", line).strip()
            formatted_lines.append(f"• {cleaned}")
        
        return "\n".join(formatted_lines)  # \n\n 대신 \n 사용


    def _render_program_entries(
        self, programs: List[Dict[str, Any]], fallback_reason: str
    ) -> str:
        """세부 프로그램명을 안정적으로 카드 형태로 변환한다."""

        entries: List[str] = []

        cleaned_programs: List[Dict[str, str]] = []
        for item in programs or []:
            if not isinstance(item, dict):
                continue

            program_name = (
                str(
                    item.get("name")
                    or item.get("program")
                    or item.get("title")
                    or item.get("label")
                    or ""
                ).strip()
            )
            reason_text = str(
                item.get("reason")
                or item.get("description")
                or item.get("detail")
                or ""
            ).strip()

            if not program_name and not reason_text:
                continue

            cleaned_programs.append(
                {
                    "name": program_name or "세부 프로그램",
                    "reason": reason_text or fallback_reason,
                }
            )

            if len(cleaned_programs) >= 3:
                break

        if not cleaned_programs:
            cleaned_programs = [
                {"name": "세부 프로그램 제안 필요", "reason": fallback_reason}
            ]

        for entry in cleaned_programs:
            entries.append(
                "".join(
                    [
                        '<div class="usage-line">',
                        f'<div class="reason-label">{entry["name"]}</div>',
                        f'<div>• {entry["reason"]}</div>',
                        "</div>",
                    ]
                )
            )

        return "".join(entries)

    def _normalise_usage_programs(
        self, raw_data: Any
    ) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[int, List[Dict[str, Any]]]]:
        """LLM이 반환한 usage_programs를 용도/순위 기준으로 정리한다."""

        usage_map: Dict[str, List[Dict[str, Any]]] = {}
        rank_map: Dict[int, List[Dict[str, Any]]] = {}

        def _coerce_programs(value: Any) -> List[Dict[str, Any]]:
            if isinstance(value, dict):
                value = value.get("programs") or value.get("details") or value.get("items")

            if not isinstance(value, list):
                return []

            programs: List[Dict[str, Any]] = []
            for item in value:
                if isinstance(item, dict):
                    programs.append(item)
            return programs

        if isinstance(raw_data, dict):
            for key, value in raw_data.items():
                programs = _coerce_programs(value)
                if not programs:
                    continue

                if self._is_number(key):
                    try:
                        rank_map[int(float(key))] = programs
                    except Exception:
                        continue
                else:
                    usage_map[str(key).strip().lower()] = programs

        elif isinstance(raw_data, list):
            for item in raw_data:
                if not isinstance(item, dict):
                    continue

                programs = _coerce_programs(
                    item.get("programs")
                    or item.get("items")
                    or item.get("details")
                    or item.get("sub_programs")
                )
                if not programs:
                    continue

                usage_key = (
                    item.get("usage")
                    or item.get("usage_type")
                    or item.get("category")
                    or item.get("title")
                )
                rank_key = item.get("rank")

                if usage_key:
                    usage_map[str(usage_key).strip().lower()] = programs

                if self._is_number(rank_key):
                    try:
                        rank_map[int(float(rank_key))] = programs
                    except Exception:
                        continue

        return usage_map, rank_map

    def _render_rank_cards_structured(
        self,
        recommendations: List[Dict[str, Any]],
        usage_programs: Any,
        *,
        fallback_reason_text: Optional[str] = None,
    ) -> str:
        """추천 API 순위에 맞춰 카드형 활용안을 안정적으로 렌더링한다."""

        if not recommendations:
            return '<div class="text-box">추천 활용안을 불러오지 못했습니다.</div>'

        programs_by_usage, programs_by_rank = self._normalise_usage_programs(usage_programs)

        cards: List[str] = []
        for idx, item in enumerate(recommendations[:3], start=1):
            usage = (
                item.get("type")
                or item.get("usage_type")
                or item.get("category")
                or "제안 용도"
            )

            normalized_usage = str(usage).strip().lower()
            programs = programs_by_usage.get(normalized_usage) or programs_by_rank.get(idx) or []
            fallback_reason = (
                (item.get("detail") or item.get("description") or item.get("note") or "").strip()
                or (fallback_reason_text or "세부 선정 이유는 추후 보완이 필요합니다.")
            )

            body_html = self._render_program_entries(programs, fallback_reason)
            title = f"{idx}순위: {usage}"
            cards.append(
                f"<div class=\"rank-card\">"
                f"<div class=\"rank-card-header\"><span class=\"rank-title\">{title}</span></div>"
                f"<div class=\"rank-card-body\">{body_html}</div>"
                f"</div>"
            )

        if not cards:
            return '<div class="text-box">추천 활용안을 불러오지 못했습니다.</div>'

        return '<div class="rank-grid">' + "".join(cards) + "</div>"



    def _compose_stats_section(self, payload: Optional[Dict[str, Any]]) -> str:
        metrics = (payload or {}).get("metrics") or {}
        relative = (payload or {}).get("relative") or {}

        label_map = {
            "traffic": "일교통량(AADT)",
            "tourism": "관광지수(행정동)",
            "population": "인구수(행정동)",
            "commercial_density": "상권지수",
            "parcel_300m": "반경 300m 필지수",
            "parcel_500m": "반경 500m 필지수",
        }

        def _fmt_value(val: Any) -> str:
            if val is None:
                return "-"
            try:
                num = float(val)
            except (TypeError, ValueError):
                return str(val)
            if abs(num) < 1:
                return f"{num:.3f}"
            if abs(num) < 1000:
                return f"{num:,.0f}"
            return f"{num:,.0f}"

        metric_rows: List[str] = []
        chart_bars: List[str] = []
        radar_values: List[float] = []

        ordered_keys = list(label_map.keys())
        max_relative = max(
            (abs(float(v)) for v in relative.values() if self._is_number(v)),
            default=0.0,
        )
        scale = max_relative if max_relative > 0 else 1.0

        for key in ordered_keys:
            value = _fmt_value(metrics.get(key))
            label = label_map[key]
            metric_rows.append(
                f"""
                <div class="metric-row">
                    <span class="metric-label">{label}</span>
                    <span class="metric-value">{value}</span>
                </div>
                """
            )

            rel_val = relative.get(key)
            bar_height = 0
            rel_text = "-"
            bar_class = "bar"
            value_class = "bar-value"
            radar_point = 50.0
            if self._is_number(rel_val):
                rel_num = float(rel_val)
                rel_text = f"{rel_num:+.1f}%"
                bar_height = min(max((abs(rel_num) / scale) * 90, 0), 90)
                radar_point = max(0.0, min(100.0, 50 + rel_num / 2))
                if rel_num < 0:
                    bar_class += " negative"
                    value_class += " negative"
                else:
                    value_class += " positive"
            radar_values.append(radar_point)

            chart_bars.append(
                f"""
                <div class="bar-wrapper">
                    <div class="{value_class}">{rel_text}</div>
                    <div class="bar-area">
                        <div class="bar-axis"></div>
                        <div class="{bar_class}" style="height:{round(bar_height, 1)}px" title="{label} 상대값 {rel_text}"></div>
                    </div>
                    <div class="bar-label">{label}</div>
                </div>
                """
            )

        if not metric_rows:
            return '<div class="text-box">분석 지표를 불러오지 못했습니다.</div>'

        def _polar_to_cart(index: int, value: float, total: int) -> str:
            angle = -math.pi / 2 + 2 * math.pi * index / total
            radius = 90
            center = 120
            r = (value / 100) * radius
            x = center + r * math.cos(angle)
            y = center + r * math.sin(angle)
            return f"{x:.1f},{y:.1f}"

        def _label_position(index: int, total: int) -> str:
            angle = -math.pi / 2 + 2 * math.pi * index / total
            radius = 115
            center = 120
            x = center + radius * math.cos(angle)
            y = center + radius * math.sin(angle)
            return f"{x:.1f},{y:.1f}"

        radar_points = " ".join(
            _polar_to_cart(idx, val, len(ordered_keys))
            for idx, val in enumerate(radar_values)
        )

        grid_levels = [100, 70, 40]
        grid_polygons = "".join(
            (
                "<polygon class=\"radar-bg-line\" points=\""
                + " ".join(
                    _polar_to_cart(idx, level, len(ordered_keys))
                    for idx in range(len(ordered_keys))
                )
                + "\"></polygon>"
            )
            for level in grid_levels
        )

        axis_lines = "".join(
            f"<line class=\"radar-axis\" x1=\"120\" y1=\"120\" x2=\"{_polar_to_cart(idx, 100, len(ordered_keys)).split(',')[0]}\" y2=\"{_polar_to_cart(idx, 100, len(ordered_keys)).split(',')[1]}\"></line>"
            for idx in range(len(ordered_keys))
        )

        labels_svg = "".join(
            f"<text class=\"radar-label\" x=\"{_label_position(idx, len(ordered_keys)).split(',')[0]}\" y=\"{_label_position(idx, len(ordered_keys)).split(',')[1]}\">{label_map[key]}</text>"
            for idx, key in enumerate(ordered_keys)
        )

        radar_svg = f"""
            <svg viewBox=\"0 0 240 240\" aria-label=\"입지 지표 레이더 차트\">
                {grid_polygons}
                {axis_lines}
                <polygon class=\"radar-area\" points=\"{radar_points}\"></polygon>
                {labels_svg}
            </svg>
        """

        return """
        <section class="metrics-section">
            <div class="metrics-header">
                <h3 class="metrics-title">지표 요약</h3>
                <p class="metrics-hint">주요 입지 지표와 지역 평균 대비 상대값을 함께 확인하세요.</p>
                <div class="metrics-list">{metrics_list}</div>
            </div>
            <div class="metrics-body">
                <div class="metrics-column">
                    <div class="radar-card">
                        <div class="radar-title">입지 지표 레이더</div>
                        <div class="radar-chart">{radar_svg}</div>
                    </div>
                </div>
                <div class="metric-chart">
                    <div class="bar-chart">{bars}</div>
                    <div class="subtle" style="margin-top:10px; text-align:center;">상대값(%) 기준 차트</div>
                </div>
            </div>
        </section>
        """.format(
            metrics_list="".join(metric_rows),
            bars="".join(chart_bars),
            radar_svg=radar_svg,
        )

    @staticmethod
    def _is_number(val: Any) -> bool:
        try:
            float(val)
            return True
        except (TypeError, ValueError):
            return False

    def _fetch_satellite_image_b64(
        self,
        lat: float,
        lng: float,
        width: int = 640,
        height: int = 480,
        zoom: int = 18,
    ) -> Optional[str]:
        """
        Google Maps Static API를 호출해 위성사진을 받아서 Base64 문자열로 반환.
        """
        if not self.google_maps_api_key:
            return None

        base_url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            "center": f"{lat},{lng}",
            "zoom": str(zoom),
            "size": f"{width}x{height}",
            "maptype": "satellite",
            "key": self.google_maps_api_key,
        }

        try:
            r = requests.get(base_url, params=params, timeout=10)
            r.raise_for_status()
            return base64.b64encode(r.content).decode("utf-8")
        except Exception as e:
            print(f"[StaticMap] 호출 실패: {e}")
            return None

    def _fetch_streetview_image_b64(
        self,
        lat: float,
        lng: float,
        heading: int = 0,
        pitch: int = 0,
        width: int = 640,
        height: int = 480,
        fov: int = 90,
    ) -> Optional[str]:
        """
        Google Street View Static API를 호출해 로드뷰 이미지를 Base64 문자열로 반환.
        """
        if not self.google_maps_api_key:
            return None

        base_url = "https://maps.googleapis.com/maps/api/streetview"
        params = {
            "location": f"{lat},{lng}",
            "size": f"{width}x{height}",
            "heading": str(heading),
            "pitch": str(pitch),
            "fov": str(fov),
            "key": self.google_maps_api_key,
        }

        try:
            r = requests.get(base_url, params=params, timeout=10)
            r.raise_for_status()
            return base64.b64encode(r.content).decode("utf-8")
        except Exception as e:
            print(f"[StreetView] 호출 실패(heading={heading}): {e}")
            return None

        """
        Google Street View Static API를 호출해 로드뷰 이미지를 Base64 문자열로 반환.
        """
        if not self.google_maps_api_key:
            return None

        base_url = "https://maps.googleapis.com/maps/api/streetview"
        params = {
            "location": f"{lat},{lng}",
            "size": f"{width}x{height}",
            "heading": str(heading),
            "pitch": str(pitch),
            "fov": str(fov),
            "key": self.google_maps_api_key,
        }

        try:
            r = requests.get(base_url, params=params, timeout=10)
            r.raise_for_status()
            return base64.b64encode(r.content).decode("utf-8")
        except Exception as e:
            print(f"[StreetView] 호출 실패(heading={heading}): {e}")
            return None

        """
        Google Street View Static API를 사용하여 로드뷰 이미지 URL을 생성합니다.

        Args:
            lat: 위도
            lng: 경도
            heading: 카메라 방향 (0-360, 0=북쪽, 90=동쪽, 180=남쪽, 270=서쪽)
            pitch: 카메라 상하 각도 (-90 ~ 90, 0=수평)
            width: 이미지 너비
            height: 이미지 높이
            fov: 시야각 (10-120도)

        Returns:
            로드뷰 이미지 URL
        """
        if not self.google_maps_api_key:
            return ""

        base_url = "https://maps.googleapis.com/maps/api/streetview"
        params = [
            f"location={lat},{lng}",
            f"size={width}x{height}",
            f"heading={heading}",
            f"pitch={pitch}",
            f"fov={fov}",
            f"key={self.google_maps_api_key}",
        ]
        return f"{base_url}?{'&'.join(params)}"