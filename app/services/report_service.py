"""LLM 기반 보고서 생성을 위한 서비스 모듈."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

import httpx
from dotenv import load_dotenv


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
    ) -> Dict[str, Any]:
        """보고서에 포함할 요약/인사이트/실행항목을 반환한다."""

        route_config = self._resolve_route(station_id)
        llm_response = await self._request_llm(
            station,
            recommendations,
            parcel_summary,
            route_config,
            station_id,
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

        user_prompt = (
            "당신은 도시 재생 및 부동산 활용 전략을 제시하는 컨설턴트입니다. 아래 주유소 정보를 분석하여 "
            "입지 특성 요약(2~3문장), 인사이트 3개, 권장 실행 항목 3개, 그리고 세부 추천 활용안을 JSON으로만 응답하세요.\n"
            "JSON 키는 summary(문장), insights(문장 리스트), actions(문장 리스트), detailed_usage(문자열)입니다.\n"
            "detailed_usage는 다음 형식의 한국어 멀티라인 텍스트로 작성합니다.\n"
            "각 순위별로 3개의 세부 프로그램을 제안하고, 각 프로그램 선정 이유를 2~3문장으로 설명하세요.\n"
            "예시 형식:\n"
            "1순위: 근린생활시설\n"
            "- 카페: 선정 이유를 2~3문장으로 서술.\n"
            "- 드라이브스루 매장: 선정 이유를 2~3문장으로 서술.\n"
            "- 공원·휴게공간: 선정 이유를 2~3문장으로 서술.\n"
            "2순위: ...\n"
            "3순위: ...\n"
            "모든 문장은 한국어 비즈니스 보고서 어투로 작성하고, JSON 이외의 다른 설명이나 마크다운은 포함하지 마세요.\n\n"
            f"[대상 주유소] {station_identifier}\n"
            f"[주유소 정보]\n{station_summary}\n\n"
            f"[추천 활용 방안]\n{recommendation_summary}\n"
            f"[반경 300m 필지 통계]\n{parcel_context}\n"
        )

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
        detailed_usage = str(data.get("detailed_usage", "") or data.get("recommendations_text", "")).strip()

        if not summary and not insights and not actions and not detailed_usage:
            return None

        return {
            "summary": summary,
            "insights": insights,
            "actions": actions,
            "detailed_usage": detailed_usage,
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

        parcel_phrase = self._describe_parcel_summary(parcel_summary)
        if parcel_phrase:
            summary_parts.append(parcel_phrase)

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

    def _describe_parcel_summary(
        self, parcel_summary: Optional[Dict[str, Any]]
    ) -> Optional[str]:
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
        nearby_parcels_available: bool = False,
    ) -> str:
        """주어진 데이터로 폐·휴업 주유소 실태조사 보고서 HTML을 만든다."""

        report_date = report_date or datetime.now()
        name = station.get("상호") or station.get("name") or "주유소"
        address = station.get("주소") or station.get("지번주소") or "주소 정보 없음"
        lat = station.get("위도")
        lng = station.get("경도")

        summary_text = ""
        insights: List[str] = []
        actions: List[str] = []
        detailed_usage_text = ""

        if isinstance(llm_report, dict):
            summary_text = llm_report.get("summary") or ""
            insights = llm_report.get("insights") or []
            actions = llm_report.get("actions") or []
            detailed_usage_text = (
                llm_report.get("detailed_usage")
                or llm_report.get("recommendations_text")
                or ""
            )

        environment_text = summary_text or "LLM 분석 결과를 불러오지 못했습니다. 기본 현황을 참고하세요."
        investigation_text = self._compose_investigation_section(insights, actions)

        # LLM이 detailed_usage를 돌려주면 그대로 활용안 섹션에 사용
        if detailed_usage_text:
            # 1·2·3순위 제목을 span.rank-title 로 감싸기
            recommendation_text = self._decorate_rank_titles(detailed_usage_text)
        else:
            # LLM 실패 시에만 간단한 정적 요약 사용
            base_text = self._compose_recommendations(recommendations)
            recommendation_text = self._decorate_rank_titles(base_text)


        stats_section = self._compose_stats_section(stats_payload)
        parcel_text = self._describe_parcel_summary(parcel_summary) or "반경 300m 필지 정보가 확보되지 않았습니다."

        coords_text = f"위도 {lat}, 경도 {lng}" if lat and lng else "좌표 정보 없음"

        # Google Maps API를 사용하여 위성사진 및 로드뷰 이미지 생성
        satellite_img = ""
        streetview1_img = ""
        streetview2_img = ""

        if lat and lng and self.google_maps_api_key:
            # 위성사진 (기존 terrain_html 대신 사용)
            satellite_url = self._get_satellite_image_url(lat, lng, width=600, height=450, zoom=18)
            if satellite_url:
                satellite_img = f'<img src="{satellite_url}" alt="위성사진" style="width:100%; height:100%; object-fit:cover; border-radius:10px;">'

            # 로드뷰 1 (heading=0, 북쪽 방향)
            streetview1_url = self._get_streetview_image_url(lat, lng, heading=0, pitch=0, width=600, height=450, fov=90)
            if streetview1_url:
                streetview1_img = f'<img src="{streetview1_url}" alt="현장사진(로드뷰1)" style="width:100%; height:100%; object-fit:cover; border-radius:10px;">'

            # 로드뷰 2 (heading=180, 남쪽 방향 - 반대편 보기)
            streetview2_url = self._get_streetview_image_url(lat, lng, heading=180, pitch=0, width=600, height=450, fov=90)
            if streetview2_url:
                streetview2_img = f'<img src="{streetview2_url}" alt="현장사진(로드뷰2)" style="width:100%; height:100%; object-fit:cover; border-radius:10px;">'

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
            <title>폐·휴업주유소실태조사보고서 - {name}</title>
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
                    padding: 14px;
                    background: #fdfdfd;
                    line-height: 1.7;
                    white-space: pre-line;
                }}
                /* 추천 활용안에서 1순위/2순위/3순위 제목 */
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
                    padding: 18px;
                    background: #f9fafb;
                    display: grid;
                    gap: 14px;
                }}
                .metrics-header {{
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
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
                    grid-template-columns: 1.1fr 1fr;
                    gap: 14px;
                }}
                .metrics-list {{
                    display: grid;
                    gap: 8px;
                    margin-bottom: 14px;
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


            </style>
        </head>
        <body>
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
                            <td class="location-box" rowspan="6"></td>
                            <th class="label">주유소 이름</th>
                            <td colspan="3">{name}</td>
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
                            <th class="label">면적</th>
                            <td></td>
                            <th class="label">공시지가</th>
                            <td></td>
                        </tr>
                        <tr>
                            <th class="label">지목</th>
                            <td></td>
                            <th class="label">용도지역·지구</th>
                            <td></td>
                        </tr>
                        <tr>
                            <th class="label">주변환경</th>
                            <td colspan="3">{environment_text}</td>
                        </tr>
                    </table>
                    <p class="subtle">좌표: {coords_text} | 반경 300m 필지 현황: {parcel_text}</p>
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
                    <div class=\"text-box\">
                        {recommendation_text}
                        </div>    
                </section>

                <section>
                    <h2>지적도 / 토지이용계획확인원</h2>
                    <div class=\"info-grid\">
                        <div class=\"info-item\"><div class=\"placeholder\">지적도 이미지</div></div>
                        <div class=\"info-item\"><div class=\"placeholder\">토지이용계획확인원</div></div>
                    </div>
                </section>

                <section class=\"section\">
                    <h2>기타 참고</h2>
                    <p class=\"subtle\">주변 필지 데이터: { '확보됨' if nearby_parcels_available else '미확보' }</p>
                </section>
            </article>
        </body>
        </html>
        """
    def _decorate_rank_titles(self, text: str) -> str:
        """
        '1순위: XXX' 같은 텍스트를
        <span class="rank-title">1순위: XXX</span> 으로 변환해주는 후처리 유틸 함수.
        """
        pattern = r"(\d+순위\s*:\s*[^\n]+)"
        return re.sub(pattern, r'<span class="rank-title">\1</span>', text)


    def _compose_investigation_section(self, insights: List[str], actions: List[str]) -> str:
        paragraphs = []
        if insights:
            paragraphs.append("[주요 인사이트]")
            paragraphs.extend(f"- {item}" for item in insights)
        if actions:
            paragraphs.append("\n[권장 조치]")
            paragraphs.extend(f"- {item}" for item in actions)
        if not paragraphs:
            return "LLM 조사 결과를 수집하지 못했습니다. 현장 확인 후 업데이트하세요."
        return "\n".join(paragraphs)

    def _compose_recommendations(self, recommendations: List[Dict[str, Any]]) -> str:
        """
        추천 활용안 섹션 텍스트 구성.
        1순위 / 2순위 / 3순위 제목은 .rank-title 로 감싸서 굵게 표시.
        LLM이 생성한 세부 설명은 그대로 줄바꿈 유지.
        """
        if not recommendations:
            return "추천 활용안을 불러오지 못했습니다."

        blocks: List[str] = []

        for idx, item in enumerate(recommendations[:3], start=1):
            usage = item.get("type") or item.get("usage_type") or item.get("category") or "제안 용도"
            # LLM이 생성한 상세 텍스트(프로그램 + 이유)를 한 덩어리로 받는다고 가정
            detail = (item.get("detail") or item.get("description") or "").strip()

            # 제목은 굵게/크게
            lines: List[str] = [f'<span class="rank-title">{idx}순위: {usage}</span>']

            if detail:
                # 여러 줄이면 그대로 살리되, 앞에 "- " 붙여서 목록처럼 보이게
                for para in detail.split("\n"):
                    p = para.strip()
                    if not p:
                        continue
                    lines.append(f"- {p}")

            blocks.append("\n".join(lines))

        # 4~5순위가 있다면 한 줄로만 추가 검토 용도로 표시
        if len(recommendations) > 3:
            extra = [
                (item.get("type") or item.get("usage_type") or item.get("category"))
                for item in recommendations[3:5]
                if (item.get("type") or item.get("usage_type") or item.get("category"))
            ]
            if extra:
                blocks.append("추가 검토 대상: " + ", ".join(extra))

        # text-box 에서 white-space: pre-line 이라 \n 기준으로 자연스럽게 줄바꿈됨
        return "\n\n".join(blocks)



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
            if self._is_number(rel_val):
                rel_num = float(rel_val)
                rel_text = f"{rel_num:+.1f}%"
                bar_height = min(max((abs(rel_num) / scale) * 90, 0), 90)
                if rel_num < 0:
                    bar_class += " negative"
                    value_class += " negative"
                else:
                    value_class += " positive"


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

        return """
        <section class="metrics-section">
            <div class="metrics-header">
                <h3 class="metrics-title">지표 요약</h3>
                <p class="metrics-hint">주요 입지 지표와 지역 평균 대비 상대값을 함께 확인하세요.</p>
            </div>
            <div class="metrics-body">
                <div class="metrics-list">{metrics_list}</div>
                <div class="metric-chart">
                    <div class="bar-chart">{bars}</div>
                    <div class="subtle" style="margin-top:10px; text-align:center;">상대값(%) 기준 차트</div>
                </div>
            </div>
        </section>
        """.format(metrics_list="".join(metric_rows), bars="".join(chart_bars))

    @staticmethod
    def _is_number(val: Any) -> bool:
        try:
            float(val)
            return True
        except (TypeError, ValueError):
            return False

    def _get_satellite_image_url(self, lat: float, lng: float, width: int = 640, height: int = 480, zoom: int = 18) -> str:
        """
        Google Maps Static API를 사용하여 위성사진 URL을 생성합니다.

        Args:
            lat: 위도
            lng: 경도
            width: 이미지 너비
            height: 이미지 높이
            zoom: 줌 레벨 (1-20)

        Returns:
            위성사진 이미지 URL
        """
        if not self.google_maps_api_key:
            return ""

        base_url = "https://maps.googleapis.com/maps/api/staticmap"
        params = [
            f"center={lat},{lng}",
            f"zoom={zoom}",
            f"size={width}x{height}",
            "maptype=satellite",
            f"key={self.google_maps_api_key}",
        ]
        return f"{base_url}?{'&'.join(params)}"

    def _get_streetview_image_url(
        self, lat: float, lng: float, heading: int = 0, pitch: int = 0,
        width: int = 640, height: int = 480, fov: int = 90
    ) -> str:
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