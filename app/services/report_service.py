"""LLM 기반 보고서 생성을 위한 서비스 모듈."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx


class LLMReportService:
    """LLM을 활용해 주유소 보고서를 생성하는 서비스."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv(
            "LLM_API_URL", "https://api.openai.com/v1/chat/completions"
        )

        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        try:
            default_timeout = float(os.getenv("LLM_TIMEOUT", "30"))
        except ValueError:
            default_timeout = 30.0
        self.timeout = timeout or default_timeout

        self.force_json_response = os.getenv("LLM_FORCE_JSON", "true").lower() != "false"


    async def generate_report(
        self, station: Dict[str, Any], recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """보고서에 포함할 요약/인사이트/실행항목을 반환한다."""

        llm_response = await self._request_llm(station, recommendations)
        if llm_response:
            parsed = self._parse_llm_response(llm_response)
            if parsed:
                return parsed

        return self._fallback_report(station, recommendations)

    async def _request_llm(
        self, station: Dict[str, Any], recommendations: List[Dict[str, Any]]
    ) -> Optional[str]:
        """LLM API 호출. 실패 시 None."""

        if not self.api_key:
            return None

        station_summary = self._summarise_station(station)
        recommendation_summary = self._summarise_recommendations(recommendations)

        user_prompt = (

            "당신은 도시 재생 및 부동산 활용 전략을 제시하는 컨설턴트입니다. 아래 주유소 정보를 분석하여 "
            "입지 특성 요약(2~3문장), 인사이트 3개, 권장 실행 항목 3개를 JSON으로만 응답하세요.\n"
            "JSON 키는 summary(문장), insights(문장 리스트), actions(문장 리스트)입니다.\n"
            "모든 문장은 한국어 비즈니스 보고서 어투로 작성하고, 다른 설명이나 마크다운은 포함하지 마세요.\n\n"
            f"[주유소 정보]\n{station_summary}\n\n"
            f"[추천 활용 방안]\n{recommendation_summary}\n"
        )

        messages = [
            {
                "role": "system",
                "content": "도시 입지 분석을 수행하는 한국어 컨설턴트입니다.",
            },
            {"role": "user", "content": user_prompt},
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
        }

        if self.force_json_response:
            payload["response_format"] = {"type": "json_object"}


        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.base_url, headers=headers, json=payload)
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

        if not summary and not insights and not actions:
            return None

        return {
            "summary": summary,
            "insights": insights,
            "actions": actions,
        }

    def _fallback_report(
        self, station: Dict[str, Any], recommendations: List[Dict[str, Any]]
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
            top_type = recommendations[0].get("type") or recommendations[0].get("usage_type")
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
            usage = item.get("type") or item.get("usage_type") or "미정"
            score = item.get("score") or item.get("similarity") or item.get("rank")
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
