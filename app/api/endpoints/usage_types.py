"""
용도 유형 관련 API 엔드포인트
"""

from fastapi import APIRouter, Depends, Query, HTTPException, Path
from typing import Optional, List, Dict, Any

from app.api.dependencies import get_recommendation_service
from app.services.recommend_service import RecommendationService
from app.schemas.usage_type import UsageTypeList, UsageTypeResponse, UsageTypeCentroidList


router = APIRouter(
    prefix="/api/usage-types",
    tags=["usage_types"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", response_model=UsageTypeList)
async def get_usage_types(
    service: RecommendationService = Depends(get_recommendation_service),
):
    """
    용도 유형 목록 조회 API
    
    사용 가능한 모든 용도 유형(대분류)을 반환합니다.
    """
    try:
        # 센트로이드 데이터에서 고유한 용도 유형 추출
        if not service.centroids is None and "usage_type" in service.centroids.columns:
            usage_types = []
            unique_types = service.centroids["usage_type"].dropna().unique()
            
            for i, usage_type in enumerate(unique_types):
                # 설명 및 예시 (실제로는 더 정교한 데이터가 필요함)
                description = ""
                examples = []
                
                if usage_type == "근린생활시설":
                    description = "일상생활에 필요한 서비스를 제공하는 시설"
                    examples = ["편의점", "카페", "미용실", "세탁소"]
                elif usage_type == "공동주택":
                    description = "여러 세대가 거주하는 주택"
                    examples = ["아파트", "연립주택", "다세대주택"]
                elif usage_type == "자동차관련시설":
                    description = "자동차 관련 서비스를 제공하는 시설"
                    examples = ["정비소", "세차장", "전기차 충전소"]
                elif usage_type == "판매시설":
                    description = "물품을 판매하는 시설"
                    examples = ["소매점", "슈퍼마켓", "약국"]
                
                # 해당 용도 유형의 첫 번째 행의 벡터값 사용
                type_df = service.centroids[service.centroids["usage_type"] == usage_type]
                
                usage_type_item = {
                    "id": i + 1,
                    "name": usage_type,
                    "description": description,
                    "examples": examples,
                }
                
                # 벡터값 추가
                if len(type_df) > 0:
                    for col in service.norm_cols:
                        if col in type_df.columns:
                            usage_type_item[col] = float(type_df[col].iloc[0]) if not pd.isna(type_df[col].iloc[0]) else 0.0
                
                usage_types.append(usage_type_item)
            
            return {"count": len(usage_types), "items": usage_types}
        
        # 대안: 추천 결과 데이터에서 용도 유형 추출
        elif "recommend_result" in service.data and "대분류" in service.data["recommend_result"].columns:
            usage_types = []
            unique_types = service.data["recommend_result"]["대분류"].dropna().unique()
            
            for i, usage_type in enumerate(unique_types):
                usage_types.append({
                    "id": i + 1,
                    "name": usage_type,
                    "description": "",
                    "examples": []
                })
            
            return {"count": len(usage_types), "items": usage_types}
        
        else:
            # 임시 데이터 반환
            return {
                "count": 5,
                "items": [
                    {
                        "id": 1,
                        "name": "근린생활시설",
                        "description": "일상생활에 필요한 서비스를 제공하는 시설",
                        "examples": ["편의점", "카페", "미용실", "세탁소"]
                    },
                    {
                        "id": 2,
                        "name": "공동주택",
                        "description": "여러 세대가 거주하는 주택",
                        "examples": ["아파트", "연립주택", "다세대주택"]
                    },
                    {
                        "id": 3,
                        "name": "자동차관련시설",
                        "description": "자동차 관련 서비스를 제공하는 시설",
                        "examples": ["정비소", "세차장", "전기차 충전소"]
                    },
                    {
                        "id": 4,
                        "name": "판매시설",
                        "description": "물품을 판매하는 시설",
                        "examples": ["소매점", "슈퍼마켓", "약국"]
                    },
                    {
                        "id": 5,
                        "name": "업무시설",
                        "description": "업무를 수행하기 위한 시설",
                        "examples": ["사무실", "금융기관", "콜센터"]
                    }
                ]
            }
    except Exception as e:
        print(f"⚠️ 용도 유형 목록 API 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"용도 유형 목록 조회 중 오류가 발생했습니다: {str(e)}")


@router.get("/centroids", response_model=UsageTypeCentroidList)
async def get_usage_type_centroids(
    region: Optional[str] = Query(None, description="권역 필터"),
    service: RecommendationService = Depends(get_recommendation_service),
):
    """
    용도 유형 센트로이드 목록 조회 API
    
    용도 유형(대분류)별 센트로이드 벡터를 반환합니다.
    
    - **region**: 권역 필터 (선택)
    """
    try:
        if service.centroids is None:
            return {"count": 0, "items": []}
        
        # 센트로이드 데이터 복사
        centroids = service.centroids.copy()
        
        # 권역 필터 적용
        if region and "region" in centroids.columns:
            import re
            pattern = re.compile(f".*{region}.*", re.IGNORECASE)
            centroids = centroids[centroids["region"].str.match(pattern, na=False)]
        
        # 결과 형식화
        result = []
        
        for _, centroid in centroids.iterrows():
            centroid_item = {
                "usage_type": centroid.get("usage_type", ""),
                "region": centroid.get("region", "")
            }
            
            # 벡터값 추가
            for col in service.norm_cols:
                if col in centroids.columns:
                    centroid_item[col] = float(centroid.get(col, 0)) if not pd.isna(centroid.get(col, 0)) else 0.0
            
            result.append(centroid_item)
        
        return {"count": len(result), "items": result}
    except Exception as e:
        print(f"⚠️ 용도 유형 센트로이드 API 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"용도 유형 센트로이드 조회 중 오류가 발생했습니다: {str(e)}")


@router.get("/{usage_type}", response_model=UsageTypeResponse)
async def get_usage_type(
    usage_type: str = Path(..., description="용도 유형명"),
    service: RecommendationService = Depends(get_recommendation_service),
):
    """
    개별 용도 유형 상세 정보 API
    
    - **usage_type**: 용도 유형명 (필수)
    """
    try:
        # 센트로이드 데이터에서 용도 유형 검색
        if not service.centroids is None and "usage_type" in service.centroids.columns:
            # 대소문자 무시 검색
            import re
            pattern = re.compile(f"^{usage_type}$", re.IGNORECASE)
            type_df = service.centroids[service.centroids["usage_type"].str.match(pattern, na=False)]
            
            if len(type_df) > 0:
                # 첫 번째 행 사용
                centroid = type_df.iloc[0]
                
                # 설명 및 예시 (실제로는 더 정교한 데이터가 필요함)
                description = ""
                examples = []
                
                if usage_type.lower() == "근린생활시설":
                    description = "일상생활에 필요한 서비스를 제공하는 시설"
                    examples = ["편의점", "카페", "미용실", "세탁소"]
                elif usage_type.lower() == "공동주택":
                    description = "여러 세대가 거주하는 주택"
                    examples = ["아파트", "연립주택", "다세대주택"]
                elif usage_type.lower() == "자동차관련시설":
                    description = "자동차 관련 서비스를 제공하는 시설"
                    examples = ["정비소", "세차장", "전기차 충전소"]
                elif usage_type.lower() == "판매시설":
                    description = "물품을 판매하는 시설"
                    examples = ["소매점", "슈퍼마켓", "약국"]
                
                result = {
                    "id": 1,
                    "name": centroid.get("usage_type", usage_type),
                    "description": description,
                    "examples": examples,
                }
                
                # 벡터값 추가
                for col in service.norm_cols:
                    if col in centroid.index:
                        result[col] = float(centroid.get(col, 0)) if not pd.isna(centroid.get(col, 0)) else 0.0
                
                return result
        
        # 용도 유형을 찾지 못한 경우
        raise HTTPException(status_code=404, detail=f"용도 유형 '{usage_type}'을(를) 찾을 수 없습니다.")
    except HTTPException:
        raise
    except Exception as e:
        print(f"⚠️ 용도 유형 상세 API 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"용도 유형 상세 조회 중 오류가 발생했습니다: {str(e)}")
