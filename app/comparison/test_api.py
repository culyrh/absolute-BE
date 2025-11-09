"""
알고리즘 테스트용 API 엔드포인트
app/comparison/test_api.py

실행:
    main.py에서 자동으로 로드됨 (ENABLE_TESTING=true)
"""

from fastapi import APIRouter, Query, Body, HTTPException
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

from app.api.dependencies import get_recommendation_service
from app.services.recommend_service import RecommendationService

from app.comparison.algorithms.cosine_similarity import CosineSimilarityAlgorithm
from app.comparison.algorithms.euclidean_distance import EuclideanDistanceAlgorithm
from app.comparison.algorithms.pearson_correlation import PearsonCorrelationAlgorithm
from app.comparison.algorithms.popularity import PopularityAlgorithm
from app.comparison.algorithms.collaborative import CollaborativeAlgorithm
from app.comparison.algorithms.ahp_topsis import AHPTopsisAlgorithm


router = APIRouter(
    prefix="/api/test",
    tags=["algorithm_testing"],
    responses={404: {"description": "Not found"}},
)


@router.post("/compare-algorithms")
async def compare_algorithms(
    query: str = Body(..., description="검색 주소"),
    algorithms: Optional[List[str]] = Body(
        default=None, 
        description="테스트할 알고리즘 리스트"
    ),
    top_k: int = Body(default=10, ge=1, le=100),
    service: RecommendationService = Depends(get_recommendation_service),
):
    """
    여러 알고리즘 동시 비교
    
    **사용 예시**:
    ```json
    {
      "query": "서울 강남구",
      "algorithms": ["cosine_similarity", "ahp_topsis"],
      "top_k": 10
    }
    ```
    """
    try:
        # 주소 검색
        gas_df = service.data["gas_station"]
        filtered_df = gas_df[gas_df["주소"].astype(str).str.contains(query, na=False)]
        
        if filtered_df.empty:
            filtered_df = gas_df[gas_df["행정구역"].astype(str).str.contains(query, na=False)]
        
        if filtered_df.empty:
            return {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "message": "검색 결과가 없습니다.",
                "results": {}
            }
        
        # 알고리즘 인스턴스
        available_algorithms = {
            "cosine_similarity": CosineSimilarityAlgorithm(
                service.centroids, service.norm_cols
            ),
            "euclidean_distance": EuclideanDistanceAlgorithm(
                service.centroids, service.norm_cols
            ),
            "pearson_correlation": PearsonCorrelationAlgorithm(
                service.centroids, service.norm_cols
            ),
            "popularity": PopularityAlgorithm(
                service.centroids, service.norm_cols, service.data["recommend_result"]
            ),
            "collaborative": CollaborativeAlgorithm(
                service.centroids, service.norm_cols, service.data["recommend_result"]
            ),
            "ahp_topsis": AHPTopsisAlgorithm(
                service.centroids, service.norm_cols, service.data["recommend_result"]
            ),
        }
        
        # 선택된 알고리즘만 테스트
        if algorithms:
            test_algorithms = {k: v for k, v in available_algorithms.items() if k in algorithms}
        else:
            test_algorithms = available_algorithms
        
        # 각 알고리즘 실행
        results = {}
        for algo_name, algo in test_algorithms.items():
            start_time = time.time()
            
            try:
                recommendations = algo.recommend(filtered_df, top_k)
                execution_time = time.time() - start_time
                
                results[algo_name] = {
                    "algorithm_name": algo.name,
                    "description": algo.description,
                    "execution_time_ms": round(execution_time * 1000, 2),
                    "count": len(recommendations),
                    "items": recommendations
                }
            except Exception as e:
                results[algo_name] = {
                    "algorithm_name": algo.name,
                    "error": str(e),
                    "execution_time_ms": round((time.time() - start_time) * 1000, 2)
                }
        
        return {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "total_algorithms_tested": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"알고리즘 비교 중 오류: {str(e)}")


@router.get("/algorithms")
async def list_algorithms():
    """사용 가능한 알고리즘 목록"""
    return [
        {
            "name": "cosine_similarity",
            "display_name": "코사인 유사도",
            "description": "특징 벡터 간 코사인 유사도를 계산합니다.",
            "status": "production"
        },
        {
            "name": "euclidean_distance",
            "display_name": "유클리드 거리",
            "description": "특징 벡터 간 유클리드 거리를 계산합니다.",
            "status": "testing"
        },
        {
            "name": "pearson_correlation",
            "display_name": "피어슨 상관계수",
            "description": "특징 벡터 간 피어슨 상관계수를 계산합니다.",
            "status": "testing"
        },
        {
            "name": "popularity",
            "display_name": "인기도 기반",
            "description": "용도 유형의 빈도수를 기반으로 추천합니다.",
            "status": "testing"
        },
        {
            "name": "collaborative",
            "display_name": "협업 필터링",
            "description": "유사한 주유소의 패턴을 기반으로 추천합니다.",
            "status": "testing"
        },
        {
            "name": "ahp_topsis",
            "display_name": "AHP-TOPSIS",
            "description": "다기준 의사결정 방법으로 추천합니다.",
            "status": "testing"
        }
    ]


@router.post("/benchmark")
async def benchmark_algorithms(
    test_queries: List[str] = Body(..., description="테스트할 주소 목록"),
    iterations: int = Body(default=1, ge=1, le=10),
    service: RecommendationService = Depends(get_recommendation_service),
):
    """
    알고리즘 성능 벤치마크
    
    **사용 예시**:
    ```json
    {
      "test_queries": ["서울 강남구", "부산 해운대구"],
      "iterations": 3
    }
    ```
    """
    try:
        available_algorithms = {
            "cosine_similarity": CosineSimilarityAlgorithm(
                service.centroids, service.norm_cols
            ),
            "euclidean_distance": EuclideanDistanceAlgorithm(
                service.centroids, service.norm_cols
            ),
            "pearson_correlation": PearsonCorrelationAlgorithm(
                service.centroids, service.norm_cols
            ),
            "popularity": PopularityAlgorithm(
                service.centroids, service.norm_cols, service.data["recommend_result"]
            ),
            "collaborative": CollaborativeAlgorithm(
                service.centroids, service.norm_cols, service.data["recommend_result"]
            ),
            "ahp_topsis": AHPTopsisAlgorithm(
                service.centroids, service.norm_cols, service.data["recommend_result"]
            ),
        }
        
        benchmark_results = {}
        
        for algo_name, algo in available_algorithms.items():
            total_time = 0
            success_count = 0
            error_count = 0
            
            for query in test_queries:
                for _ in range(iterations):
                    try:
                        gas_df = service.data["gas_station"]
                        filtered_df = gas_df[gas_df["주소"].astype(str).str.contains(query, na=False)]
                        
                        if filtered_df.empty:
                            filtered_df = gas_df[gas_df["행정구역"].astype(str).str.contains(query, na=False)]
                        
                        if not filtered_df.empty:
                            start_time = time.time()
                            algo.recommend(filtered_df, 10)
                            total_time += (time.time() - start_time)
                            success_count += 1
                        
                    except Exception:
                        error_count += 1
            
            avg_time = (total_time / success_count * 1000) if success_count > 0 else 0
            
            benchmark_results[algo_name] = {
                "algorithm_name": algo.name,
                "total_queries": len(test_queries) * iterations,
                "success_count": success_count,
                "error_count": error_count,
                "avg_execution_time_ms": round(avg_time, 2),
                "total_time_ms": round(total_time * 1000, 2)
            }
        
        # 정렬
        sorted_results = dict(sorted(
            benchmark_results.items(),
            key=lambda x: x[1]["avg_execution_time_ms"]
        ))
        
        return {
            "timestamp": datetime.now().isoformat(),
            "test_queries_count": len(test_queries),
            "iterations_per_query": iterations,
            "results": sorted_results,
            "fastest_algorithm": list(sorted_results.keys())[0] if sorted_results else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"벤치마크 중 오류: {str(e)}")