"""
알고리즘 성능 벤치마크 (속도 측정)

사용법:
    python -m app.comparison.benchmark
"""

import time
import pandas as pd
from typing import Dict, List

from app.utils.data_loader import load_all_data
from app.utils.preprocessing import preprocess_gas_station_data, merge_with_stats, normalize_features

from app.comparison.algorithms.cosine_similarity import CosineSimilarityAlgorithm
from app.comparison.algorithms.euclidean_distance import EuclideanDistanceAlgorithm
from app.comparison.algorithms.pearson_correlation import PearsonCorrelationAlgorithm
from app.comparison.algorithms.popularity import PopularityAlgorithm
from app.comparison.algorithms.collaborative import CollaborativeAlgorithm
from app.comparison.algorithms.ahp_topsis import AHPTopsisAlgorithm


def load_and_prepare_data():
    """데이터 로드 및 전처리"""
    print("📊 데이터 로딩 중...")
    data = load_all_data()
    
    data["gas_station"] = preprocess_gas_station_data(data["gas_station"])
    data["gas_station"] = merge_with_stats(
        data["gas_station"],
        data["population"],
        data["business"]
    )
    
    feature_cols = ["인구[명]", "교통량", "숙박업소(관광지수)", "상권밀집도(비율)", "공시지가(토지단가)"]
    available_cols = [col for col in feature_cols if col in data["gas_station"].columns]
    data["gas_station"] = normalize_features(data["gas_station"], available_cols)
    
    norm_cols = [f"{col}_norm" for col in feature_cols]
    
    return data, norm_cols


def run_benchmark(queries: List[str] = None, iterations: int = 5):
    """벤치마크 실행"""
    
    if queries is None:
        queries = ["서울 강남구", "부산 해운대구", "전주시"]
    
    # 데이터 준비
    data, norm_cols = load_and_prepare_data()
    centroids = data["centroid"]
    gas_df = data["gas_station"]
    recommend_result = data["recommend_result"]
    
    # 알고리즘 인스턴스 생성
    algorithms = {
        "코사인 유사도": CosineSimilarityAlgorithm(centroids, norm_cols),
        "유클리드 거리": EuclideanDistanceAlgorithm(centroids, norm_cols),
        "피어슨 상관계수": PearsonCorrelationAlgorithm(centroids, norm_cols),
        "인기도 기반": PopularityAlgorithm(centroids, norm_cols, recommend_result),
        "협업 필터링": CollaborativeAlgorithm(centroids, norm_cols, recommend_result),
        "AHP-TOPSIS": AHPTopsisAlgorithm(centroids, norm_cols, recommend_result),
    }
    
    print(f"\n🚀 벤치마크 시작: {len(queries)}개 쿼리 × {iterations}회 반복\n")
    
    results = []
    
    for algo_name, algo in algorithms.items():
        print(f"⏱️  {algo_name} 테스트 중...")
        
        total_time = 0
        success_count = 0
        error_count = 0
        
        for query in queries:
            for i in range(iterations):
                try:
                    # 주소 검색
                    filtered_df = gas_df[gas_df["주소"].astype(str).str.contains(query, na=False)]
                    
                    if filtered_df.empty:
                        filtered_df = gas_df[gas_df["행정구역"].astype(str).str.contains(query, na=False)]
                    
                    if not filtered_df.empty:
                        start_time = time.time()
                        recommendations = algo.recommend(filtered_df, 10)
                        execution_time = time.time() - start_time
                        
                        total_time += execution_time
                        success_count += 1
                    
                except Exception as e:
                    error_count += 1
                    print(f"  ❌ 오류: {query} - {str(e)}")
        
        avg_time = (total_time / success_count) if success_count > 0 else 0
        
        results.append({
            "알고리즘": algo_name,
            "성공": success_count,
            "실패": error_count,
            "평균 시간(ms)": round(avg_time * 1000, 2),
            "총 시간(s)": round(total_time, 2)
        })
    
    # 결과 출력
    print("\n" + "="*70)
    print("📊 벤치마크 결과")
    print("="*70 + "\n")
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("평균 시간(ms)")
    
    print(df_results.to_string(index=False))
    
    print("\n" + "="*70)
    print(f"🏆 가장 빠른 알고리즘: {df_results.iloc[0]['알고리즘']}")
    print(f"⚡ 평균 실행 시간: {df_results.iloc[0]['평균 시간(ms)']} ms")
    print("="*70 + "\n")
    
    return df_results


def main():
    """메인 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description="추천 알고리즘 성능 벤치마크")
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["서울 강남구", "부산 해운대구", "전주시"],
        help="테스트할 주소 목록"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="각 쿼리당 반복 횟수"
    )
    
    args = parser.parse_args()
    run_benchmark(args.queries, args.iterations)


if __name__ == "__main__":
    main()