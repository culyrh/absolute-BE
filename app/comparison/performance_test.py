"""
추천 알고리즘 성능 테스트 스크립트

사용법:
    python -m app.comparison.performance_test
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import time
from pathlib import Path

from app.utils.data_loader import load_all_data
from app.utils.preprocessing import (
    preprocess_gas_station_data, merge_with_stats, normalize_features
)
from app.comparison.algorithms.cosine_similarity import CosineSimilarityAlgorithm
from app.comparison.algorithms.euclidean_distance import EuclideanDistanceAlgorithm
from app.comparison.algorithms.pearson_correlation import PearsonCorrelationAlgorithm
from app.comparison.algorithms.popularity import PopularityAlgorithm
from app.comparison.algorithms.collaborative import CollaborativeAlgorithm
from app.comparison.algorithms.ahp_topsis import AHPTopsisAlgorithm


class PerformanceTest:
    """성능 테스트 클래스"""
    
    def __init__(self):
        self.data = None
        self.train_data = None
        self.test_data = None
        self.centroids = None
        self.norm_cols = None
        self.results = {}
        
    def load_data(self):
        """데이터 로드 및 전처리"""
        print("📊 데이터 로딩 중...")
        
        # 모든 데이터 로드
        self.data = load_all_data()
        
        # Train 데이터: 추천결과_행단위.csv (1,440개)
        self.train_data = self.data["recommend_result"]
        print(f"✅ Train 데이터 로드: {len(self.train_data)}개")
        
        # 센트로이드 데이터
        self.centroids = self.data["centroid"]
        
        # 정규화 컬럼
        feature_cols = ["인구[명]", "교통량", "숙박업소(관광지수)", "상권밀집도(비율)", "공시지가(토지단가)"]
        self.norm_cols = [f"{col}_norm" for col in feature_cols]
        
    def generate_test_data(self) -> pd.DataFrame:
        """
        증강 테스트 데이터 생성
        
        방법: 대분류별 권역당 3개씩 생성
        - 각 대분류의 권역별 중위값 기준
        - 약간의 노이즈 추가하여 3개 샘플 생성
        """
        print("\n🔬 테스트 데이터 생성 중...")
        
        # 대분류 목록
        usage_types = self.train_data["대분류"].unique()
        
        # 권역 목록 (17개)
        regions = self.train_data["권역"].unique() if "권역" in self.train_data.columns else []
        
        if len(regions) == 0:
            regions = [
                "서울특별시", "부산광역시", "대구광역시", "인천광역시",
                "광주광역시", "대전광역시", "울산광역시", "세종특별자치시",
                "경기도", "강원특별자치도", "충청북도", "충청남도",
                "전북특별자치도", "전라남도", "경상북도", "경상남도", "제주특별자치도"
            ]
        
        test_samples = []
        
        for usage_type in usage_types:
            for region in regions:
                # 해당 대분류 + 권역의 데이터 필터링
                subset = self.train_data[
                    (self.train_data["대분류"] == usage_type) &
                    (self.train_data["권역"] == region)
                ]
                
                if len(subset) > 0:
                    # 중위값 계산
                    available_norm_cols = [col for col in self.norm_cols if col in subset.columns]
                    
                    medians = {}
                    for col in available_norm_cols:
                        medians[col] = subset[col].median()
                    
                    # 3개 샘플 생성 (약간의 노이즈 추가)
                    for i in range(3):
                        sample = {
                            "대분류": usage_type,
                            "권역": region,
                            "test_id": f"{usage_type}_{region}_{i+1}"
                        }
                        
                        # 각 특징에 노이즈 추가 (±10% 범위)
                        for col in available_norm_cols:
                            noise = np.random.uniform(-0.1, 0.1)
                            sample[col] = max(0, min(1, medians[col] + noise))
                        
                        # 원본 정보 (주소 등)
                        if len(subset) > 0:
                            sample_row = subset.iloc[0]
                            sample["주소"] = f"{region} (테스트 샘플 {i+1})"
                            sample["행정구역"] = sample_row.get("행정구역", region)
                        
                        test_samples.append(sample)
        
        self.test_data = pd.DataFrame(test_samples)
        print(f"✅ 테스트 데이터 생성 완료: {len(self.test_data)}개")
        print(f"   - 대분류 수: {len(usage_types)}")
        print(f"   - 권역 수: {len(regions)}")
        print(f"   - 샘플당 개수: 3개")
        
        return self.test_data
    
    def run_algorithm_test(self, algorithm, algorithm_name: str) -> Dict:
        """단일 알고리즘 성능 테스트"""
        print(f"\n⏱️  {algorithm_name} 테스트 중...")
        
        results = {
            "algorithm": algorithm_name,
            "top1_correct": 0,
            "top3_correct": 0,
            "top5_correct": 0,
            "total": len(self.test_data),
            "execution_times": [],
            "region_accuracy": {},
            "usage_type_accuracy": {}
        }
        
        for idx, row in self.test_data.iterrows():
            # 정답
            true_label = row["대분류"]
            region = row["권역"]
            
            # 테스트 데이터를 DataFrame으로 변환
            test_df = pd.DataFrame([row])
            
            # 추천 실행
            start_time = time.time()
            try:
                recommendations = algorithm.recommend(test_df, top_k=5)
                execution_time = time.time() - start_time
                results["execution_times"].append(execution_time)
                
                if len(recommendations) > 0:
                    # Top-1 정확도
                    if recommendations[0]["usage_type"] == true_label:
                        results["top1_correct"] += 1
                    
                    # Top-3 정확도
                    top3_types = [r["usage_type"] for r in recommendations[:3]]
                    if true_label in top3_types:
                        results["top3_correct"] += 1
                    
                    # Top-5 정확도
                    top5_types = [r["usage_type"] for r in recommendations[:5]]
                    if true_label in top5_types:
                        results["top5_correct"] += 1
                    
                    # 권역별 정확도
                    if region not in results["region_accuracy"]:
                        results["region_accuracy"][region] = {"correct": 0, "total": 0}
                    
                    results["region_accuracy"][region]["total"] += 1
                    if recommendations[0]["usage_type"] == true_label:
                        results["region_accuracy"][region]["correct"] += 1
                    
                    # 대분류별 정확도
                    if true_label not in results["usage_type_accuracy"]:
                        results["usage_type_accuracy"][true_label] = {"correct": 0, "total": 0}
                    
                    results["usage_type_accuracy"][true_label]["total"] += 1
                    if recommendations[0]["usage_type"] == true_label:
                        results["usage_type_accuracy"][true_label]["correct"] += 1
                        
            except Exception as e:
                print(f"  ❌ 오류: {str(e)}")
                results["execution_times"].append(0)
        
        # 정확도 계산
        results["top1_accuracy"] = (results["top1_correct"] / results["total"]) * 100
        results["top3_accuracy"] = (results["top3_correct"] / results["total"]) * 100
        results["top5_accuracy"] = (results["top5_correct"] / results["total"]) * 100
        results["avg_execution_time"] = np.mean(results["execution_times"]) * 1000  # ms
        
        print(f"   ✅ Top-1 정확도: {results['top1_accuracy']:.2f}%")
        print(f"   ✅ Top-3 정확도: {results['top3_accuracy']:.2f}%")
        print(f"   ✅ Top-5 정확도: {results['top5_accuracy']:.2f}%")
        print(f"   ⚡ 평균 실행시간: {results['avg_execution_time']:.2f} ms")
        
        return results
    
    def run_all_tests(self):
        """모든 알고리즘 테스트"""
        print("\n" + "="*70)
        print("🚀 성능 테스트 시작")
        print("="*70)
        
        # 알고리즘 인스턴스 생성
        algorithms = {
            "AHP-TOPSIS": AHPTopsisAlgorithm(
                self.centroids, self.norm_cols, self.train_data
            ),
            "기본 CF": CollaborativeAlgorithm(
                self.centroids, self.norm_cols, self.train_data
            ),
            "코사인 유사도 CF": CosineSimilarityAlgorithm(
                self.centroids, self.norm_cols
            ),
            "피어슨 상관계수 CF": PearsonCorrelationAlgorithm(
                self.centroids, self.norm_cols
            ),
        }
        
        # 각 알고리즘 테스트
        for name, algorithm in algorithms.items():
            result = self.run_algorithm_test(algorithm, name)
            self.results[name] = result
        
        # 결과 출력
        self.print_results()
        
        # 결과 저장
        self.save_results()
    
    def print_results(self):
        """결과 출력"""
        print("\n" + "="*70)
        print("📊 전체 정확도 비교")
        print("="*70 + "\n")
        
        # 테이블 형식으로 출력
        print(f"{'알고리즘':<20} {'Top-1':<10} {'Top-3':<10} {'Top-5':<10} {'실행시간(ms)':<15}")
        print("-" * 70)
        
        for name, result in self.results.items():
            print(f"{name:<20} {result['top1_accuracy']:>6.2f}%  {result['top3_accuracy']:>6.2f}%  "
                  f"{result['top5_accuracy']:>6.2f}%  {result['avg_execution_time']:>10.2f}")
        
        print("\n" + "="*70)
        
        # 최고 성능 알고리즘
        best_algo = max(self.results.items(), key=lambda x: x[1]["top1_accuracy"])
        print(f"🏆 최고 성능 알고리즘: {best_algo[0]}")
        print(f"   Top-1 정확도: {best_algo[1]['top1_accuracy']:.2f}%")
        print("="*70 + "\n")
    
    def save_results(self):
        """결과를 CSV 파일로 저장"""
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 전체 결과
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                "알고리즘": name,
                "Top-1 정확도(%)": result["top1_accuracy"],
                "Top-3 정확도(%)": result["top3_accuracy"],
                "Top-5 정확도(%)": result["top5_accuracy"],
                "평균 실행시간(ms)": result["avg_execution_time"],
                "테스트 샘플 수": result["total"]
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / f"summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
        print(f"💾 전체 결과 저장: {summary_file}")
        
        # 권역별 결과
        for algo_name, result in self.results.items():
            region_data = []
            for region, acc in result["region_accuracy"].items():
                region_data.append({
                    "권역": region,
                    "정확도(%)": (acc["correct"] / acc["total"]) * 100 if acc["total"] > 0 else 0,
                    "정확 수": acc["correct"],
                    "전체 수": acc["total"]
                })
            
            region_df = pd.DataFrame(region_data)
            region_file = output_dir / f"region_{algo_name.replace(' ', '_')}_{timestamp}.csv"
            region_df.to_csv(region_file, index=False, encoding="utf-8-sig")
        
        print(f"💾 권역별 결과 저장 완료")
        
        # 대분류별 결과
        for algo_name, result in self.results.items():
            usage_data = []
            for usage_type, acc in result["usage_type_accuracy"].items():
                usage_data.append({
                    "대분류": usage_type,
                    "정확도(%)": (acc["correct"] / acc["total"]) * 100 if acc["total"] > 0 else 0,
                    "정확 수": acc["correct"],
                    "전체 수": acc["total"]
                })
            
            usage_df = pd.DataFrame(usage_data)
            usage_file = output_dir / f"usage_type_{algo_name.replace(' ', '_')}_{timestamp}.csv"
            usage_df.to_csv(usage_file, index=False, encoding="utf-8-sig")
        
        print(f"💾 대분류별 결과 저장 완료\n")


def main():
    """메인 실행 함수"""
    test = PerformanceTest()
    
    # 1. 데이터 로드
    test.load_data()
    
    # 2. 테스트 데이터 생성
    test.generate_test_data()
    
    # 3. 모든 알고리즘 테스트
    test.run_all_tests()
    
    print("\n✅ 성능 테스트 완료!")


if __name__ == "__main__":
    main()