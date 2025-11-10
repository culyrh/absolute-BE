"""
통합 벤치마크 테스트
모든 추천 알고리즘(기본 + ML)을 한번에 비교 테스트

사용법:
    python -m app.comparison.benchmark
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime
import time
from pathlib import Path

from app.utils.data_loader import load_all_data
from app.comparison.algorithms.cosine_similarity import CosineSimilarityAlgorithm
from app.comparison.algorithms.euclidean_distance import EuclideanDistanceAlgorithm
from app.comparison.algorithms.pearson_correlation import PearsonCorrelationAlgorithm
from app.comparison.algorithms.popularity import PopularityAlgorithm
from app.comparison.algorithms.collaborative import CollaborativeAlgorithm
from app.comparison.algorithms.ahp_topsis import AHPTopsisAlgorithm
from app.services.ml_location_recommender import MLLocationRecommender


class BenchmarkTest:
    """통합 벤치마크 테스트 클래스"""
    
    def __init__(self):
        self.data = None
        self.train_data = None
        self.test_data = None
        self.centroids = None
        self.norm_cols = None
        self.ml_recommender = None
        self.results = {}
        
    def load_data(self):
        """데이터 로드"""
        print("="*80)
        print("📊 벤치마크 테스트 - 데이터 로딩")
        print("="*80 + "\n")
        
        # 모든 데이터 로드
        self.data = load_all_data()
        
        # Train 데이터
        self.train_data = self.data["recommend_result"]
        print(f"✅ Train 데이터 로드: {len(self.train_data)}개")
        
        # 센트로이드 데이터
        self.centroids = self.data["centroid"]
        
        # 정규화 컬럼
        feature_cols = ["인구[명]", "교통량", "숙박업소(관광지수)", "상권밀집도(비율)", "공시지가(토지단가)"]
        self.norm_cols = [f"{col}_norm" for col in feature_cols]
        
    def load_test_data(self, test_file_path: str = "data/test_data.csv"):
        """테스트 데이터 로드"""
        print(f"\n📂 테스트 데이터 로드 중: {test_file_path}")
        
        try:
            self.test_data = pd.read_csv(test_file_path, encoding="utf-8-sig")
            print(f"✅ 테스트 데이터 로드 완료: {len(self.test_data)}개")
            
            if "대분류" in self.test_data.columns:
                print(f"   - 대분류 종류: {self.test_data['대분류'].nunique()}개")
                print(f"   - 대분류 분포:\n{self.test_data['대분류'].value_counts()}")
            
            return self.test_data
            
        except FileNotFoundError:
            print(f"❌ 오류: 테스트 데이터 파일을 찾을 수 없습니다: {test_file_path}")
            raise
        except Exception as e:
            print(f"❌ 테스트 데이터 로드 실패: {str(e)}")
            raise
    
    def initialize_ml(self):
        """ML 모델 초기화 및 학습"""
        print("\n" + "="*80)
        print("🤖 ML 모델 학습")
        print("="*80 + "\n")
        
        self.ml_recommender = MLLocationRecommender()
        
        start_time = time.time()
        accuracy = self.ml_recommender.train()
        train_time = time.time() - start_time
        
        print(f"✅ ML 학습 완료: 정확도 {accuracy:.3f}, 소요 시간 {train_time:.2f}초")
        
        return accuracy
    
    def run_traditional_algorithm_test(self, algorithm, algorithm_name: str) -> Dict:
        """전통적인 알고리즘 테스트"""
        print(f"\n⏱️  {algorithm_name} 테스트 중...")
        
        results = {
            "algorithm": algorithm_name,
            "type": "traditional",
            "top1_correct": 0,
            "top3_correct": 0,
            "top5_correct": 0,
            "total": len(self.test_data),
            "execution_times": [],
            "usage_type_accuracy": {}
        }
        
        for idx, row in self.test_data.iterrows():
            true_label = row["대분류"]
            test_df = pd.DataFrame([row])
            
            start_time = time.time()
            try:
                recommendations = algorithm.recommend(test_df, top_k=5)
                execution_time = time.time() - start_time
                results["execution_times"].append(execution_time)
                
                if len(recommendations) > 0:
                    # Top-1
                    if recommendations[0]["usage_type"] == true_label:
                        results["top1_correct"] += 1
                    
                    # Top-3
                    top3_types = [r["usage_type"] for r in recommendations[:3]]
                    if true_label in top3_types:
                        results["top3_correct"] += 1
                    
                    # Top-5
                    top5_types = [r["usage_type"] for r in recommendations[:5]]
                    if true_label in top5_types:
                        results["top5_correct"] += 1
                    
                    # 대분류별
                    if true_label not in results["usage_type_accuracy"]:
                        results["usage_type_accuracy"][true_label] = {"correct": 0, "total": 0}
                    
                    results["usage_type_accuracy"][true_label]["total"] += 1
                    if recommendations[0]["usage_type"] == true_label:
                        results["usage_type_accuracy"][true_label]["correct"] += 1
                        
            except Exception as e:
                print(f"  ❌ 오류 (인덱스 {idx}): {str(e)}")
                results["execution_times"].append(0)
        
        # 정확도 계산
        results["top1_accuracy"] = (results["top1_correct"] / results["total"]) * 100 if results["total"] > 0 else 0
        results["top3_accuracy"] = (results["top3_correct"] / results["total"]) * 100 if results["total"] > 0 else 0
        results["top5_accuracy"] = (results["top5_correct"] / results["total"]) * 100 if results["total"] > 0 else 0
        results["avg_execution_time"] = np.mean(results["execution_times"]) * 1000 if results["execution_times"] else 0
        
        print(f"   ✅ Top-1: {results['top1_accuracy']:.2f}% | Top-3: {results['top3_accuracy']:.2f}% | Top-5: {results['top5_accuracy']:.2f}%")
        print(f"   ⚡ 평균 실행시간: {results['avg_execution_time']:.2f} ms")
        
        return results
    
    def run_ml_test(self) -> Dict:
        """ML 알고리즘 테스트"""
        print(f"\n⏱️  ML (Random Forest) 테스트 중...")
        
        results = {
            "algorithm": "ML (Random Forest)",
            "type": "ml",
            "top1_correct": 0,
            "top3_correct": 0,
            "top5_correct": 0,
            "total": len(self.test_data),
            "execution_times": [],
            "usage_type_accuracy": {}
        }
        
        for idx, row in self.test_data.iterrows():
            true_label = row["대분류"]
            
            start_time = time.time()
            try:
                predictions = self.ml_recommender._predict_from_row(row, top_n=5)
                execution_time = time.time() - start_time
                results["execution_times"].append(execution_time)
                
                if len(predictions) > 0:
                    # Top-1
                    if predictions[0]["category"] == true_label:
                        results["top1_correct"] += 1
                    
                    # Top-3
                    top3_types = [p["category"] for p in predictions[:3]]
                    if true_label in top3_types:
                        results["top3_correct"] += 1
                    
                    # Top-5
                    top5_types = [p["category"] for p in predictions[:5]]
                    if true_label in top5_types:
                        results["top5_correct"] += 1
                    
                    # 대분류별
                    if true_label not in results["usage_type_accuracy"]:
                        results["usage_type_accuracy"][true_label] = {"correct": 0, "total": 0}
                    
                    results["usage_type_accuracy"][true_label]["total"] += 1
                    if predictions[0]["category"] == true_label:
                        results["usage_type_accuracy"][true_label]["correct"] += 1
                        
            except Exception as e:
                print(f"  ❌ 오류 (인덱스 {idx}): {str(e)}")
                results["execution_times"].append(0)
        
        # 정확도 계산
        results["top1_accuracy"] = (results["top1_correct"] / results["total"]) * 100 if results["total"] > 0 else 0
        results["top3_accuracy"] = (results["top3_correct"] / results["total"]) * 100 if results["total"] > 0 else 0
        results["top5_accuracy"] = (results["top5_correct"] / results["total"]) * 100 if results["total"] > 0 else 0
        results["avg_execution_time"] = np.mean(results["execution_times"]) * 1000 if results["execution_times"] else 0
        
        print(f"   ✅ Top-1: {results['top1_accuracy']:.2f}% | Top-3: {results['top3_accuracy']:.2f}% | Top-5: {results['top5_accuracy']:.2f}%")
        print(f"   ⚡ 평균 실행시간: {results['avg_execution_time']:.2f} ms")
        
        return results
    
    def run_all_tests(self):
        """모든 알고리즘 벤치마크 테스트"""
        print("\n" + "="*80)
        print("🚀 벤치마크 테스트 시작")
        print("="*80)
        
        # 전통적인 알고리즘들
        traditional_algorithms = {
            "코사인 유사도": CosineSimilarityAlgorithm(self.centroids, self.norm_cols),
            "유클리드 거리": EuclideanDistanceAlgorithm(self.centroids, self.norm_cols),
            "피어슨 상관계수": PearsonCorrelationAlgorithm(self.centroids, self.norm_cols),
            "인기도 기반": PopularityAlgorithm(self.centroids, self.norm_cols, self.train_data),
            "협업 필터링": CollaborativeAlgorithm(self.centroids, self.norm_cols, self.train_data),
            "AHP-TOPSIS": AHPTopsisAlgorithm(self.centroids, self.norm_cols, self.train_data),
        }
        
        # 전통적인 알고리즘 테스트
        for name, algorithm in traditional_algorithms.items():
            result = self.run_traditional_algorithm_test(algorithm, name)
            self.results[name] = result
        
        # ML 알고리즘 테스트
        if self.ml_recommender:
            result = self.run_ml_test()
            self.results["ML (Random Forest)"] = result
        
        # 결과 출력
        self.print_results()
        
        # 결과 저장
        self.save_results()
    
    def print_results(self):
        """결과 출력"""
        print("\n" + "="*80)
        print("📊 벤치마크 결과 - 전체 알고리즘 비교")
        print("="*80 + "\n")
        
        # 테이블 헤더
        print(f"{'알고리즘':<25} {'유형':<12} {'Top-1':<10} {'Top-3':<10} {'Top-5':<10} {'실행시간(ms)':<15}")
        print("-" * 80)
        
        # 결과 출력
        for name, result in self.results.items():
            algo_type = "ML" if result.get("type") == "ml" else "Traditional"
            print(f"{name:<25} {algo_type:<12} {result['top1_accuracy']:>6.2f}%  {result['top3_accuracy']:>6.2f}%  "
                  f"{result['top5_accuracy']:>6.2f}%  {result['avg_execution_time']:>10.2f}")
        
        print("\n" + "="*80)
        
        # 최고 성능 분석
        if self.results:
            # Top-1 기준 최고
            best_top1 = max(self.results.items(), key=lambda x: x[1]["top1_accuracy"])
            print(f"🥇 Top-1 최고: {best_top1[0]} ({best_top1[1]['top1_accuracy']:.2f}%)")
            
            # Top-3 기준 최고
            best_top3 = max(self.results.items(), key=lambda x: x[1]["top3_accuracy"])
            print(f"🥈 Top-3 최고: {best_top3[0]} ({best_top3[1]['top3_accuracy']:.2f}%)")
            
            # 가장 빠른 알고리즘
            fastest = min(self.results.items(), key=lambda x: x[1]["avg_execution_time"])
            print(f"⚡ 가장 빠름: {fastest[0]} ({fastest[1]['avg_execution_time']:.2f} ms)")
        
        print("="*80 + "\n")
    
    def save_results(self):
        """결과 저장"""
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 전체 결과
        summary_data = []
        for name, result in self.results.items():
            summary_data.append({
                "알고리즘": name,
                "유형": "ML" if result.get("type") == "ml" else "Traditional",
                "Top-1 정확도(%)": result["top1_accuracy"],
                "Top-3 정확도(%)": result["top3_accuracy"],
                "Top-5 정확도(%)": result["top5_accuracy"],
                "평균 실행시간(ms)": result["avg_execution_time"],
                "테스트 샘플 수": result["total"]
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / f"benchmark_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
        print(f"💾 벤치마크 결과 저장: {summary_file}\n")


def main():
    """메인 실행 함수"""
    print("\n" + "="*80)
    print("🎯 통합 벤치마크 테스트")
    print("   모든 추천 알고리즘 성능 비교")
    print("="*80 + "\n")
    
    benchmark = BenchmarkTest()
    
    # 1. 데이터 로드
    benchmark.load_data()
    
    # 2. 테스트 데이터 로드
    try:
        benchmark.load_test_data("data/test_data.csv")
    except Exception as e:
        print("\n❌ 테스트 종료: 테스트 데이터를 준비해주세요.")
        return
    
    # 3. ML 모델 학습
    try:
        benchmark.initialize_ml()
    except Exception as e:
        print(f"\n⚠️ ML 모델 초기화 실패: {str(e)}")
        print("ML 알고리즘을 제외하고 테스트를 진행합니다.\n")
    
    # 4. 모든 알고리즘 테스트
    benchmark.run_all_tests()
    
    print("\n✅ 벤치마크 테스트 완료!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()