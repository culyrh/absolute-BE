"""
ML 알고리즘 성능 테스트 스크립트
app/comparison/ml_performance_test.py

사용법:
    python -m app.comparison.ml_performance_test
    
테스트 데이터는 data/test_data.csv에 수동으로 준비되어 있어야 합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict
from datetime import datetime
import time
from pathlib import Path

from app.services.ml_location_recommender import MLLocationRecommender


class MLPerformanceTest:
    """ML 알고리즘 성능 테스트 클래스"""
    
    def __init__(self):
        self.recommender = MLLocationRecommender()
        self.train_data = None
        self.test_data = None
        self.results = {}
        
    def load_and_train(self):
        """ML 모델 학습"""
        print("🚀 ML 모델 학습 중...")
        
        start_time = time.time()
        accuracy = self.recommender.train()
        train_time = time.time() - start_time
        
        print(f"✅ 학습 완료: 정확도 {accuracy:.3f}, 소요 시간 {train_time:.2f}초")
        
        return accuracy, train_time
    
    def load_test_data(self, test_file_path: str = "data/test_data.csv"):
        """
        수동으로 준비된 ML 테스트 데이터 로드
        
        Args:
            test_file_path: 테스트 데이터 파일 경로
            
        테스트 데이터 형식:
        - 필수 컬럼: 대분류, 인구[명], 교통량, 숙박업소(관광지수), 상권밀집도(비율)
        - 선택 컬럼: 권역, 주소, 행정구역
        """
        print(f"\n📂 ML 테스트 데이터 로드 중: {test_file_path}")
        
        try:
            self.test_data = pd.read_csv(test_file_path, encoding="utf-8-sig")
            print(f"✅ 테스트 데이터 로드 완료: {len(self.test_data)}개")
            
            # 필수 컬럼 확인
            required_cols = ["대분류"] + self.recommender.FEATURE_COLS
            missing_cols = [col for col in required_cols if col not in self.test_data.columns]
            
            if missing_cols:
                print(f"⚠️ 경고: 다음 컬럼이 테스트 데이터에 없습니다: {missing_cols}")
            
            # 대분류 분포 출력
            if "대분류" in self.test_data.columns:
                print(f"   - 대분류 종류: {self.test_data['대분류'].nunique()}개")
                print(f"   - 대분류 분포:\n{self.test_data['대분류'].value_counts()}")
            
            return self.test_data
            
        except FileNotFoundError:
            print(f"❌ 오류: 테스트 데이터 파일을 찾을 수 없습니다: {test_file_path}")
            print("   테스트 데이터를 data/test_data.csv에 수동으로 준비해주세요.")
            raise
        except Exception as e:
            print(f"❌ 테스트 데이터 로드 실패: {str(e)}")
            raise
    
    def run_test(self) -> Dict:
        """ML 알고리즘 성능 테스트"""
        print(f"\n⏱️  ML (Random Forest) 테스트 중...")
        
        results = {
            "algorithm": "ML (Random Forest)",
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
                predictions = self.recommender._predict_from_row(row, top_n=5)
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
                    if len(predictions) >= 5:
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
        
        results["top1_accuracy"] = (results["top1_correct"] / results["total"]) * 100 if results["total"] > 0 else 0
        results["top3_accuracy"] = (results["top3_correct"] / results["total"]) * 100 if results["total"] > 0 else 0
        results["top5_accuracy"] = (results["top5_correct"] / results["total"]) * 100 if results["total"] > 0 else 0
        results["avg_execution_time"] = np.mean(results["execution_times"]) * 1000 if results["execution_times"] else 0
        
        print(f"   ✅ Top-1 정확도: {results['top1_accuracy']:.2f}%")
        print(f"   ✅ Top-3 정확도: {results['top3_accuracy']:.2f}%")
        print(f"   ✅ Top-5 정확도: {results['top5_accuracy']:.2f}%")
        print(f"   ⚡ 평균 실행시간: {results['avg_execution_time']:.2f} ms")
        
        self.results = results
        return results
    
    def save_results(self):
        """결과 저장"""
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 전체 결과
        summary_data = {
            "알고리즘": self.results["algorithm"],
            "Top-1 정확도(%)": self.results["top1_accuracy"],
            "Top-3 정확도(%)": self.results["top3_accuracy"],
            "Top-5 정확도(%)": self.results["top5_accuracy"],
            "평균 실행시간(ms)": self.results["avg_execution_time"],
            "테스트 샘플 수": self.results["total"]
        }
        
        summary_df = pd.DataFrame([summary_data])
        summary_file = output_dir / f"ml_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
        print(f"\n💾 ML 결과 저장: {summary_file}")
        
        # 대분류별 결과
        usage_data = []
        for usage_type, acc in self.results["usage_type_accuracy"].items():
            usage_data.append({
                "대분류": usage_type,
                "정확도(%)": (acc["correct"] / acc["total"]) * 100 if acc["total"] > 0 else 0,
                "정확 수": acc["correct"],
                "전체 수": acc["total"]
            })
        
        if usage_data:
            usage_df = pd.DataFrame(usage_data)
            usage_file = output_dir / f"ml_usage_type_{timestamp}.csv"
            usage_df.to_csv(usage_file, index=False, encoding="utf-8-sig")
            print(f"💾 대분류별 결과 저장 완료\n")


def main():
    """메인 실행 함수"""
    print("="*70)
    print("🚀 ML 알고리즘 성능 테스트")
    print("="*70 + "\n")
    
    test = MLPerformanceTest()
    
    # 1. ML 모델 학습
    test.load_and_train()
    
    # 2. 테스트 데이터 로드 (수동으로 준비된 데이터)
    try:
        test.load_test_data("data/test_data.csv")
    except Exception as e:
        print("\n❌ 테스트 데이터 로드 실패")
        print("테스트 데이터를 data/test_data.csv에 준비해주세요.")
        print("\n필요한 컬럼:")
        print("  - 대분류")
        print("  - 인구[명]")
        print("  - 교통량")
        print("  - 숙박업소(관광지수)")
        print("  - 상권밀집도(비율)")
        #print("  - 공시지가(토지단가)")
        return
    
    # 3. ML 알고리즘 테스트
    test.run_test()
    
    # 4. 결과 저장
    test.save_results()
    
    print("\n✅ ML 성능 테스트 완료!")


if __name__ == "__main__":
    main()