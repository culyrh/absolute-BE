"""
센트로이드 생성 스크립트
추천결과_행단위.csv에서 대분류×권역별 평균값을 계산하여 센트로이드 생성
"""
import pandas as pd
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def generate_centroids():
    print("🚀 센트로이드 생성 시작...")
    
    # 추천결과 CSV 로드
    recommend_result_path = project_root / "data" / "추천결과_행단위.csv"
    df = pd.read_csv(recommend_result_path)
    print(f"📊 추천결과 데이터 로드: {len(df)}개 행")
    
    # 필요한 컬럼
    norm_cols = [
        "인구[명]_norm",
        "교통량_norm", 
        "숙박업소(관광지수)_norm",
        "상권밀집도(비율)_norm",
        "공시지가(토지단가)_norm"
    ]
    
    # 대분류×권역별 그룹화하여 평균 계산
    centroids = []
    
    for usage_type in df["대분류"].dropna().unique():
        type_df = df[df["대분류"] == usage_type]
        
        for region in type_df["권역"].dropna().unique():
            region_df = type_df[type_df["권역"] == region]
            
            centroid = {
                "대분류": usage_type,
                "권역": region
            }
            
            # 각 특징별 평균값 계산
            for col in norm_cols:
                if col in region_df.columns:
                    mean_val = region_df[col].mean()
                    centroid[col] = mean_val if not pd.isna(mean_val) else 0.0
                else:
                    centroid[col] = 0.0
            
            centroids.append(centroid)
    
    # 데이터프레임 생성
    centroids_df = pd.DataFrame(centroids)
    
    print(f"📊 생성된 센트로이드: {len(centroids_df)}개")
    print(f"📊 대분류 종류: {centroids_df['대분류'].nunique()}개")
    print(f"📊 권역 종류: {centroids_df['권역'].nunique()}개")
    
    # CSV 저장
    output_path = project_root / "data" / "대분류_센터로이드.csv"
    centroids_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ 센트로이드 저장 완료: {output_path}")
    
    # 샘플 출력
    print("\n📋 샘플 데이터:")
    print(centroids_df.head(10))
    
    return centroids_df

if __name__ == "__main__":
    generate_centroids()