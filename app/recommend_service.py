import pandas as pd
import numpy as np
import re
from typing import List, Dict

# =========================================================
# 데이터 로드 및 전처리
# =========================================================
def load_data() -> pd.DataFrame:
    print("📂 데이터 로드 중...")

    gas_path = "data/jeonju_gas_station.csv"
    pop_path = "data/전국인구수.xlsx"
    biz_path = "data/전국1000명당사업체수.xlsx"

    # === 파일 로드 ===
    gas_df = pd.read_csv(gas_path)
    pop_df = pd.read_excel(pop_path)
    biz_df = pd.read_excel(biz_path)

    # === 주소 컬럼 자동 탐색 ===
    address_col = None
    for c in gas_df.columns:
        if any(k in c for k in ["주소", "소재지", "지번", "address"]):
            address_col = c
            break

    if not address_col:
        raise ValueError(f"주소 컬럼을 찾을 수 없습니다. 현재 컬럼: {list(gas_df.columns)}")

    # === 행정구역 추출 (시/군/구 단위) ===
    gas_df["행정구역"] = gas_df[address_col].apply(
        lambda x: re.findall(r"[가-힣]+시|[가-힣]+군|[가-힣]+구", str(x))[0]
        if isinstance(x, str) and re.findall(r"[가-힣]+시|[가-힣]+군|[가-힣]+구", str(x))
        else None
    )

    # === 인구, 상권 컬럼 자동 인식 ===
    pop_col = next((c for c in pop_df.columns if "인구" in c), pop_df.columns[-1])
    biz_col = next((c for c in biz_df.columns if "사업체" in c or "천명" in c), biz_df.columns[-1])

    # === 병합 ===
    df = gas_df.merge(pop_df[["행정구역", pop_col]], on="행정구역", how="left")
    df = df.merge(biz_df[["행정구역", biz_col]], on="행정구역", how="left")

    # === 컬럼 통일 ===
    df.rename(
        columns={
            address_col: "주소",
            pop_col: "인구[명]",
            biz_col: "인구천명당사업체수",
        },
        inplace=True,
    )

    # === 결측치 처리 ===
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # === 표준화 ===
    for col in ["인구[명]", "인구천명당사업체수"]:
        mean, std = df[col].mean(), df[col].std()
        df[f"{col}_norm"] = (df[col] - mean) / (std + 1e-9)

    print(f"✅ 데이터 통합 및 표준화 완료: {len(df)}개 행")
    return df


# =========================================================
# 추천 로직
# =========================================================
def recommend_by_query(df: pd.DataFrame, query: str, topk: int = 10) -> List[Dict]:
    if not query:
        return []

    # === 주소 컬럼 탐색 ===
    address_col = None
    for c in df.columns:
        if any(k in c for k in ["주소", "소재지", "지번", "address"]):
            address_col = c
            break

    if not address_col:
        raise ValueError(f"추천 단계에서 주소 컬럼을 찾을 수 없습니다. 현재 컬럼: {list(df.columns)}")

    # === 검색 ===
    sub = df[df[address_col].astype(str).str.contains(query, na=False)]

    # === 유사도(점수) 계산 ===
    score_cols = [c for c in df.columns if c.endswith("_norm")]
    if not score_cols:
        raise ValueError("표준화된 지표 컬럼(_norm)이 없습니다.")

    # 원본 df에 점수 추가
    df["추천점수"] = df[score_cols].sum(axis=1)

    # 검색 결과(sub)에 추천점수 결합
    sub = sub.merge(df[["주소", "추천점수"]], on="주소", how="left")

    # === 상위 K개 선택 ===
    sub = sub.sort_values("추천점수", ascending=False).head(topk)

    # === JSON 직렬화 안전 변환 ===
    items = []
    for _, row in sub.iterrows():
        items.append({
            "주소": row.get("주소"),
            "행정구역": row.get("행정구역"),
            "인구[명]": float(row.get("인구[명]", 0)),
            "인구천명당사업체수": float(row.get("인구천명당사업체수", 0)),
            "인구[명]_norm": float(row.get("인구[명]_norm", 0)),
            "인구천명당사업체수_norm": float(row.get("인구천명당사업체수_norm", 0)),
            "추천점수": float(row.get("추천점수", 0)),
        })

    return items
