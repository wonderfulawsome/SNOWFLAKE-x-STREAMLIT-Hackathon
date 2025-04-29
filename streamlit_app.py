import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─────────────────────────────
# 0. 기본 세팅
# ─────────────────────────────
st.set_page_config(page_title="서울시 감성 지수 대시보드", layout="wide")
sns.set_style("whitegrid")

# 한글 폰트 설정
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Malgun Gothic', 'Noto Sans KR', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────
DATA_DIR = Path(__file__).parent / "data"

@st.cache_data(show_spinner=False)
def load_data():
    fp = pd.read_csv(DATA_DIR / "floating population.csv")
    card = pd.read_csv(DATA_DIR / "Shinhan card sales.csv")
    scco = pd.read_csv(DATA_DIR / "scco.csv")
    return fp, card, scco

fp_df, card_df, scco_df = load_data()

# ─────────────────────────────
# 2. 병합 및 파생 변수 생성
# ─────────────────────────────
merge_keys = [
    "PROVINCE_CODE", "CITY_CODE", "DISTRICT_CODE",
    "STANDARD_YEAR_MONTH", "WEEKDAY_WEEKEND",
    "GENDER", "AGE_GROUP", "TIME_SLOT"
]

df = pd.merge(fp_df, card_df, on=merge_keys, how="inner")

# 행정동 이름 매핑
if "DISTRICT_KOR_NAME" in scco_df.columns:
    name_map = scco_df.set_index("DISTRICT_CODE")["DISTRICT_KOR_NAME"].to_dict()
    df["DISTRICT_KOR_NAME"] = df["DISTRICT_CODE"].map(name_map)
else:
    df["DISTRICT_KOR_NAME"] = df["DISTRICT_CODE"].astype(str)

# 파생 변수
df["전체인구"] = df["COUNT"]  # COUNT를 전체 인구로 사용
df["엔터전체매출"] = (
    df.get("FOOD_SALES", 0) + df.get("COFFEE_SALES", 0) + df.get("BEAUTY_SALES", 0) +
    df.get("ENTERTAINMENT_SALES", 0) + df.get("SPORTS_CULTURE_LEISURE_SALES", 0) +
    df.get("TRAVEL_SALES", 0) + df.get("CLOTHING_ACCESSORIES_SALES", 0)
)
df["소비활력지수"] = df["엔터전체매출"] / df["전체인구"].replace(0, np.nan)
df["유입지수"] = np.nan  # 계산 불가능하므로 NaN 처리
df["엔터매출비율"] = df["엔터전체매출"] / df["TOTAL_SALES"].replace(0, np.nan)

cnt_cols = [
    "FOOD_COUNT", "COFFEE_COUNT", "BEAUTY_COUNT", "ENTERTAINMENT_COUNT",
    "SPORTS_CULTURE_LEISURE_COUNT", "TRAVEL_COUNT", "CLOTHING_ACCESSORIES_COUNT"
]
df["엔터전체방문자수"] = df[cnt_cols].sum(axis=1)

# ─────────────────────────────
# 3. FEEL_IDX 계산
# ─────────────────────────────
X = df.drop(columns=merge_keys + ["DISTRICT_KOR_NAME"]).select_dtypes(include="number").dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=1)
pc1 = pca.fit_transform(X_scaled).ravel()
pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-9)

df.loc[X.index, "FEEL_IDX"] = pc1_norm

# 샘플링
data = df.sample(frac=0.01, random_state=42)

# ─────────────────────────────
# 4. UI 구성
# ─────────────────────────────
st.title("서울시 인스타 감성 지수 분석")

top_districts = data["DISTRICT_KOR_NAME"].value_counts().head(10).index.tolist()

with st.sidebar:
    st.markdown("## 🔎 필터")
    districts = st.multiselect("행정동 (상위 10개)", options=top_districts, default=top_districts)
    age_groups = st.multiselect("연령대", sorted(data["AGE_GROUP"].dropna().unique()), default=sorted(data["AGE_GROUP"].dropna().unique()))
    gender = st.multiselect("성별", ["M", "F"], default=["M", "F"])

districts_mask = data["DISTRICT_KOR_NAME"].isin(districts)
age_groups_mask = data["AGE_GROUP"].isin(age_groups)
gender_mask = data["GENDER"].isin(gender)
mask = districts_mask & age_groups_mask & gender_mask
view = data.loc[mask]

# ─────────────────────────────
# 5. 요약 지표
# ─────────────────────────────
st.subheader("요약 지표")
c1, c2, c3 = st.columns(3)

if not view.empty:
    c1.metric("평균 FEEL_IDX", f"{view['FEEL_IDX'].mean():.2f}")
    c2.metric("평균 소비활력지수", f"{view['소비활력지수'].mean():.2f}")
    c3.metric("평균 엔터매출비율", f"{view['엔터매출비율'].mean():.2f}")
else:
    c1.metric("평균 FEEL_IDX", "-")
    c2.metric("평균 소비활력지수", "-")
    c3.metric("평균 엔터매출비율", "-")

# ─────────────────────────────
# 6. 분석 탭
# ─────────────────────────────
tab1, tab2, tab3 = st.tabs(["지수 상위 지역", "성별·연령 분석", "산점도"])

with tab1:
    st.subheader("지수 상위 지역 Top 20")
    if not view.empty:
        top = view.groupby("DISTRICT_KOR_NAME")["FEEL_IDX"].mean().sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top.values, y=top.index, palette="rocket", ax=ax)
        st.pyplot(fig)
    else:
        st.info("선택된 데이터가 없습니다.")

with tab2:
    st.subheader("성별 · 연령대별 FEEL_IDX")
    if not view.empty:
        grp = view.groupby(["AGE_GROUP", "GENDER"])["FEEL_IDX"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=grp, x="AGE_GROUP", y="FEEL_IDX", hue="GENDER", palette={"M": "skyblue", "F": "lightpink"}, ax=ax)
        st.pyplot(fig)
    else:
        st.info("선택된 데이터가 없습니다.")

with tab3:
    st.subheader("산점도 분석")
    if not view.empty:
        x_axis = st.selectbox("X축 변수", ["엔터전체매출", "소비활력지수", "엔터전체방문자수"])
        y_axis = st.selectbox("Y축 변수", ["FEEL_IDX", "엔터매출비율"])
        if all(col in view.columns for col in [x_axis, y_axis, "FEEL_IDX"]):
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=view, x=x_axis, y=y_axis, hue="FEEL_IDX", palette="viridis", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("선택한 컬럼이 데이터에 없습니다.")
    else:
        st.info("선택된 데이터가 없습니다.")

st.divider()
st.caption("© 2025 데이터 출처: 리포지토리 data 폴더")
