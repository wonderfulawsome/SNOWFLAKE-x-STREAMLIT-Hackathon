import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import matplotlib.font_manager as fm
import matplotlib

# 기본 세팅
st.set_page_config(page_title="서울시 감성 지수 대시보드", layout="wide")
sns.set_style("whitegrid")

# ── 폰트 로드 ──
font_path = os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic.TTF")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    nanum_font_name = fm.FontProperties(fname=font_path).get_name()
    matplotlib.rcParams['font.family'] = nanum_font_name
    matplotlib.rcParams['axes.unicode_minus'] = False
else:
    st.warning("❌ NanumGothic.TTF 파일이 fonts 폴더에 없습니다.")

st.markdown(f"""
<style>
@font-face {{
    font-family: 'NanumGothic';
    src: url('fonts/NanumGothic.TTF') format('truetype');
}}
html, body, [class*="css"] {{
    font-family: 'NanumGothic', 'sans-serif';
}}
</style>
""", unsafe_allow_html=True)

# 데이터 로드
DATA_DIR = Path(__file__).parent / "data"

@st.cache_data(show_spinner=False)
def load_data():
    fp = pd.read_csv(DATA_DIR / "floating population.csv")
    card = pd.read_csv(DATA_DIR / "Shinhan card sales.csv")
    scco = pd.read_csv(DATA_DIR / "scco.csv")
    return fp, card, scco

fp_df, card_df, scco_df = load_data()

# 병합
merge_keys = ["PROVINCE_CODE", "CITY_CODE", "DISTRICT_CODE"]
df = pd.merge(fp_df, scco_df[merge_keys + ["DISTRICT_KOR_NAME"]], on=merge_keys, how="left")
df = pd.merge(df, card_df, on=merge_keys + ["AGE_GROUP", "GENDER", "STANDARD_YEAR_MONTH"], how="left")

# 필수 컬럼 확인
required_cols = ["RESIDENTIAL_POPULATION", "WORKING_POPULATION", "VISITING_POPULATION"]
if not all(col in df.columns for col in required_cols):
    st.error("❌ 인구 관련 컬럼이 누락되어 분석을 진행할 수 없습니다.")
    st.stop()

# 파생변수 생성
enter_count_columns = [
    'FOOD_COUNT', 'COFFEE_COUNT', 'BEAUTY_COUNT', 'ENTERTAINMENT_COUNT',
    'SPORTS_CULTURE_LEISURE_COUNT', 'TRAVEL_COUNT', 'CLOTHING_ACCESSORIES_COUNT'
]

sales_columns = [
    'FOOD_SALES', 'COFFEE_SALES', 'BEAUTY_SALES', 'ENTERTAINMENT_SALES',
    'SPORTS_CULTURE_LEISURE_SALES', 'TRAVEL_SALES', 'CLOTHING_ACCESSORIES_SALES'
]

df["전체인구"] = df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"] + df["VISITING_POPULATION"]
df["엔터전체매출"] = df[sales_columns].sum(axis=1)
df["엔터전체방문자수"] = df[enter_count_columns].sum(axis=1)
df["엔터활동밀도"] = df["엔터전체매출"] / df["전체인구"].replace(0, np.nan)
df["방문자1인당엔터매출"] = df["엔터전체매출"] / df["엔터전체방문자수"].replace(0, np.nan)
df["유입지수"] = df["VISITING_POPULATION"] / (df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"]).replace(0, np.nan)
df["엔터매출비율"] = df["엔터전체매출"] / df["TOTAL_SALES"].replace(0, np.nan)
df["엔터방문자비율"] = df["엔터전체방문자수"] / df["TOTAL_COUNT"].replace(0, np.nan)
df["엔터매출밀도"] = df["엔터전체매출"] / df["엔터전체방문자수"].replace(0, np.nan)
df["소비활력지수"] = df["VISITING_POPULATION"] / df["전체인구"].replace(0, np.nan)

# FEEL_IDX 계산
pca_vars = ["엔터전체매출", "유입지수", "엔터매출비율", "엔터전체방문자수", "엔터방문자비율", "엔터활동밀도", "엔터매출밀도"]
X = df[pca_vars].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=1)
pc1 = pca.fit_transform(X_scaled).ravel()
pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-9)
df.loc[X.index, "FEEL_IDX"] = pc1_norm

# 샘플링
data = df.sample(frac=0.01, random_state=42)

# UI
st.title("서울시 감성 지수 매출분석")

top_districts = data["DISTRICT_KOR_NAME"].value_counts().head(10).index.tolist()

with st.sidebar:
    st.markdown("## 🔎 필터")
    districts = st.multiselect("행정동 (상위 10개)", options=top_districts, default=top_districts)
    age_groups = st.multiselect("연령대", sorted(data["AGE_GROUP"].dropna().unique()), default=sorted(data["AGE_GROUP"].dropna().unique()))
    gender = st.multiselect("성별", ["M", "F"], default=["M", "F"])

mask = (
    data["DISTRICT_KOR_NAME"].isin(districts) &
    data["AGE_GROUP"].isin(age_groups) &
    data["GENDER"].isin(gender)
)
view = data.loc[mask]

# 요약 지표
st.subheader("요약 지표")
c1, c2, c3 = st.columns(3)

if not view.empty:
    c1.metric("평균 FEEL_IDX", f"{view['FEEL_IDX'].mean():.2f}")
    c2.metric("평균 소비활력지수", f"{view['소비활력지수'].mean():.2f}")
    c3.metric("평균 유입지수", f"{view['유입지수'].mean():.2f}")
else:
    c1.metric("평균 FEEL_IDX", "-")
    c2.metric("평균 소비활력지수", "-")
    c3.metric("평균 유입지수", "-")

# 탭 시각화
with st.tabs(["지수 상위 지역", "성별·연령 분석", "산점도"]) as (tab1, tab2, tab3):
    with tab1:
        st.subheader("지수 상위 지역 Top 10")
        if not view.empty:
            top = view.groupby("DISTRICT_KOR_NAME")["FEEL_IDX"].mean().sort_values(ascending=False).head(10)
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
            x_axis = st.selectbox("X축 변수", ["소비활력지수", "유입지수"])
            y_axis = st.selectbox("Y축 변수", ["FEEL_IDX"])
            if all(col in view.columns for col in [x_axis, y_axis]):
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=view, x=x_axis, y=y_axis, hue="FEEL_IDX", palette="viridis", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("선택한 컬럼이 데이터에 없습니다.")
        else:
            st.info("선택된 데이터가 없습니다.")

st.divider()
st.caption("© 2025 데이터 출처: snowflake park")
