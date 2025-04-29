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

# ──── 한글 폰트 설정 ────
import matplotlib
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
st.markdown(
    """
    <style>
      * { font-family: "Malgun Gothic", sans-serif !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────
DATA_DIR = Path(__file__).parent / "data"

@st.cache_data(show_spinner=False)
def load_data():
    fp    = pd.read_csv(DATA_DIR / "floating population.csv")
    card  = pd.read_csv(DATA_DIR / "Shinhan card sales.csv")
    asset = pd.read_csv(DATA_DIR / "asset income.csv")
    scco  = pd.read_csv(DATA_DIR / "scco.csv")
    return fp, card, asset, scco

try:
    fp_df, card_df, asset_df, scco_df = load_data()
except FileNotFoundError as e:
    st.error(f"데이터 파일을 찾을 수 없습니다: {e}")
    st.stop()

# ─────────────────────────────
# 2. 전처리 & 파생변수
# ─────────────────────────────
def preprocess(fp, card, asset, scco):
    # merge fp + card
    df = pd.merge(
        fp, card,
        on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER","TIME_SLOT","WEEKDAY_WEEKEND"],
        how="inner", validate="m:m"
    )
    # merge + asset
    df = pd.merge(
        df, asset,
        on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER"],
        how="inner", validate="m:m"
    )
    # 중복 컬럼 제거
    dup = [c for c in df.columns if c.endswith(("_x","_y"))]
    df = df.drop(columns=dup)
    # district name 매핑
    scco_map = (scco[["DISTRICT_CODE","DISTRICT_KOR_NAME"]]
                .drop_duplicates("DISTRICT_CODE")
                .set_index("DISTRICT_CODE")["DISTRICT_KOR_NAME"]
                .to_dict())
    df["DISTRICT_KOR_NAME"] = df["DISTRICT_CODE"].map(scco_map)
    # 파생 변수
    df["전체인구"]     = df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"] + df["VISITING_POPULATION"]
    df["엔터전체매출"] = (df["FOOD_SALES"] + df["COFFEE_SALES"] + df["BEAUTY_SALES"]
                          + df["ENTERTAINMENT_SALES"] + df["SPORTS_CULTURE_LEISURE_SALES"]
                          + df["TRAVEL_SALES"] + df["CLOTHING_ACCESSORIES_SALES"])
    df["소비활력지수"] = df["엔터전체매출"] / df["전체인구"].replace(0, np.nan)
    df["유입지수"]     = df["VISITING_POPULATION"] / (df["RESIDENTIAL_POPULATION"]+df["WORKING_POPULATION"]).replace(0, np.nan)
    df["엔터매출비율"] = df["엔터전체매출"] / df["TOTAL_SALES"].replace(0, np.nan)
    cnt_cols = ["FOOD_COUNT","COFFEE_COUNT","BEAUTY_COUNT","ENTERTAINMENT_COUNT",
                "SPORTS_CULTURE_LEISURE_COUNT","TRAVEL_COUNT","CLOTHING_ACCESSORIES_COUNT"]
    df["엔터전체방문자수"] = df[cnt_cols].sum(axis=1)
    df["엔터매출밀도"]   = df["엔터전체매출"] / df["엔터전체방문자수"].replace(0, np.nan)
    # PCA 기반 FEEL_IDX
    emo = ["엔터전체매출","소비활력지수","유입지수","엔터매출비율","엔터전체방문자수","엔터매출밀도"]
    X   = df[emo].dropna()
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(StandardScaler().fit_transform(X))
    pc1n = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-9)
    df.loc[X.index, "FEEL_IDX"] = pc1n.ravel()
    # 샘플링
    return df.sample(frac=0.01, random_state=42)

data = preprocess(fp_df, card_df, asset_df, scco_df)

# ─────────────────────────────
# 3. UI: 필터
# ─────────────────────────────
st.title("서울시 인스타 감성 지수 대시보드")
with st.sidebar:
    sel_d = st.multiselect("행정동", sorted(data["DISTRICT_KOR_NAME"].dropna().unique()))
    sel_a = st.multiselect("연령대", sorted(data["AGE_GROUP"].unique()))
    sel_g = st.multiselect("성별", ["M","F"])

mask = (
    (data["DISTRICT_KOR_NAME"].isin(sel_d) if sel_d else True) &
    (data["AGE_GROUP"].isin(sel_a) if sel_a else True) &
    (data["GENDER"].isin(sel_g) if sel_g else True)
)
view = data[mask]

# ─────────────────────────────
# 4. 요약 지표
# ─────────────────────────────
st.subheader("요약 지표")
c1, c2, c3 = st.columns(3)
c1.metric("평균 FEEL_IDX",      f"{view['FEEL_IDX'].mean():.2f}")
c2.metric("평균 소비활력지수",   f"{view['소비활력지수'].mean():.2f}")
c3.metric("평균 유입지수",      f"{view['유입지수'].mean():.2f}")

# ─────────────────────────────
# 5. 탭별 시각화
# ─────────────────────────────
tab1, tab2, tab3 = st.tabs(["지수 상위 지역","성별·연령 분석","산점도"])

with tab1:
    top20 = view.groupby("DISTRICT_KOR_NAME")["소비활력지수"].mean().nlargest(20)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=top20.index, y=top20.values, palette="rocket", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

with tab2:
    grp = view.groupby(["AGE_GROUP","GENDER"])["TOTAL_SALES"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=grp, x="AGE_GROUP", y="TOTAL_SALES", hue="GENDER", ax=ax)
    st.pyplot(fig)

with tab3:
    x = st.selectbox("X축 변수", ["엔터전체매출","소비활력지수","유입지수","엔터전체방문자수"])
    y = st.selectbox("Y축 변수", ["FEEL_IDX","엔터매출비율","엔터매출밀도"])
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=view, x=x, y=y, hue="FEEL_IDX", palette="viridis", alpha=0.6, ax=ax)
    st.pyplot(fig)

st.sidebar.caption("© 2025 Park Sungsoo")
