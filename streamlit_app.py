import streamlit as st
import snowflake.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 기본 세팅
st.set_page_config(page_title="서울시 감성 지수 대시보드", layout="wide")
sns.set_style("whitegrid")

# ── 한글 폰트 설정 ──
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

# ── Snowflake 연결 & 쿼리 함수 ──
def get_conn():
    return snowflake.connector.connect(
        user=st.secrets["user"],
        password=st.secrets["password"],
        account=st.secrets["account"],
        warehouse="COMPUTE_WH",
        ocsp_fail_open=True,
        insecure_mode=True
    )

@st.cache_data(show_spinner=False)
def load_query(q: str) -> pd.DataFrame:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(q)
    df = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
    cur.close()
    conn.close()
    return df

# ── 쿼리 정의 ──
Q_FP    = """SELECT * FROM ...FLOATING_POPULATION_INFO"""
Q_CARD  = """SELECT * FROM ...CARD_SALES_INFO"""
Q_ASSET = """SELECT * FROM ...ASSET_INCOME_INFO"""
Q_SCCO  = """SELECT * FROM ...M_SCCO_MST"""

@st.cache_data(show_spinner=True)
def load_all():
    return (
        load_query(Q_FP),
        load_query(Q_CARD),
        load_query(Q_ASSET),
        load_query(Q_SCCO)
    )

# ── 전처리 ──
def preprocess() -> pd.DataFrame:
    fp, card, asset, scco = load_all()
    # fp + card
    df = pd.merge(
        fp, card,
        on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER","TIME_SLOT","WEEKDAY_WEEKEND"],
        how="inner", validate="m:m"
    )
    # + asset
    df = pd.merge(
        df, asset,
        on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER"],
        how="inner", validate="m:m"
    )
    # 중복 제거
    df = df.drop(columns=[c for c in df.columns if c.endswith(("_x","_y"))])
    # district 이름
    name_map = (
        scco[["DISTRICT_CODE","DISTRICT_KOR_NAME"]]
        .drop_duplicates("DISTRICT_CODE")
        .set_index("DISTRICT_CODE")["DISTRICT_KOR_NAME"]
        .to_dict()
    )
    df["DISTRICT_KOR_NAME"] = df["DISTRICT_CODE"].map(name_map)
    # 파생
    df["전체인구"]     = df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"] + df["VISITING_POPULATION"]
    df["엔터전체매출"] = (
        df["FOOD_SALES"] + df["COFFEE_SALES"] + df["BEAUTY_SALES"]
        + df["ENTERTAINMENT_SALES"] + df["SPORTS_CULTURE_LEISURE_SALES"]
        + df["TRAVEL_SALES"] + df["CLOTHING_ACCESSORIES_SALES"]
    )
    df["소비활력지수"] = df["엔터전체매출"] / df["전체인구"].replace(0, np.nan)
    df["유입지수"]     = df["VISITING_POPULATION"] / (df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"]).replace(0, np.nan)
    df["엔터매출비율"] = df["엔터전체매출"] / df["TOTAL_SALES"].replace(0, np.nan)
    cnt_cols = [
        "FOOD_COUNT","COFFEE_COUNT","BEAUTY_COUNT","ENTERTAINMENT_COUNT",
        "SPORTS_CULTURE_LEISURE_COUNT","TRAVEL_COUNT","CLOTHING_ACCESSORIES_COUNT"
    ]
    df["엔터전체방문자수"] = df[cnt_cols].sum(axis=1)
    df["엔터방문자비율"]   = df["엔터전체방문자수"] / df["TOTAL_COUNT"].replace(0, np.nan)
    df["엔터활동밀도"]     = df["엔터전체매출"] / df["전체인구"].replace(0, np.nan)
    df["엔터매출밀도"]     = df["엔터전체매출"] / df["엔터전체방문자수"].replace(0, np.nan)
    # FEEL_IDX
    emo_vars = [
        "엔터전체매출","소비활력지수","유입지수","엔터매출비율",
        "엔터전체방문자수","엔터방문자비율","엔터활동밀도","엔터매출밀도"
    ]
    X = df[emo_vars].dropna()
    pc1 = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(X))
    pc1n = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-9)
    df.loc[X.index, "FEEL_IDX"] = pc1n.ravel()
    return df.sample(frac=0.01, random_state=42)

@st.cache_data(show_spinner=True)
def get_data():
    return preprocess()

data = get_data()

# ─────────────────────────────
# 사이드바: 필터
# ─────────────────────────────
with st.sidebar:
    districts = st.multiselect("행정동", sorted(data["DISTRICT_KOR_NAME"].dropna().unique()))
    age_groups = st.multiselect("연령대", sorted(data["AGE_GROUP"].unique()))
    gender = st.multiselect("성별", ["M", "F"])
    
# 필터링
mask = (
    (data["DISTRICT_KOR_NAME"].isin(districts) if districts else True) &
    (data["AGE_GROUP"].isin(age_groups)       if age_groups else True) &
    (data["GENDER"].isin(gender)              if gender else True)
)
view = data[mask]

st.title("서울시 인스타 감성 지수 분석")

# ─────────────────────────────
# 분석 유형 선택
# ─────────────────────────────
option = st.selectbox(
    "분석 항목을 선택하세요",
    ["지수 상위 지역", "성별·연령 분석", "산점도"]
)

# ─────────────────────────────
# 1) 지수 상위 지역
# ─────────────────────────────
if option == "지수 상위 지역":
    st.subheader("FEEL_IDX 상위 20개 행정동")
    top20 = view.groupby("DISTRICT_KOR_NAME")["FEEL_IDX"].mean().nlargest(20)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=top20.values, y=top20.index, palette="rocket", ax=ax)
    ax.set_xlabel("FEEL_IDX")
    st.pyplot(fig)

# ─────────────────────────────
# 2) 성별·연령 분석
# ─────────────────────────────
elif option == "성별·연령 분석":
    st.subheader("연령대·성별별 평균 FEEL_IDX")
    grp = view.groupby(["AGE_GROUP","GENDER"])["FEEL_IDX"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=grp, x="AGE_GROUP", y="FEEL_IDX", hue="GENDER", ax=ax)
    st.pyplot(fig)

# ─────────────────────────────
# 3) 산점도
# ─────────────────────────────
else:
    st.subheader("FEEL_IDX vs 엔터전체매출")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=view, x="엔터전체매출", y="FEEL_IDX", hue="AGE_GROUP", alpha=0.6, ax=ax)
    st.pyplot(fig)

st.sidebar.caption("© 2025 Park Sungsoo")
