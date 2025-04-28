import os, streamlit as st, snowflake.connector as sf
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ───────── 기본 세팅 ─────────
os.environ["STREAMLIT_SERVER_HEALTH_CHECK_ENABLED"] = "false"
st.set_page_config(page_title="서울시 감성 지수 대시보드", layout="wide")
sns.set_style("whitegrid")
import matplotlib
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
st.markdown('<style>*{font-family:"Malgun Gothic",sans-serif!important;}</style>', unsafe_allow_html=True)
st.write("🚀 Streamlit 앱 시작!")

# ───────── Snowflake 연결 ─────────
@st.cache_resource(show_spinner=False)
def get_conn():
    s = st.secrets["snowflake"]
    return sf.connect(
        user=s["user"], password=s["password"], account=s["account"],
        warehouse="COMPUTE_WH", ocsp_fail_open=True, insecure_mode=True
    )

def load(q: str) -> pd.DataFrame:
    with get_conn().cursor() as cur:
        cur.execute(q)
        return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])

# ───────── 쿼리 (LIMIT 3000) ─────────
BASE  = "SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA"
LIMIT = 3000
Q_FP   = f"SELECT * FROM {BASE}.FLOATING_POPULATION_INFO LIMIT {LIMIT}"
Q_CARD = f"SELECT * FROM {BASE}.CARD_SALES_INFO           LIMIT {LIMIT}"
Q_ASSET= f"SELECT * FROM {BASE}.ASSET_INCOME_INFO         LIMIT {LIMIT}"
Q_SCCO = f"SELECT * FROM {BASE}.M_SCCO_MST"

# ───────── 전처리 ─────────
@st.cache_data(show_spinner=True)
def preprocess() -> pd.DataFrame:
    fp, card, asset, scco = load(Q_FP), load(Q_CARD), load(Q_ASSET), load(Q_SCCO)

    fp_card = pd.merge(
        fp, card, how="left",
        on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER","TIME_SLOT","WEEKDAY_WEEKEND"]
    )
    data = pd.merge(
        fp_card, asset, how="left",
        on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER"]
    )

    data["DISTRICT_KOR_NAME"] = data["DISTRICT_CODE"].map(
        scco.drop_duplicates("DISTRICT_CODE")
            .set_index("DISTRICT_CODE")["DISTRICT_KOR_NAME"]
    )

    # 파생 변수
    data["전체인구"] = data[
        ["RESIDENTIAL_POPULATION","WORKING_POPULATION","VISITING_POPULATION"]
    ].sum(axis=1)

    sales_cols = [
        "FOOD_SALES","COFFEE_SALES","BEAUTY_SALES","ENTERTAINMENT_SALES",
        "SPORTS_CULTURE_LEISURE_SALES","TRAVEL_SALES","CLOTHING_ACCESSORIES_SALES"
    ]
    data["엔터전체매출"] = data[sales_cols].sum(axis=1)
    data["소비활력지수"] = data["엔터전체매출"] / data["전체인구"].replace(0, np.nan)
    data["유입지수"]    = data["VISITING_POPULATION"] / (
        data["RESIDENTIAL_POPULATION"] + data["WORKING_POPULATION"]
    ).replace(0, np.nan)
    data["엔터매출비율"] = data["엔터전체매출"] / data["TOTAL_SALES"].replace(0, np.nan)

    cnt_cols = [
        "FOOD_COUNT","COFFEE_COUNT","BEAUTY_COUNT","ENTERTAINMENT_COUNT",
        "SPORTS_CULTURE_LEISURE_COUNT","TRAVEL_COUNT","CLOTHING_ACCESSORIES_COUNT"
    ]
    data["엔터전체방문자수"] = data[cnt_cols].sum(axis=1)
    data["엔터방문자비율"] = data["엔터전체방문자수"] / data["TOTAL_COUNT"].replace(0, np.nan)
    data["엔터활동밀도"]  = data["엔터전체매출"] / data["전체인구"].replace(0, np.nan)
    data["엔터매출밀도"]  = data["엔터전체매출"] / data["엔터전체방문자수"].replace(0, np.nan)

    emo_vars = [
        "엔터전체매출","소비활력지수","유입지수","엔터매출비율",
        "엔터전체방문자수","엔터방문자비율","엔터활동밀도","엔터매출밀도"
    ]
    X = (
        data[emo_vars]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if not X.empty:
        pc1 = PCA(n_components=1).fit_transform(
            StandardScaler().fit_transform(X)
        )
        data.loc[X.index, "FEEL_IDX"] = (
            (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-9)
        )
    return data

# ───────── 데이터 로드 ─────────
data = preprocess()
st.title("서울시 인스타 감성 지수 분석")

if data.empty:
    st.error("데이터가 없습니다.")
    st.stop()

# ───────── 사이드바 ─────────
with st.sidebar:
    districts = st.multiselect("행정동", sorted(data["DISTRICT_KOR_NAME"].dropna().unique()), [])
    ages      = st.multiselect("연령대",  sorted(data["AGE_GROUP"].unique()), [])
    gender    = st.multiselect("성별", ["M","F"], [])

mask = pd.Series(True, index=data.index)            # ← 빈 필터 대비 기본 시리즈
if districts:
    mask &= data["DISTRICT_KOR_NAME"].isin(districts)
if ages:
    mask &= data["AGE_GROUP"].isin(ages)
if gender:
    mask &= data["GENDER"].isin(gender)

view = data[mask]

# ───────── 요약 지표 ─────────
c1, c2, c3 = st.columns(3)
c1.metric("평균 FEEL_IDX",     f"{view['FEEL_IDX'].mean():.2f}")
c2.metric("평균 소비활력지수", f"{view['소비활력지수'].mean():.2f}")
c3.metric("평균 유입지수",     f"{view['유입지수'].mean():.2f}")

tab1, tab2, tab3 = st.tabs(["지수 상위 지역", "성별·연령 분석", "산점도"])

with tab1:
    top = view.groupby("DISTRICT_KOR_NAME")["소비활력지수"].mean().nlargest(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top.index, y=top.values, palette="rocket", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

with tab2:
    agg = (
        view.groupby(["AGE_GROUP", "GENDER"])["TOTAL_SALES"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=agg, x="AGE_GROUP", y="TOTAL_SALES",
        hue="GENDER", palette={"M": "#3498db", "F": "#e75480"}, ax=ax
    )
    st.pyplot(fig)

with tab3:
    xx = st.selectbox("X변수", ["엔터전체매출","소비활력지수","유입지수","엔터전체방문자수"])
    yy = st.selectbox("Y변수", ["FEEL_IDX","엔터활동밀도","엔터매출비율"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=view, x=xx, y=yy,
        hue="FEEL_IDX", palette="viridis", alpha=0.6, ax=ax
    )
    st.pyplot(fig)

st.divider()
st.caption("데이터 출처 · Snowflake Marketplace")
