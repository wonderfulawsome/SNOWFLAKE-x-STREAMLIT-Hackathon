import os, streamlit as st, snowflake.connector as sf
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── 기본 UI 세팅 ───────────────────────────────────────────────────
os.environ["STREAMLIT_SERVER_HEALTH_CHECK_ENABLED"] = "false"
st.set_page_config(page_title="서울시 감성 지수 대시보드", layout="wide")
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
st.markdown('<style>*{font-family:"Malgun Gothic",sans-serif!important;}</style>', unsafe_allow_html=True)
st.write("🚀 Streamlit 앱 시작!")

# ──── NEW: Matplotlib ‧ Streamlit 한글 폰트 설정 ────
import matplotlib
matplotlib.rcParams["font.family"] = "Malgun Gothic"   # 윈도 기본 한글 폰트
matplotlib.rcParams["axes.unicode_minus"] = False      # - 기호 깨짐 방지
import streamlit as st

st.markdown(
    """
    <style>
    /* Streamlit 기본 글꼴을 한글 지원 폰트로 변경 */
    * { font-family: "Malgun Gothic", sans-serif !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Snowflake 연결 ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_conn():
    s = st.secrets["snowflake"]
    return sf.connect(
        user=s["user"], password=s["password"], account=s["account"],
        role=s.get("role", "ACCOUNTADMIN"), warehouse="COMPUTE_WH",
        database="SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS",
        schema="GRANDATA", ocsp_fail_open=True, insecure_mode=True
    )

def run_sql(sql: str) -> pd.DataFrame:
    with get_conn().cursor() as cur:
        cur.execute(sql)
        return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])

# ── 쿼리 & 로드 ────────────────────────────────────────────────────
LIMIT = 50_000
BASE  = "SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA"
SQL   = {
    "fp"   : f"SELECT * FROM {BASE}.FLOATING_POPULATION_INFO LIMIT {LIMIT}",
    "card" : f"SELECT * FROM {BASE}.CARD_SALES_INFO           LIMIT {LIMIT}",
    "asset": f"SELECT * FROM {BASE}.ASSET_INCOME_INFO         LIMIT {LIMIT}",
    "scco" : f"SELECT * FROM {BASE}.M_SCCO_MST"
}

# ── 전처리 ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def preprocess() -> pd.DataFrame:
    fp, card, asset, scco = (run_sql(SQL["fp"]), run_sql(SQL["card"]),
                             run_sql(SQL["asset"]), run_sql(SQL["scco"]))

    df = (fp
          .merge(card , how="left", on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP",
                                        "GENDER","TIME_SLOT","WEEKDAY_WEEKEND"])
          .merge(asset, how="left", on=["STANDARD_YEAR_MONTH","DISTRICT_CODE",
                                        "AGE_GROUP","GENDER"])
          .sample(frac=0.01, random_state=42))             # 1 % 샘플

    df["DISTRICT_KOR_NAME"] = df["DISTRICT_CODE"].map(
        scco.drop_duplicates("DISTRICT_CODE")
            .set_index("DISTRICT_CODE")["DISTRICT_KOR_NAME"]
    )

    # ★ 숫자 열 전부 float 로 변환
    num_cols = df.columns.difference(["STANDARD_YEAR_MONTH","DISTRICT_CODE",
                                      "AGE_GROUP","GENDER","TIME_SLOT",
                                      "WEEKDAY_WEEKEND","DISTRICT_KOR_NAME"])
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # ─ 파생 변수
    df["전체인구"] = df[["RESIDENTIAL_POPULATION","WORKING_POPULATION","VISITING_POPULATION"]].sum(axis=1)

    sales_cols = ["FOOD_SALES","COFFEE_SALES","BEAUTY_SALES","ENTERTAINMENT_SALES",
                  "SPORTS_CULTURE_LEISURE_SALES","TRAVEL_SALES","CLOTHING_ACCESSORIES_SALES"]
    df["엔터전체매출"] = df[sales_cols].sum(axis=1)
    df["소비활력지수"] = df["엔터전체매출"] / df["전체인구"].replace(0, np.nan)
    df["유입지수"]    = df["VISITING_POPULATION"] / (
        df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"]).replace(0, np.nan)
    df["엔터매출비율"] = df["엔터전체매출"] / df["TOTAL_SALES"].replace(0, np.nan)

    cnt_cols = ["FOOD_COUNT","COFFEE_COUNT","BEAUTY_COUNT","ENTERTAINMENT_COUNT",
                "SPORTS_CULTURE_LEISURE_COUNT","TRAVEL_COUNT","CLOTHING_ACCESSORIES_COUNT"]
    df["엔터전체방문자수"] = df[cnt_cols].sum(axis=1)
    df["엔터방문자비율"] = df["엔터전체방문자수"] / df["TOTAL_COUNT"].replace(0, np.nan)
    df["엔터활동밀도"]  = df["엔터전체매출"] / df["전체인구"].replace(0, np.nan)
    df["엔터매출밀도"]  = df["엔터전체매출"] / df["엔터전체방문자수"].replace(0, np.nan)

    # ─ FEEL_IDX
    emo_vars = ["엔터전체매출","소비활력지수","유입지수","엔터매출비율",
                "엔터전체방문자수","엔터방문자비율","엔터활동밀도","엔터매출밀도"]
    df["FEEL_IDX"] = np.nan
    X = df[emo_vars].dropna()
    if not X.empty:
        pc1 = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(X))
        df.loc[X.index, "FEEL_IDX"] = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-9)

    return df

data = preprocess()
st.title("서울시 인스타 감성 지수 분석")

if data.empty:
    st.error("데이터가 없습니다. LIMIT/샘플 비율을 늘려 보세요.")
    st.stop()

# ── 사이드바 필터 ──────────────────────────────────────────────────
with st.sidebar:
    districts = st.multiselect("행정동", sorted(data["DISTRICT_KOR_NAME"].dropna().unique()))
    ages      = st.multiselect("연령대",  sorted(data["AGE_GROUP"].unique()))
    gender    = st.multiselect("성별", ["M","F"])

mask = pd.Series(True, index=data.index)
if districts: mask &= data["DISTRICT_KOR_NAME"].isin(districts)
if ages:      mask &= data["AGE_GROUP"].isin(ages)
if gender:    mask &= data["GENDER"].isin(gender)

view = data[mask]
if view.empty:
    st.warning("선택 조건에 맞는 데이터가 없습니다.")
    st.stop()

# ── 요약 지표 & 그래프(생략 부분 동일) ─────────────────────────────
c1,c2,c3 = st.columns(3)
c1.metric("평균 FEEL_IDX",     f"{view['FEEL_IDX'].mean():.2f}")
c2.metric("평균 소비활력지수", f"{view['소비활력지수'].mean():.2f}")
c3.metric("평균 유입지수",     f"{view['유입지수'].mean():.2f}")

tab1,tab2,tab3 = st.tabs(["지수 상위 지역","성별·연령 분석","산점도"])

with tab1:
    top = view.groupby("DISTRICT_KOR_NAME")["소비활력지수"].mean().nlargest(20)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=top.index, y=top.values, palette="rocket", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig)

with tab2:
    agg = view.groupby(["AGE_GROUP","GENDER"])["TOTAL_SALES"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=agg, x="AGE_GROUP", y="TOTAL_SALES",
                hue="GENDER", palette={"M":"#3498db","F":"#e75480"}, ax=ax)
    st.pyplot(fig)

with tab3:
    xx = st.selectbox("X변수", ["엔터전체매출","소비활력지수","유입지수","엔터전체방문자수"])
    yy = st.selectbox("Y변수", ["FEEL_IDX","엔터활동밀도","엔터매출비율"])
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=view, x=xx, y=yy,
                    hue="FEEL_IDX", palette="viridis", alpha=0.6, ax=ax)
    st.pyplot(fig)

st.divider(); st.caption("데이터 출처 · Snowflake Marketplace")
