# streamlit_app.py
import os, urllib.request, streamlit as st, snowflake.connector as sf
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import font_manager

# ────────────── 0. UI & 한글 폰트 ──────────────────────────────────────
os.environ["STREAMLIT_SERVER_HEALTH_CHECK_ENABLED"] = "false"
st.set_page_config(page_title="서울시 감성 지수 대시보드", layout="wide")
sns.set_style("whitegrid")

# ▶ Google Fonts 다운로드 → Matplotlib + Streamlit 적용
FONT_URL  = ("https://github.com/googlefonts/noto-cjk/raw/main/"
             "Sans/OTF/Korean/NotoSansCJKkr-Regular.otf")
FONT_PATH = "/tmp/NotoSansKR.otf"
if not os.path.exists(FONT_PATH):
    urllib.request.urlretrieve(FONT_URL, FONT_PATH)
    font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False

st.markdown("""
<style>
@font-face { font-family: "Noto Sans KR"; src: url("https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap"); }
* { font-family: "Noto Sans KR", sans-serif !important; }
</style>
""", unsafe_allow_html=True)

st.write("🚀 **Streamlit 앱 시작!**")

# ────────────── 1. Snowflake 연결 ─────────────────────────────────────
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

# ────────────── 2. 데이터 로드 & 전처리 ───────────────────────────────
LIMIT = 50_000   # → 1% 샘플 직전, Snowflake 쿼리 자체를 기·본·적·으·로 제한
BASE  = "SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA"
SQL   = {t: f"SELECT * FROM {BASE}.{tbl} LIMIT {LIMIT}" for t, tbl in {
            "fp"   : "FLOATING_POPULATION_INFO",
            "card" : "CARD_SALES_INFO",
            "asset": "ASSET_INCOME_INFO",
            "scco" : "M_SCCO_MST"}.items()}

@st.cache_data(show_spinner=True)
def preprocess() -> pd.DataFrame:
    fp, card, asset, scco = (run_sql(SQL["fp"]), run_sql(SQL["card"]),
                             run_sql(SQL["asset"]), run_sql(SQL["scco"]))

    df = (fp
          .merge(card , how="left", on=["STANDARD_YEAR_MONTH","DISTRICT_CODE",
                                        "AGE_GROUP","GENDER","TIME_SLOT","WEEKDAY_WEEKEND"])
          .merge(asset, how="left", on=["STANDARD_YEAR_MONTH","DISTRICT_CODE",
                                        "AGE_GROUP","GENDER"])
          .sample(frac=0.01, random_state=42))                      # 1 % 추출

    df["DISTRICT_KOR_NAME"] = df["DISTRICT_CODE"].map(
        scco.drop_duplicates("DISTRICT_CODE")
            .set_index("DISTRICT_CODE")["DISTRICT_KOR_NAME"]
    )

    # ▶ 숫자 열 전체 실수형 변환
    num_cols = df.select_dtypes(include="object").columns.difference(
        ["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER",
         "TIME_SLOT","WEEKDAY_WEEKEND","DISTRICT_KOR_NAME"])
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # ─ 파생 변수 계산
    df["전체인구"] = df[["RESIDENTIAL_POPULATION","WORKING_POPULATION",
                         "VISITING_POPULATION"]].sum(axis=1)

    sales_cols = ["FOOD_SALES","COFFEE_SALES","BEAUTY_SALES","ENTERTAINMENT_SALES",
                  "SPORTS_CULTURE_LEISURE_SALES","TRAVEL_SALES",
                  "CLOTHING_ACCESSORIES_SALES"]
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

    # ─ FEEL_IDX (PCA 1-축)
    emo_vars = ["엔터전체매출","소비활력지수","유입지수","엔터매출비율",
                "엔터전체방문자수","엔터방문자비율","엔터활동밀도","엔터매출밀도"]
    df["FEEL_IDX"] = np.nan
    X = df[emo_vars].dropna()
    if not X.empty:
        pc1 = PCA(1).fit_transform(StandardScaler().fit_transform(X))
        df.loc[X.index, "FEEL_IDX"] = (pc1 - pc1.min())/(pc1.max()-pc1.min()+1e-9)

    return df

data = preprocess()
st.title("서울시 인스타 감성 지수 분석")

if data.empty:
    st.error("Snowflake에서 불러온 데이터가 없습니다 - LIMIT 값을 늘려 보세요.")
    st.stop()

# ────────────── 3. 사이드바 필터 ─────────────────────────────────────
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
    st.warning("선택한 조건에 해당하는 데이터가 없습니다. 필터를 완화해 보세요!")
    st.stop()

# ────────────── 4. 요약 지표 & 그래프 ───────────────────────────────
c1,c2,c3 = st.columns(3)
c1.metric("평균 FEEL_IDX",     f"{view['FEEL_IDX'].mean():.2f}")
c2.metric("평균 소비활력지수", f"{view['소비활력지수'].mean():.2f}")
c3.metric("평균 유입지수",     f"{view['유입지수'].mean():.2f}")

tab1,tab2,tab3 = st.tabs(["지수 상위 지역","성별·연령 분석","산점도"])

with tab1:
    top = view.groupby("DISTRICT_KOR_NAME")["소비활력지수"].mean().nlargest(20)
    if top.empty:
        st.info("그래프를 그릴 데이터가 없습니다.")
    else:
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x=top.index, y=top.values, palette="rocket", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)

with tab2:
    agg = view.groupby(["AGE_GROUP","GENDER"])["TOTAL_SALES"].mean().reset_index()
    if agg.empty:
        st.info("그래프를 그릴 데이터가 없습니다.")
    else:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=agg, x="AGE_GROUP", y="TOTAL_SALES",
                    hue="GENDER", palette={"M":"#3498db","F":"#e75480"}, ax=ax)
        st.pyplot(fig)

with tab3:
    xx = st.selectbox("X변수", ["엔터전체매출","소비활력지수","유입지수","엔터전체방문자수"])
    yy = st.selectbox("Y변수", ["FEEL_IDX","엔터활동밀도","엔터매출비율"])
    if view[[xx,yy]].dropna().empty:
        st.info("그래프를 그릴 데이터가 없습니다.")
    else:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(data=view, x=xx, y=yy,
                        hue="FEEL_IDX", palette="viridis", alpha=0.6, ax=ax)
        st.pyplot(fig)

st.divider(); st.caption("데이터 출처 · Snowflake Marketplace")
