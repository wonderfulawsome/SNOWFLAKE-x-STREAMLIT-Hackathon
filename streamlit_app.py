# streamlit_app.py
# =========================================================
# 0. 기본 세팅 + 한글(부호 포함) 폰트 -----------------------
# =========================================================
import os, urllib.request, json, streamlit as st
import snowflake.connector as sf
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

os.environ["STREAMLIT_SERVER_HEALTH_CHECK_ENABLED"] = "false"
st.set_page_config(page_title="서울시 인스타 감성 지수", layout="wide")
sns.set_style("whitegrid")

# ▶ Google Fonts Noto Sans KR → Matplotlib + Streamlit
FONT_URL = ("https://github.com/googlefonts/noto-cjk/raw/main/"
            "Sans/OTF/Korean/NotoSansCJKkr-Regular.otf")
FONT_PATH = "/tmp/NotoSansKR.otf"
if not os.path.exists(FONT_PATH):
    urllib.request.urlretrieve(FONT_URL, FONT_PATH)
    font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = "Noto Sans CJK KR"
plt.rcParams["axes.unicode_minus"] = False
st.markdown("""
<style>
@font-face{font-family:"Noto Sans KR";
 src:url("https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap");}
*{font-family:"Noto Sans KR",sans-serif!important;}
</style>
""", unsafe_allow_html=True)

st.write("🚀 **Streamlit 앱 시작!**")

# =========================================================
# 1. Snowflake 연결 & 쿼리 ---------------------------------
# =========================================================
@st.cache_resource(show_spinner=False)
def get_conn():
    s = st.secrets["snowflake"]
    return sf.connect(
        user=s["user"], password=s["password"], account=s["account"],
        role=s.get("role","ACCOUNTADMIN"), warehouse=s.get("warehouse","COMPUTE_WH"),
        database="SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS",
        schema="GRANDATA", ocsp_fail_open=True, insecure_mode=True
    )

def run(sql:str)->pd.DataFrame:
    with get_conn().cursor() as cur:
        cur.execute(sql)
        return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])

BASE = "SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA"
SQL  = lambda tbl: f"SELECT * FROM {BASE}.{tbl} SAMPLE BERNOULLI (1) "  # 1 % 표본

# =========================================================
# 2. 전처리 -----------------------------------------------
# =========================================================
@st.cache_data(show_spinner=True)
def load_and_prepare() -> pd.DataFrame:
    fp     = run(SQL("FLOATING_POPULATION_INFO"))
    card   = run(SQL("CARD_SALES_INFO"))
    asset  = run(SQL("ASSET_INCOME_INFO"))
    scco   = run(SQL("M_SCCO_MST"))

    # ---------------- 병합 ----------------
    df = (fp
          .merge(card , how="left", on=["STANDARD_YEAR_MONTH","DISTRICT_CODE",
                                        "AGE_GROUP","GENDER","TIME_SLOT","WEEKDAY_WEEKEND"])
          .merge(asset, how="left", on=["STANDARD_YEAR_MONTH","DISTRICT_CODE",
                                        "AGE_GROUP","GENDER"]))
    df["DISTRICT_KOR_NAME"] = df["DISTRICT_CODE"].map(
        scco.drop_duplicates("DISTRICT_CODE").set_index("DISTRICT_CODE")["DISTRICT_KOR_NAME"])

    # ---------------- 숫자형 변환 ----------------
    obj_cols = ["STANDARD_YEAR_MONTH","DISTRICT_CODE",
                "AGE_GROUP","GENDER","TIME_SLOT","WEEKDAY_WEEKEND","DISTRICT_KOR_NAME"]
    num_cols = df.columns.difference(obj_cols)
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # ---------------- 파생 변수 ----------------
    df["전체인구"] = df[["RESIDENTIAL_POPULATION","WORKING_POPULATION",
                         "VISITING_POPULATION"]].sum(axis=1)

    sales_cols = ["FOOD_SALES","COFFEE_SALES","BEAUTY_SALES","ENTERTAINMENT_SALES",
                  "SPORTS_CULTURE_LEISURE_SALES","TRAVEL_SALES","CLOTHING_ACCESSORIES_SALES"]
    df["엔터전체매출"] = df[sales_cols].sum(axis=1)
    df["유입지수"]    = df["VISITING_POPULATION"] / (
                        df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"]).replace(0,np.nan)
    df["엔터매출비율"] = df["엔터전체매출"] / df["TOTAL_SALES"].replace(0,np.nan)

    cnt_cols = ["FOOD_COUNT","COFFEE_COUNT","BEAUTY_COUNT","ENTERTAINMENT_COUNT",
                "SPORTS_CULTURE_LEISURE_COUNT","TRAVEL_COUNT","CLOTHING_ACCESSORIES_COUNT"]
    df["엔터전체방문자수"] = df[cnt_cols].sum(axis=1)
    df["엔터방문자비율"] = df["엔터전체방문자수"] / df["TOTAL_COUNT"].replace(0,np.nan)
    df["엔터활동밀도"]  = df["엔터전체매출"] / df["전체인구"].replace(0,np.nan)
    df["엔터매출밀도"]  = df["엔터전체매출"] / df["엔터전체방문자수"].replace(0,np.nan)

    # ---------------- FEEL_IDX (PCA) ----------------
    emo_vars = ["엔터전체매출","유입지수","엔터매출비율",
                "엔터전체방문자수","엔터방문자비율","엔터활동밀도","엔터매출밀도"]
    df["FEEL_IDX"] = 0.0                                    # 기본값 0
    X = df[emo_vars].dropna()
    if not X.empty:
        pc1 = PCA(1).fit_transform(StandardScaler().fit_transform(X))
        df.loc[X.index, "FEEL_IDX"] = (pc1 - pc1.min())/(pc1.max()-pc1.min()+1e-9)

    return df

data = load_and_prepare()
st.title("서울시 인스타 감성 지수 분석")

if data.empty:
    st.error("❌ 샘플이 너무 작아 데이터가 없습니다. 샘플 비율을 올려 주세요!")
    st.stop()

# =========================================================
# 3. 사이드바 필터 -----------------------------------------
# =========================================================
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
    st.warning("선택한 조건에 해당하는 데이터가 없습니다. 필터를 완화해 보세요.")
    st.stop()

# =========================================================
# 4. 요약 지표 ---------------------------------------------
# =========================================================
c1,c2,c3 = st.columns(3)
c1.metric("평균 FEEL_IDX",     f"{view['FEEL_IDX'].mean():.3f}")
c2.metric("평균 소비활력지수", f"{view['엔터활동밀도'].mean():.2f}")
c3.metric("평균 유입지수",     f"{view['유입지수'].mean():.2f}")

# =========================================================
# 5. 그래프 ------------------------------------------------
# =========================================================
tab1,tab2,tab3 = st.tabs(["지수 상위 지역","성별·연령 분석","산점도"])

with tab1:
    top = view.groupby("DISTRICT_KOR_NAME")["엔터활동밀도"].mean().nlargest(20)
    if top.empty:
        st.info("그래프를 그릴 데이터가 없습니다.")
    else:
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x=top.index, y=top.values, palette="rocket", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("행정동"); ax.set_ylabel("엔터활동밀도")
        st.pyplot(fig)

with tab2:
    agg = view.groupby(["AGE_GROUP","GENDER"])["엔터전체매출"].mean().reset_index()
    if agg.empty:
        st.info("그래프를 그릴 데이터가 없습니다.")
    else:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=agg, x="AGE_GROUP", y="엔터전체매출",
                    hue="GENDER", palette={"M":"#3498db","F":"#e75480"}, ax=ax)
        st.pyplot(fig)

with tab3:
    x_opt = ["엔터전체매출","유입지수","엔터전체방문자수"]
    y_opt = ["FEEL_IDX","엔터활동밀도","엔터매출비율"]
    xx = st.selectbox("X축 변수", x_opt, index=0)
    yy = st.selectbox("Y축 변수", y_opt, index=0)
    base = view[[xx,yy,"FEEL_IDX"]].dropna()
    if base.empty:
        st.info("그래프를 그릴 데이터가 없습니다.")
    else:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(data=base, x=xx, y=yy,
                        hue="FEEL_IDX", palette="viridis", alpha=0.6, ax=ax)
        st.pyplot(fig)

st.divider()
st.caption("데이터 출처 · Snowflake Marketplace")
