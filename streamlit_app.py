###############################################################################
# streamlit_app.py  (2025-04-29 版)
###############################################################################
import os, sys, traceback, warnings
import streamlit as st
import snowflake.connector as sf
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ───────────────── 1. 기본 환경 ─────────────────
os.environ["STREAMLIT_SERVER_HEALTH_CHECK_ENABLED"] = "false"
st.set_page_config(page_title="서울시 감성 지수 대시보드", layout="wide")
sns.set_style("whitegrid")

import matplotlib
if sys.platform.startswith("win"):
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
else:
    matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
st.markdown('<style>*{font-family:"Malgun Gothic",sans-serif!important;}</style>',
            unsafe_allow_html=True)

st.write("🚀 **Streamlit 앱 시작!**")

# ───────────────── 2. Snowflake 연결 ─────────────────
@st.cache_resource(show_spinner=False)
def make_conn():
    try:
        s = st.secrets["snowflake"]
        conn = sf.connect(
            user      = s["user"],
            password  = s["password"],
            account   = s["account"],           # 예: "TGLBRLT-EKB57194"
            warehouse = "COMPUTE_WH",
            ocsp_fail_open = True,
            insecure_mode  = True,
            session_parameters = {"QUERY_TAG":"feel_idx_dashboard"}
        )
        return conn
    except Exception as e:
        st.error(f"❌ Snowflake 연결 실패: {e}")
        st.stop()

def run_query(q:str) -> pd.DataFrame:
    with make_conn() as conn:
        cur = conn.cursor()
        cur.execute(q)
        df  = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
        cur.close()
    return df

# ───────────────── 3. 데이터 전처리 ─────────────────
BASE = "SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA"
Q_FP, Q_CARD, Q_ASSET, Q_SCCO = (
    f"SELECT * FROM {BASE}.FLOATING_POPULATION_INFO",
    f"SELECT * FROM {BASE}.CARD_SALES_INFO",
    f"SELECT * FROM {BASE}.ASSET_INCOME_INFO",
    f"SELECT * FROM {BASE}.M_SCCO_MST",
)

@st.cache_data(show_spinner=True)
def load_and_prep(sample_frac:float=0.01) -> pd.DataFrame:

    # ---- ① 로드 (실패 시 중단) ----
    try:
        fp, card, asset, scco = (run_query(Q_FP),
                                 run_query(Q_CARD),
                                 run_query(Q_ASSET),
                                 run_query(Q_SCCO))
    except Exception as e:
        st.error(f"쿼리 실패: {e}")
        st.stop()

    # ---- ② 1 % 샘플링 ----
    fp = fp.sample(frac=sample_frac, random_state=42)

    # ---- ③ 머지 ----
    fp_card = pd.merge(fp, card, how="inner",
                       on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER",
                           "TIME_SLOT","WEEKDAY_WEEKEND"])
    data = pd.merge(fp_card, asset, how="inner",
                    on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER"])

    # ---- ④ 행정동 한글명 ----
    scco_map = (scco.drop_duplicates("DISTRICT_CODE")
                     .set_index("DISTRICT_CODE")["DISTRICT_KOR_NAME"].to_dict())
    data["DISTRICT_KOR_NAME"] = data["DISTRICT_CODE"].map(scco_map)

    # ---- ⑤ 파생 변수 ----
    data["전체인구"]     = data[["RESIDENTIAL_POPULATION","WORKING_POPULATION",
                                 "VISITING_POPULATION"]].sum(axis=1)
    sale_cols = ["FOOD_SALES","COFFEE_SALES","BEAUTY_SALES",
                 "ENTERTAINMENT_SALES","SPORTS_CULTURE_LEISURE_SALES",
                 "TRAVEL_SALES","CLOTHING_ACCESSORIES_SALES"]
    data["엔터전체매출"] = data[sale_cols].sum(axis=1)
    data["소비활력지수"] = data["엔터전체매출"] / data["전체인구"].replace(0,np.nan)
    data["유입지수"]     = data["VISITING_POPULATION"] / (
                            data["RESIDENTIAL_POPULATION"]+data["WORKING_POPULATION"]
                           ).replace(0,np.nan)
    data["엔터매출비율"] = data["엔터전체매출"] / data["TOTAL_SALES"].replace(0,np.nan)

    cnt_cols = ["FOOD_COUNT","COFFEE_COUNT","BEAUTY_COUNT",
                "ENTERTAINMENT_COUNT","SPORTS_CULTURE_LEISURE_COUNT",
                "TRAVEL_COUNT","CLOTHING_ACCESSORIES_COUNT"]
    data["엔터전체방문자수"] = data[cnt_cols].sum(axis=1)
    data["엔터방문자비율"] = data["엔터전체방문자수"] / data["TOTAL_COUNT"].replace(0,np.nan)
    data["엔터활동밀도"]   = data["엔터전체매출"] / data["전체인구"].replace(0,np.nan)
    data["엔터매출밀도"]   = data["엔터전체매출"] / data["엔터전체방문자수"].replace(0,np.nan)

    # ---- ⑥ FEEL_IDX (PCA-1) ----
    emo_vars = ["엔터전체매출","유입지수","엔터매출비율",
                "엔터전체방문자수","엔터방문자비율","엔터활동밀도","엔터매출밀도"]
    data["FEEL_IDX"] = np.nan
    X = data[emo_vars].apply(pd.to_numeric, errors="coerce").dropna()
    if not X.empty:
        try:
            pc1 = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(X)).ravel()
            pc1_n = (pc1-pc1.min())/(pc1.max()-pc1.min()+1e-9)
            data.loc[X.index,"FEEL_IDX"] = pc1_n.astype(float)
        except Exception as e:
            warnings.warn(f"PCA 실패: {e}")

    return data

df = load_and_prep()

st.title("서울시 인스타 감성 지수 분석")

if df.empty:
    st.error("⚠️ 로드된 데이터가 없습니다.")
    st.stop()

# ───────────────── 4. 필터 ─────────────────
with st.sidebar:
    st.header("필터")
    sel_dist = st.multiselect("행정동", sorted(df["DISTRICT_KOR_NAME"].dropna().unique()))
    sel_age  = st.multiselect("연령대",  sorted(df["AGE_GROUP"].unique()))
    sel_gen  = st.multiselect("성별",     ["M","F"])

mask = (
    (df["DISTRICT_KOR_NAME"].isin(sel_dist) if sel_dist else True) &
    (df["AGE_GROUP"].isin(sel_age)          if sel_age  else True) &
    (df["GENDER"].isin(sel_gen)             if sel_gen  else True)
)
view = df.loc[mask]

if view.empty:
    st.info("🛈 선택 조건에 해당하는 데이터가 없습니다.")
    st.stop()

# ───────────────── 5. 요약 카드 ─────────────────
def safe_mean(series):
    val = series.mean()
    return "N/A" if pd.isna(val) else f"{val:,.3f}"

c1,c2,c3 = st.columns(3)
c1.metric("평균 FEEL_IDX",     safe_mean(view["FEEL_IDX"]))
c2.metric("평균 소비활력지수", safe_mean(view["소비활력지수"]))
c3.metric("평균 유입지수",     safe_mean(view["유입지수"]))

tab1,tab2,tab3 = st.tabs(["지수 상위 지역","성별·연령 분석","산점도"])

# ───────── 탭 1 ─────────
with tab1:
    top20 = (view.groupby("DISTRICT_KOR_NAME")["소비활력지수"]
                  .mean().nlargest(20))
    if top20.empty:
        st.info("데이터가 없습니다.")
    else:
        fig, ax = plt.subplots(figsize=(10,4))
        sns.barplot(x=top20.index, y=top20.values, palette="rocket", ax=ax)
        ax.set_xlabel("행정동"); ax.set_ylabel("소비활력지수")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)

# ───────── 탭 2 ─────────
with tab2:
    grp = (view.groupby(["AGE_GROUP","GENDER"])["TOTAL_SALES"]
                .mean().reset_index())
    if grp.empty:
        st.info("데이터가 없습니다.")
    else:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=grp, x="AGE_GROUP", y="TOTAL_SALES",
                    hue="GENDER", palette={"M":"#3498db","F":"#e75480"}, ax=ax)
        ax.set_ylabel("평균 매출")
        st.pyplot(fig)

# ───────── 탭 3 ─────────
with tab3:
    base = view.dropna(subset=["FEEL_IDX"])
    if base.empty:
        st.info("FEEL_IDX 값이 없습니다.")
    else:
        X_CHOICES = ["엔터전체매출","소비활력지수","유입지수","엔터전체방문자수"]
        Y_CHOICES = ["FEEL_IDX","엔터활동밀도","엔터매출비율"]
        xx = st.selectbox("X 변수", X_CHOICES, key="xx")
        yy = st.selectbox("Y 변수", Y_CHOICES, key="yy")
        fig, ax = plt.subplots(figsize=(6,4))
        try:
            sns.scatterplot(data=base, x=xx, y=yy,
                            hue="FEEL_IDX", palette="viridis", alpha=0.4, ax=ax)
        except ValueError as e:
            st.warning(f"플롯 오류: {e}")
        st.pyplot(fig)

st.divider()
st.caption("데이터 출처 · Snowflake Marketplace")
