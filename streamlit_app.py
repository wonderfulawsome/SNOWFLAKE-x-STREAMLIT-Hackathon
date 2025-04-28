###############################################################################
# streamlit_app.py ― 서울시 인스타 감성 지수 대시보드
###############################################################################
import os, sys, warnings
import streamlit as st
import snowflake.connector as sf
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ────────────────────────── 기본 환경 세팅 ──────────────────────────
os.environ["STREAMLIT_SERVER_HEALTH_CHECK_ENABLED"] = "false"   # Cloud health-check 에러 방지
st.set_page_config(page_title="서울시 감성 지수 대시보드", layout="wide")
sns.set_style("whitegrid")

# ① 한글 폰트 (윈도우: Malgun Gothic, mac/Linux: AppleGothic fallback)
import matplotlib
if sys.platform.startswith("win"):
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
else:
    matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
st.markdown('<style>*{font-family:"Malgun Gothic",sans-serif!important;}</style>',
            unsafe_allow_html=True)

st.write("🚀 **Streamlit 앱 시작!**")

# ──────────────────────────  Snowflake 연결  ─────────────────────────
def get_conn():
    s = st.secrets["snowflake"]             #  secrets.toml  ──>
    # [snowflake]
    # user = "..."
    # password = "..."
    # account = "TGLBRLT-EKB57194"   ← Account Identifier
    # (필요하면 role / database / schema … 추가)
    return sf.connect(
        user      = s["user"],
        password  = s["password"],
        account   = s["account"],
        warehouse = "COMPUTE_WH",
        ocsp_fail_open = True,
        insecure_mode  = True
    )

@st.cache_data(show_spinner=False)
def load(q:str) -> pd.DataFrame:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(q)
        df  = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
        cur.close()
    return df

BASE = "SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA"
Q_FP    = f"SELECT * FROM {BASE}.FLOATING_POPULATION_INFO"      # 6 백만+
Q_CARD  = f"SELECT * FROM {BASE}.CARD_SALES_INFO"
Q_ASSET = f"SELECT * FROM {BASE}.ASSET_INCOME_INFO"
Q_SCCO  = f"SELECT * FROM {BASE}.M_SCCO_MST"

# ──────────────────────────  전처리  ─────────────────────────
@st.cache_data(show_spinner=True)
def preprocess() -> pd.DataFrame:

    # ① 데이터 로드
    fp, card, asset, scco = load(Q_FP), load(Q_CARD), load(Q_ASSET), load(Q_SCCO)

    # ② 유동-인구 테이블만 1 % 샘플링해 속도 확보
    fp = fp.sample(frac=0.01, random_state=42)

    # ③ 조인 (INNER → 데이터 정합)
    fp_card = pd.merge(
        fp, card, how="inner",
        on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER","TIME_SLOT","WEEKDAY_WEEKEND"]
    )
    data = pd.merge(
        fp_card, asset, how="inner",
        on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER"]
    )

    # ④ 행정동 한글명 붙이기
    scco_map = (
        scco.drop_duplicates("DISTRICT_CODE")
            .set_index("DISTRICT_CODE")["DISTRICT_KOR_NAME"].to_dict()
    )
    data["DISTRICT_KOR_NAME"] = data["DISTRICT_CODE"].map(scco_map)

    # ⑤ 파생 변수
    data["전체인구"]       = data[["RESIDENTIAL_POPULATION","WORKING_POPULATION","VISITING_POPULATION"]].sum(axis=1)

    sales_cols = ["FOOD_SALES","COFFEE_SALES","BEAUTY_SALES",
                  "ENTERTAINMENT_SALES","SPORTS_CULTURE_LEISURE_SALES",
                  "TRAVEL_SALES","CLOTHING_ACCESSORIES_SALES"]
    data["엔터전체매출"]   = data[sales_cols].sum(axis=1)

    data["소비활력지수"]   = data["엔터전체매출"] / data["전체인구"].replace(0,np.nan)
    data["유입지수"]       = data["VISITING_POPULATION"] / (
                              data["RESIDENTIAL_POPULATION"] + data["WORKING_POPULATION"]).replace(0,np.nan)
    data["엔터매출비율"]   = data["엔터전체매출"] / data["TOTAL_SALES"].replace(0,np.nan)

    cnt_cols = ["FOOD_COUNT","COFFEE_COUNT","BEAUTY_COUNT",
                "ENTERTAINMENT_COUNT","SPORTS_CULTURE_LEISURE_COUNT",
                "TRAVEL_COUNT","CLOTHING_ACCESSORIES_COUNT"]
    data["엔터전체방문자수"] = data[cnt_cols].sum(axis=1)
    data["엔터방문자비율"] = data["엔터전체방문자수"] / data["TOTAL_COUNT"].replace(0,np.nan)
    data["엔터활동밀도"]   = data["엔터전체매출"] / data["전체인구"].replace(0,np.nan)
    data["엔터매출밀도"]   = data["엔터전체매출"] / data["엔터전체방문자수"].replace(0,np.nan)

    # ⑥ FEEL_IDX (PCA 1-주성분) — 배열 → float 로 평탄화
    emo_vars = ["엔터전체매출","유입지수","엔터매출비율",
                "엔터전체방문자수","엔터방문자비율","엔터활동밀도","엔터매출밀도"]
    data["FEEL_IDX"] = np.nan
    X = data[emo_vars].apply(pd.to_numeric, errors="coerce").dropna()
    if not X.empty:
        X_std = StandardScaler().fit_transform(X)
        pc1   = PCA(n_components=1).fit_transform(X_std).ravel()      # ← ravel로 1-D
        pc1_n = (pc1 - pc1.min())/(pc1.max() - pc1.min() + 1e-9)
        data.loc[X.index, "FEEL_IDX"] = pc1_n.astype(float)           # ← float 컬럼

    return data

df_base = preprocess()
st.title("서울시 인스타 감성 지수 분석")

if df_base.empty:
    st.error("❌ 데이터가 없습니다. (샘플링 비율을 높여 보세요)")
    st.stop()

# ──────────────────────────  사이드바 필터  ─────────────────────────
with st.sidebar:
    st.header("필터")
    sel_dist = st.multiselect("행정동", sorted(df_base["DISTRICT_KOR_NAME"].dropna().unique()))
    sel_age  = st.multiselect("연령대",  sorted(df_base["AGE_GROUP"].unique()))
    sel_gen  = st.multiselect("성별",     ["M","F"])

mask = (
    (df_base["DISTRICT_KOR_NAME"].isin(sel_dist) if sel_dist else True) &
    (df_base["AGE_GROUP"].isin(sel_age)           if sel_age  else True) &
    (df_base["GENDER"].isin(sel_gen)              if sel_gen  else True)
)
view = df_base.loc[mask]

# ──────────────────────────  요약 지표  ─────────────────────────
if view.empty:
    st.warning("◻️ 선택된 조건에 해당하는 행이 없습니다.")
    st.stop()

c1,c2,c3 = st.columns(3)
c1.metric("평균 FEEL_IDX",     f"{view['FEEL_IDX'].mean():.3f}")
c2.metric("평균 소비활력지수", f"{view['소비활력지수'].mean():.3f}")
c3.metric("평균 유입지수",     f"{view['유입지수'].mean():.3f}")

tab1,tab2,tab3 = st.tabs(["지수 상위 지역","성별·연령 분석","산점도"])

# ──────────────────────────  탭 1 ─────────────────────────
with tab1:
    top = (view.groupby("DISTRICT_KOR_NAME")["소비활력지수"]
                 .mean().sort_values(ascending=False).head(20))
    if top.empty:
        st.info("데이터가 없습니다.")
    else:
        fig, ax = plt.subplots(figsize=(10,4))
        sns.barplot(x=top.index, y=top.values, palette="rocket", ax=ax)
        ax.set_xlabel("행정동"); ax.set_ylabel("소비활력지수")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig)

# ──────────────────────────  탭 2 ─────────────────────────
with tab2:
    agg = (view.groupby(["AGE_GROUP","GENDER"])["TOTAL_SALES"]
                 .mean().reset_index())
    if agg.empty:
        st.info("데이터가 없습니다.")
    else:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=agg, x="AGE_GROUP", y="TOTAL_SALES",
                    hue="GENDER", palette={"M":"#3498db","F":"#e75480"}, ax=ax)
        ax.set_ylabel("1일 평균 매출")
        st.pyplot(fig)

# ──────────────────────────  탭 3 ─────────────────────────
with tab3:
    base = view.dropna(subset=["FEEL_IDX"])
    if base.empty:
        st.info("FEEL_IDX 가 없는 행입니다.")
    else:
        xx = st.selectbox("X 변수", ["엔터전체매출","소비활력지수","유입지수","엔터전체방문자수"])
        yy = st.selectbox("Y 변수", ["FEEL_IDX","엔터활동밀도","엔터매출비율"])
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(data=base, x=xx, y=yy, hue="FEEL_IDX",
                        palette="viridis", alpha=0.5, ax=ax)
        st.pyplot(fig)

st.divider()
st.caption("데이터 출처 · Snowflake Marketplace")
