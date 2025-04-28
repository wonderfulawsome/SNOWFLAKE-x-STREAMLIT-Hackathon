import streamlit as st
import snowflake.connector as sf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 스트림릿 페이지 설정
st.set_page_config(page_title="서울시 인스타 감성 지수 대시보드", layout="wide")
sns.set_style("whitegrid")

# 한글 폰트 설정 (Windows / Mac 구분)
import matplotlib
if sys.platform.startswith("win"):
    matplotlib.rcParams["font.family"] = "Malgun Gothic"
else:
    matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# Snowflake 연결 함수
@st.cache_resource(show_spinner=False)
def get_conn():
    creds = st.secrets["snowflake"]
    return sf.connect(
        user=creds["user"],
        password=creds["password"],
        account=creds["account"],
        warehouse="COMPUTE_WH",
        ocsp_fail_open=True,
        insecure_mode=True
    )

# 쿼리 실행 헬퍼
def run_query(query: str) -> pd.DataFrame:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(query)
    df = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
    cur.close()
    return df

# 데이터 불러오기 및 병합
@st.cache_data(show_spinner=True)
def load_data(sample_frac: float = 0.02) -> pd.DataFrame:
    base = "SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA"
    fp    = run_query(f"SELECT * FROM {base}.FLOATING_POPULATION_INFO")
    card  = run_query(f"SELECT * FROM {base}.CARD_SALES_INFO")
    asset = run_query(f"SELECT * FROM {base}.ASSET_INCOME_INFO")
    scco  = run_query(f"SELECT * FROM {base}.M_SCCO_MST")

    # 샘플링
    fp = fp.sample(frac=sample_frac, random_state=42)

    # 병합
    df = pd.merge(fp, card, on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER","TIME_SLOT","WEEKDAY_WEEKEND"], how="inner")
    df = pd.merge(df, asset, on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER"], how="inner")
    name_map = scco.set_index("DISTRICT_CODE")["DISTRICT_KOR_NAME"].to_dict()
    df["DISTRICT_KOR_NAME"] = df["DISTRICT_CODE"].map(name_map)

    # 파생 변수
    df["전체인구"]     = df[["RESIDENTIAL_POPULATION","WORKING_POPULATION","VISITING_POPULATION"]].sum(axis=1)
    sale_cols = ["FOOD_SALES","COFFEE_SALES","BEAUTY_SALES","ENTERTAINMENT_SALES",
                 "SPORTS_CULTURE_LEISURE_SALES","TRAVEL_SALES","CLOTHING_ACCESSORIES_SALES"]
    df["엔터전체매출"] = df[sale_cols].sum(axis=1)
    df["소비활력지수"] = df["엔터전체매출"] / df["전체인구"].replace(0, np.nan)
    df["유입지수"]     = df["VISITING_POPULATION"] / (df["RESIDENTIAL_POPULATION"]+df["WORKING_POPULATION"]).replace(0, np.nan)
    df["엔터매출비율"] = df["엔터전체매출"] / df["TOTAL_SALES"].replace(0, np.nan)
    cnt_cols = ["FOOD_COUNT","COFFEE_COUNT","BEAUTY_COUNT","ENTERTAINMENT_COUNT",
                "SPORTS_CULTURE_LEISURE_COUNT","TRAVEL_COUNT","CLOTHING_ACCESSORIES_COUNT"]
    df["엔터전체방문자수"] = df[cnt_cols].sum(axis=1)
    df["엔터활동밀도"]   = df["엔터전체매출"] / df["전체인구"].replace(0, np.nan)
    df["엔터매출밀도"]   = df["엔터전체매출"] / df["엔터전체방문자수"].replace(0, np.nan)

    # PCA로 FEEL_IDX 계산
    emo_vars = ["엔터전체매출","유입지수","엔터매출비율","엔터전체방문자수","엔터활동밀도","엔터매출밀도"]
    X = df[emo_vars].dropna()
    df["FEEL_IDX"] = np.nan
    if not X.empty:
        try:
            pc1 = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(X)).ravel()
            pc1n = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-9)
            df.loc[X.index, "FEEL_IDX"] = pc1n
        except Exception as e:
            warnings.warn(f"PCA 오류: {e}")
    return df

data = load_data()

# 사이드바 필터
st.sidebar.header("필터")
sel_dist = st.sidebar.multiselect("행정동", sorted(data["DISTRICT_KOR_NAME"].dropna().unique()))
sel_age  = st.sidebar.multiselect("연령대", sorted(data["AGE_GROUP"].unique()))
sel_gen  = st.sidebar.multiselect("성별", ["M","F"])

mask = (
    (data["DISTRICT_KOR_NAME"].isin(sel_dist) if sel_dist else True) &
    (data["AGE_GROUP"].isin(sel_age)          if sel_age  else True) &
    (data["GENDER"].isin(sel_gen)             if sel_gen  else True)
)
view = data[mask]

# 메인
st.title("서울시 인스타 감성 지수 대시보드")

if view.empty:
    st.warning("조건에 맞는 데이터가 없습니다.")
    st.stop()

# 지표 카드
col1, col2, col3 = st.columns(3)
col1.metric("평균 FEEL_IDX", f"{view['FEEL_IDX'].mean():.3f}")
col2.metric("평균 소비활력지수", f"{view['소비활력지수'].mean():.3f}")
col3.metric("평균 유입지수", f"{view['유입지수'].mean():.3f}")

# 탭별 시각화
tab1, tab2 = st.tabs(["지수 상위 지역", "상관관계 히트맵"])

with tab1:
    top = view.groupby("DISTRICT_KOR_NAME")["FEEL_IDX"].mean().nlargest(10)
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x=top.values, y=top.index, palette="rocket", ax=ax)
    ax.set_xlabel("FEEL_IDX")
    ax.set_ylabel("")
    st.pyplot(fig)

with tab2:
    corr_vars = ["FEEL_IDX","소비활력지수","유입지수","엔터매출비율","엔터활동밀도","엔터매출밀도"]
    corr_df = view[corr_vars].dropna()
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)
