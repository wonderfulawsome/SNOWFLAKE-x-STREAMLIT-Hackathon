import streamlit as st
import snowflake.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys

# 페이지 설정
st.set_page_config(page_title="서울시 감성 지수 대시보드", layout="wide")
sns.set_style("whitegrid")

# 한글 폰트 설정
import matplotlib as mpl
if sys.platform.startswith("win"):
    mpl.rcParams["font.family"] = "Malgun Gothic"
else:
    mpl.rcParams["font.family"] = "AppleGothic"
mpl.rcParams["axes.unicode_minus"] = False

# Streamlit sidebar 한글 글꼴
st.markdown(
    """
    <style>
      * { font-family: "Malgun Gothic", sans-serif !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Snowflake 연결
def get_conn():
    creds = st.secrets["snowflake"]
    return snowflake.connector.connect(
        user=creds["user"],
        password=creds["password"],
        account=creds["account"],
        warehouse=creds["warehouse"],
        ocsp_fail_open=True,
        insecure_mode=True
    )

@st.cache_data(show_spinner=True)
def load_query(q: str) -> pd.DataFrame:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(q)
    df = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
    cur.close()
    conn.close()
    return df

# SQL 쿼리
Q_FP    = "SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.FLOATING_POPULATION_INFO"
Q_CARD  = "SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.CARD_SALES_INFO"
Q_ASSET = "SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.ASSET_INCOME_INFO"
Q_SCCO  = "SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.M_SCCO_MST"

@st.cache_data(show_spinner=True)
def load_all():
    fp    = load_query(Q_FP).sample(frac=0.01, random_state=42)
    card  = load_query(Q_CARD).sample(frac=0.01, random_state=42)
    asset = load_query(Q_ASSET).sample(frac=0.01, random_state=42)
    scco  = load_query(Q_SCCO).sample(frac=0.01, random_state=42)
    return fp, card, asset, scco

def preprocess() -> pd.DataFrame:
    fp, card, asset, scco = load_all()

    # 병합
    df = pd.merge(fp, card,
                  on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER","TIME_SLOT","WEEKDAY_WEEKEND"],
                  how="inner", validate="m:m")
    df = pd.merge(df, asset,
                  on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER"],
                  how="inner", validate="m:m")

    # 중복 컬럼 제거
    df = df.drop(columns=[c for c in df.columns if c.endswith(("_x","_y"))])

    # 행정동 이름 매핑
    name_map = (scco[["DISTRICT_CODE","DISTRICT_KOR_NAME"]]
                .drop_duplicates("DISTRICT_CODE")
                .set_index("DISTRICT_CODE")["DISTRICT_KOR_NAME"]
                .to_dict())
    df["DISTRICT_KOR_NAME"] = df["DISTRICT_CODE"].map(name_map)

    # 파생 변수 생성
    df["전체인구"]      = df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"] + df["VISITING_POPULATION"]
    df["엔터전체매출"]  = df[[
        "FOOD_SALES","COFFEE_SALES","BEAUTY_SALES",
        "ENTERTAINMENT_SALES","SPORTS_CULTURE_LEISURE_SALES",
        "TRAVEL_SALES","CLOTHING_ACCESSORIES_SALES"
    ]].sum(axis=1)
    df["소비활력지수"]  = df["엔터전체매출"] / df["전체인구"].replace(0, np.nan)
    df["유입지수"]      = df["VISITING_POPULATION"] / (
                          df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"]
                          ).replace(0, np.nan)
    df["엔터매출비율"]  = df["엔터전체매출"] / df["TOTAL_SALES"].replace(0, np.nan)

    cnt_cols = [
        "FOOD_COUNT","COFFEE_COUNT","BEAUTY_COUNT","ENTERTAINMENT_COUNT",
        "SPORTS_CULTURE_LEISURE_COUNT","TRAVEL_COUNT","CLOTHING_ACCESSORIES_COUNT"
    ]
    df["엔터전체방문자수"] = df[cnt_cols].sum(axis=1)
    df["엔터방문자비율"]   = df["엔터전체방문자수"] / df["TOTAL_COUNT"].replace(0, np.nan)
    df["엔터활동밀도"]     = df["엔터전체매출"] / df["전체인구"].replace(0, np.nan)
    df["엔터매출밀도"]     = df["엔터전체매출"] / df["엔터전체방문자수"].replace(0, np.nan)

    # FEEL_IDX 계산
    emo_vars = [
        "엔터전체매출","소비활력지수","유입지수","엔터매출비율",
        "엔터전체방문자수","엔터방문자비율","엔터활동밀도","엔터매출밀도"
    ]
    X = df[emo_vars].dropna()
    pc1 = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(X))
    pc1_n = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-9)
    df.loc[X.index, "FEEL_IDX"] = pc1_n.flatten()

    return df

@st.cache_data(show_spinner=True)
def get_data():
    return preprocess()

# 앱 UI
st.title("서울시 인스타 감성 지수 분석")
data = get_data()

with st.sidebar:
    districts = st.multiselect("행정동", sorted(data["DISTRICT_KOR_NAME"].dropna().unique()), [])
    ages      = st.multiselect("연령대", sorted(data["AGE_GROUP"].unique()), [])
    genders   = st.multiselect("성별", ["M","F"], [])

mask = (
    (data["DISTRICT_KOR_NAME"].isin(districts) if districts else True) &
    (data["AGE_GROUP"].isin(ages)          if ages      else True) &
    (data["GENDER"].isin(genders)         if genders   else True)
)
view = data.loc[mask]

# 요약 지표
st.subheader("요약 지표")
c1, c2, c3 = st.columns(3)
c1.metric("평균 FEEL_IDX",    f"{view['FEEL_IDX'].mean():.2f}")
c2.metric("평균 소비활력지수", f"{view['소비활력지수'].mean():.2f}")
c3.metric("평균 유입지수",    f"{view['유입지수'].mean():.2f}")

tab1, tab2, tab3 = st.tabs(["지수 상위 지역","성별·연령 분석","산점도"])
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
    x = st.selectbox("X축 변수", ["엔터전체매출","소비활력지수","유입지수","엔터전체방문자수"])
    y = st.selectbox("Y축 변수", ["FEEL_IDX","엔터활동밀도","엔터매출비율"])
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=view, x=x, y=y, hue="FEEL_IDX", palette="viridis", alpha=0.6, ax=ax)
    st.pyplot(fig)

st.divider()
st.caption("데이터 출처 · Snowflake Marketplace")
