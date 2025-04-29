import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st
import snowflake.connector
import matplotlib.font_manager as fm

# ────────────────── 기본 설정 ──────────────────
st.set_page_config(page_title="서울시 감성 지수 대시보드", layout="wide")
sns.set_style("whitegrid")

# 한글 폰트 설정 (Windows 기준)
font_path = "C:/Windows/Fonts/malgun.ttf"
if os.path.exists(font_path):
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False

# ────────────────── Snowflake 연결 ──────────────────

def get_conn():
    return snowflake.connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"],
        warehouse=st.secrets["snowflake"]["warehouse"],
        ocsp_fail_open=True,
        insecure_mode=True,
    )

@st.cache_data(show_spinner=True)
def load_df(query: str) -> pd.DataFrame:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(query)
    df = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
    cur.close(); conn.close()
    return df

# ────────────────── 쿼리 문자열 ──────────────────
Q = {
    "fp":    "SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.FLOATING_POPULATION_INFO",
    "card":  "SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.CARD_SALES_INFO",
    "asset": "SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.ASSET_INCOME_INFO",
    "scco":  "SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.M_SCCO_MST",
}

with st.spinner("데이터 로딩 중..."):
    fp_df    = load_df(Q["fp"])
    card_df  = load_df(Q["card"])
    asset_df = load_df(Q["asset"])
    scco_df  = load_df(Q["scco"])

# ────────────────── 데이터 전처리 ──────────────────

def preprocess():
    df = (
        fp_df
        .merge(card_df,  on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER","TIME_SLOT","WEEKDAY_WEEKEND"], how="inner")
        .merge(asset_df, on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER"], how="inner")
        .merge(scco_df,  on="DISTRICT_CODE", how="inner")
    )

    dup_cols = [c for c in df.columns if c.endswith("_y")]
    df = df.drop(columns=dup_cols)
    df = df.rename(columns={c: c.replace("_x", "") for c in df.columns})

    df["전체인구"] = df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"] + df["VISITING_POPULATION"]

    sales_cols = [
        "FOOD_SALES", "COFFEE_SALES", "BEAUTY_SALES", "ENTERTAINMENT_SALES",
        "SPORTS_CULTURE_LEISURE_SALES", "TRAVEL_SALES", "CLOTHING_ACCESSORIES_SALES",
    ]
    count_cols = [
        "FOOD_COUNT", "COFFEE_COUNT", "BEAUTY_COUNT", "ENTERTAINMENT_COUNT",
        "SPORTS_CULTURE_LEISURE_COUNT", "TRAVEL_COUNT", "CLOTHING_ACCESSORIES_COUNT",
    ]

    df["엔터전체매출"] = df[sales_cols].sum(axis=1)
    df["엔터전체방문자수"] = df[count_cols].sum(axis=1)

    df["엔터매출비율"]   = df["엔터전체매출"] / df["TOTAL_SALES"].replace(0, np.nan)
    df["엔터방문자비율"] = df["엔터전체방문자수"] / df["TOTAL_COUNT"].replace(0, np.nan)
    df["엔터활동밀도"]   = df["엔터전체매출"] / df["전체인구"].replace(0, np.nan)
    df["엔터매출밀도"]   = df["엔터전체매출"] / df["엔터전체방문자수"].replace(0, np.nan)
    df["유입지수"]       = df["VISITING_POPULATION"] / (df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"]).replace(0, np.nan)

    emo_vars = [
        "엔터전체매출", "유입지수", "엔터매출비율", "엔터전체방문자수",
        "엔터방문자비율", "엔터활동밀도", "엔터매출밀도",
    ]
    X = df[emo_vars].dropna()
    if not X.empty:
        pc1 = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(X))
        df.loc[X.index, "FEEL_IDX"] = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-9)

    return df

@st.cache_data(show_spinner=True)
def get_data():
    return preprocess()

fp_card_scco_asset = get_data()

# ────────────────── 시각화 함수 ──────────────────
PALETTE = "rocket"

def bar_chart(df: pd.DataFrame, x: str, y: str, title: str):
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=df, x=x, y=y, palette=PALETTE, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y)
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# ────────────────── 사이드바 ──────────────────
st.sidebar.header("메뉴")
page = st.sidebar.selectbox(
    "페이지",
    ("방문자수 TOP", "방문자 1인당 매출", "유입지수 TOP", "성별·연령 소비", "상관관계")
)

# ────────────────── 페이지별 뷰 ──────────────────
if page == "방문자수 TOP":
    agg = fp_card_scco_asset.groupby("DISTRICT_KOR_NAME")["엔터전체방문자수"].sum().nlargest(30).reset_index()
    bar_chart(agg, "DISTRICT_KOR_NAME", "엔터전체방문자수", "방문자수 상위 30개 행정동")

elif page == "방문자 1인당 매출":
    temp = fp_card_scco_asset.groupby("DISTRICT_KOR_NAME").agg({"엔터전체매출": "sum", "엔터전체방문자수": "sum"})
    temp["방문자1인당엔터매출"] = temp["엔터전체매출"] / temp["엔터전체방문자수"].replace(0, np.nan)
    agg = temp.nlargest(30, "방문자1인당엔터매출").reset_index()
    bar_chart(agg, "DISTRICT_KOR_NAME", "방문자1인당엔터매출", "방문자 1인당 엔터매출 상위 30개")

elif page == "유입지수 TOP":
    agg = fp_card_scco_asset.groupby("DISTRICT_KOR_NAME")["유입지수"].mean().nlargest(30).reset_index()
    bar_chart(agg, "DISTRICT_KOR_NAME", "유입지수", "유입지수 상위 30개 행정동")

elif page == "성별·연령 소비":
    gender_palette = {"M": "#3498db", "F": "#e75480"}
    view = st.radio("보기", ("1인당 매출", "엔터 방문횟수", "방문 1회당 매출"), key="view_selector")

    if view == "1인당 매출":
        df_plot = fp_card_scco_asset.groupby(["AGE_GROUP", "GENDER"])["TOTAL_SALES"].mean().reset_index()
        ycol, title = "TOTAL_SALES", "성별·연령대별 1인당 매출"
    elif view == "엔터 방문횟수":
        df_plot = fp_card_scco_asset.groupby(["AGE_GROUP", "GENDER"])["엔터전체방문자수"].sum().reset_index()
        ycol, title = "엔터전체방문자수", "성별·연령대별 엔터 방문횟수"
    else:
        df_plot = fp_card_scco_asset.groupby(["AGE_GROUP", "GENDER"])["엔터매출밀도"].mean().reset_index()
        ycol, title = "엔터매출밀도", "성별·연령대별 방문 1회당 매출"

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_plot, x="AGE_GROUP", y=ycol, hue="GENDER", palette=gender_palette, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("AGE_GROUP")
    ax.set_ylabel(ycol)
    st.pyplot(fig)

elif page == "상관관계":
    corr_cols = [
        "엔터전체매출", "유입지수", "엔터매출비율", "엔터전체방문자수",
        "엔터방문자비율", "엔터활동밀도", "엔터매출밀도", "FEEL_IDX"
    ]
    corr_df = fp_card_scco_asset[corr_cols].dropna()

    if corr_df.shape[0] < 2:
        st.warning("상관관계를 계산하기에 데이터가 부족합니다.")
    else:
        corr = corr_df.corr()
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title("지표 간 상관관계")
        st.pyplot(fig)

        feel_corr = corr["FEEL_IDX"].drop("FEEL_IDX").sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        sns.barplot(y=feel_corr.index, x=feel_corr.values, palette=PALETTE, ax=ax2)
        ax2.set_xlim(-1, 1)
        ax2.set_title("FEEL_IDX와 지표 상관계수")
        st.pyplot(fig2)

st.success("대시보드 로딩 완료!")
