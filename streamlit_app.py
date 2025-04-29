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

st.set_page_config(page_title="서울시 감성 지수 대시보드", layout="wide")
sns.set_style("whitegrid")

# 한글 폰트
font_path = "C:/Windows/Fonts/malgun.ttf"
if os.path.exists(font_path):
    plt.rc('font', family=fm.FontProperties(fname=font_path).get_name())
    plt.rcParams['axes.unicode_minus'] = False

# Snowflake

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
def load_df(q: str) -> pd.DataFrame:
    conn = get_conn()
    cur = conn.cursor(); cur.execute(q)
    df = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
    cur.close(); conn.close()
    return df.sample(frac=0.001, random_state=42)  # 0.1% 샘플

Q = {
    "fp":    "SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.FLOATING_POPULATION_INFO",
    "card":  "SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.CARD_SALES_INFO",
    "asset": "SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.ASSET_INCOME_INFO",
    "scco":  "SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.M_SCCO_MST",
}

with st.spinner("데이터 로딩 중…"):
    fp_df    = load_df(Q["fp"])
    card_df  = load_df(Q["card"])
    asset_df = load_df(Q["asset"])
    scco_df  = load_df(Q["scco"]).sample(frac=0.01, random_state=42)  # 경계 데이터도 1% 샘플

# 전처리

def preprocess():
    df = (
        fp_df.merge(card_df, on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER","TIME_SLOT","WEEKDAY_WEEKEND"], how="inner")
             .merge(asset_df, on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER"], how="inner")
             .merge(scco_df, on="DISTRICT_CODE", how="inner")
    )

    dup = [c for c in df.columns if c.endswith("_y")]
    df = df.drop(columns=dup).rename(columns={c: c.replace("_x", "") for c in df.columns})

    df["전체인구"] = df[["RESIDENTIAL_POPULATION","WORKING_POPULATION","VISITING_POPULATION"]].sum(axis=1)

    sales_cols = [
        "FOOD_SALES","COFFEE_SALES","BEAUTY_SALES","ENTERTAINMENT_SALES",
        "SPORTS_CULTURE_LEISURE_SALES","TRAVEL_SALES","CLOTHING_ACCESSORIES_SALES",
    ]
    count_cols = [
        "FOOD_COUNT","COFFEE_COUNT","BEAUTY_COUNT","ENTERTAINMENT_COUNT",
        "SPORTS_CULTURE_LEISURE_COUNT","TRAVEL_COUNT","CLOTHING_ACCESSORIES_COUNT",
    ]

    df["엔터전체매출"] = df[sales_cols].sum(axis=1)
    df["엔터전체방문자수"] = df[count_cols].sum(axis=1)

    df["엔터매출비율"] = df["엔터전체매출"] / df["TOTAL_SALES"].replace(0, np.nan)
    df["엔터방문자비율"] = df["엔터전체방문자수"] / df["TOTAL_COUNT"].replace(0, np.nan)
    df["엔터활동밀도"] = df["엔터전체매출"] / df["전체인구"].replace(0, np.nan)
    df["엔터매출밀도"] = df["엔터전체매출"] / df["엔터전체방문자수"].replace(0, np.nan)
    df["유입지수"] = df["VISITING_POPULATION"] / (df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"]).replace(0, np.nan)

    emo = ["엔터전체매출","유입지수","엔터매출비율","엔터전체방문자수","엔터방문자비율","엔터활동밀도","엔터매출밀도"]
    X = df[emo].dropna()
    if not X.empty:
        pc1 = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(X))
        df.loc[X.index, "FEEL_IDX"] = (pc1 - pc1.min())/(pc1.max()-pc1.min()+1e-9)

    return df

@st.cache_data(show_spinner=True)
def get_data():
    return preprocess()

df_all = get_data()

# 공통 바 차트
PALETTE = "rocket"

def bar(df, x, y, ttl):
    fig, ax = plt.subplots(figsize=(14,7))
    sns.barplot(data=df, x=x, y=y, palette=PALETTE, ax=ax)
    ax.set_title(ttl); ax.set_xlabel(""); ax.set_ylabel(y)
    plt.xticks(rotation=45, ha="right"); st.pyplot(fig)

# 사이드바
st.sidebar.header("메뉴")
page = st.sidebar.selectbox("페이지", ("방문자수 TOP","방문자 1인당 매출","유입지수 TOP","성별·연령 소비","상관관계"))

a = df_all  # alias

if page == "방문자수 TOP":
    g = a.groupby("DISTRICT_KOR_NAME")["엔터전체방문자수"].sum().nlargest(30).reset_index()
    bar(g, "DISTRICT_KOR_NAME","엔터전체방문자수","방문자수 상위 30개")
elif page == "방문자 1인당 매출":
    t = a.groupby("DISTRICT_KOR_NAME").agg({"엔터전체매출":"sum","엔터전체방문자수":"sum"})
    t["방문자1인당엔터매출"] = t["엔터전체매출"] / t["엔터전체방문자수"].replace(0, np.nan)
    g = t.nlargest(30,"방문자1인당엔터매출").reset_index()
    bar(g,"DISTRICT_KOR_NAME","방문자1인당엔터매출","방문자 1인당 엔터매출")
elif page == "유입지수 TOP":
    g = a.groupby("DISTRICT_KOR_NAME")["유입지수"].mean().nlargest(30).reset_index()
    bar(g,"DISTRICT_KOR_NAME","유입지수","유입지수 상위 30개")
elif page == "성별·연령 소비":
    gender_pal = {"M":"#3498db","F":"#e75480"}
    view = st.radio("보기", ("1인당 매출","엔터 방문","방문 1회당 매출"))
    if view == "1인당 매출":
        g = a.groupby(["AGE_GROUP","GENDER"])["TOTAL_SALES"].mean().reset_index()
        y, ttl = "TOTAL_SALES","성별·연령 1인당 매출"
    elif view == "엔터 방문":
        g = a.groupby(["AGE_GROUP","GENDER"])["엔터전체방문자수"].sum().reset_index()
        y, ttl = "엔터전체방문자수","성별·연령 엔터 방문"
    else:
        g = a.groupby(["AGE_GROUP","GENDER"])["엔터매출밀도"].mean().reset_index()
        y, ttl = "엔터매출밀도","성별·연령 방문 1회당 매출"
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(data=g, x="AGE_GROUP", y=y, hue="GENDER", palette=gender_pal, ax=ax)
    ax.set_title(ttl); ax.set_xlabel("AGE"); ax.set_ylabel(y); st.pyplot(fig)
else:
    cols = ["엔터전체매출","유입지수","엔터매출비율","엔터전체방문자수","엔터방문자비율","엔터활동밀도","엔터매출밀도","FEEL_IDX"]
    cdf = a[cols].dropna();
    if cdf.shape[0] < 2:
        st.warning("데이터 샘플이 너무 작아요")
    else:
        corr = cdf.corr()
        fig, ax = plt.subplots(figsize=(8,7))
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title("지표 상관관계"); st.pyplot(fig)
        fc = corr["FEEL_IDX"].drop("FEEL_IDX").sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(7,5))
        sns.barplot(y=fc.index, x=fc.values, palette=PALETTE, ax=ax2); ax2.set_xlim(-1,1)
        ax2.set_title("FEEL_IDX 상관계수"); st.pyplot(fig2)

st.success("완료");
