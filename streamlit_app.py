import os
import streamlit as st
import snowflake.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# health-check 타임아웃 끄기
os.environ["STREAMLIT_SERVER_HEALTH_CHECK_ENABLED"] = "false"

# 기본 설정
st.set_page_config(page_title="서울시 감성 지수 대시보드", layout="wide")
sns.set_style("whitegrid")

# 한글 폰트
import matplotlib
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

st.markdown(
    '<style>*{font-family:"Malgun Gothic",sans-serif!important;}</style>',
    unsafe_allow_html=True,
)
st.write("🚀 Streamlit 앱 시작!")

# ── Snowflake 연결 ────────────────────────────
def get_conn():
    return snowflake.connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"],
        warehouse="COMPUTE_WH",
        ocsp_fail_open=True,
        insecure_mode=True,
    )

@st.cache_data(show_spinner=False)
def load_query(q: str) -> pd.DataFrame:
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(q)
    df = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
    cur.close(); conn.close()
    return df

BASE = "SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA"
Q_FP   = f"SELECT * FROM {BASE}.FLOATING_POPULATION_INFO LIMIT 8000"
Q_CARD = f"SELECT * FROM {BASE}.CARD_SALES_INFO           LIMIT 8000"
Q_ASSET= f"SELECT * FROM {BASE}.ASSET_INCOME_INFO         LIMIT 8000"
Q_SCCO = f"SELECT * FROM {BASE}.M_SCCO_MST"

@st.cache_data(show_spinner=True)
def load_all():
    return (
        load_query(Q_FP),
        load_query(Q_CARD),
        load_query(Q_ASSET),
        load_query(Q_SCCO),
    )

# ── 전처리 ─────────────────────────────────────
def preprocess() -> pd.DataFrame:
    fp, card, asset, scco = load_all()

    fp_card = pd.merge(
        fp, card,
        on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER",
            "TIME_SLOT","WEEKDAY_WEEKEND"],
        how="inner", validate="m:m"
    )
    fp_card_asset = pd.merge(
        fp_card, asset,
        on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER"],
        how="inner", validate="m:m"
    )
    dup_cols = [c for c in fp_card_asset.columns if c.endswith(("_x","_y"))]
    fp_card_asset = fp_card_asset.drop(columns=dup_cols)

    scco_map = (
        scco[["DISTRICT_CODE","DISTRICT_KOR_NAME"]]
        .drop_duplicates("DISTRICT_CODE")
        .set_index("DISTRICT_CODE")["DISTRICT_KOR_NAME"]
        .to_dict()
    )
    fp_card_asset["DISTRICT_KOR_NAME"] = fp_card_asset["DISTRICT_CODE"].map(scco_map)

    # ⬇ 필요한 만큼만 샘플링 (교집합 후)
    data = fp_card_asset.sample(frac=0.02, random_state=42)

    # 파생 변수
    data["전체인구"] = (
        data["RESIDENTIAL_POPULATION"] + data["WORKING_POPULATION"] + data["VISITING_POPULATION"]
    )
    data["엔터전체매출"] = (
        data["FOOD_SALES"] + data["COFFEE_SALES"] + data["BEAUTY_SALES"]
        + data["ENTERTAINMENT_SALES"] + data["SPORTS_CULTURE_LEISURE_SALES"]
        + data["TRAVEL_SALES"] + data["CLOTHING_ACCESSORIES_SALES"]
    )
    data["소비활력지수"] = data["엔터전체매출"] / data["전체인구"].replace(0,np.nan)
    data["유입지수"] = data["VISITING_POPULATION"] / (
        data["RESIDENTIAL_POPULATION"] + data["WORKING_POPULATION"]
    ).replace(0,np.nan)
    data["엔터매출비율"] = data["엔터전체매출"] / data["TOTAL_SALES"].replace(0,np.nan)

    cnt_cols = [
        "FOOD_COUNT","COFFEE_COUNT","BEAUTY_COUNT","ENTERTAINMENT_COUNT",
        "SPORTS_CULTURE_LEISURE_COUNT","TRAVEL_COUNT","CLOTHING_ACCESSORIES_COUNT"
    ]
    data["엔터전체방문자수"] = data[cnt_cols].sum(axis=1)
    data["엔터방문자비율"] = data["엔터전체방문자수"] / data["TOTAL_COUNT"].replace(0,np.nan)
    data["엔터활동밀도"] = data["엔터전체매출"] / data["전체인구"].replace(0,np.nan)
    data["엔터매출밀도"] = data["엔터전체매출"] / data["엔터전체방문자수"].replace(0,np.nan)

    emo_vars = [
        "엔터전체매출","소비활력지수","유입지수","엔터매출비율",
        "엔터전체방문자수","엔터방문자비율","엔터활동밀도","엔터매출밀도"
    ]
    X = (
        data[emo_vars]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf,-np.inf], np.nan)
        .dropna()
    )
    if X.empty:
        data["FEEL_IDX"] = np.nan
        return data

    scaler = StandardScaler().fit(X)
    pc1 = PCA(n_components=1).fit_transform(scaler.transform(X))
    pc1_n = (pc1 - pc1.min()) / (pc1.max()-pc1.min()+1e-9)
    data.loc[X.index,"FEEL_IDX"] = pc1_n
    return data

@st.cache_data(show_spinner=True)
def get_data():
    return preprocess()

# ── UI ────────────────────────────────────────
st.title("서울시 인스타 감성 지수 분석")
data = get_data()

if data.empty:
    st.warning("데이터가 없습니다. 샘플링 비율을 높여 보세요.")
    st.stop()

with st.sidebar:
    districts = st.multiselect("행정동", sorted(data["DISTRICT_KOR_NAME"].dropna().unique()), [])
    age_groups = st.multiselect("연령대", sorted(data["AGE_GROUP"].unique()), [])
    gender = st.multiselect("성별", ["M","F"], [])

mask = (
    (data["DISTRICT_KOR_NAME"].isin(districts) if districts else True) &
    (data["AGE_GROUP"].isin(age_groups)       if age_groups else True) &
    (data["GENDER"].isin(gender)             if gender else True)
)
view = data.loc[mask]

st.subheader("요약 지표")
c1,c2,c3 = st.columns(3)
c1.metric("평균 FEEL_IDX", f"{view['FEEL_IDX'].mean():.2f}")
c2.metric("평균 소비활력지수", f"{view['소비활력지수'].mean():.2f}")
c3.metric("평균 유입지수", f"{view['유입지수'].mean():.2f}")

tab1,tab2,tab3 = st.tabs(["지수 상위 지역","성별·연령 분석","산점도"])

with tab1:
    top = (
        view.groupby("DISTRICT_KOR_NAME")["소비활력지수"]
        .mean()
        .sort_values(ascending=False)
        .head(20)
    )
    fig,ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=top.index, y=top.values, palette="rocket", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("행정동"); ax.set_ylabel("소비활력지수")
    st.pyplot(fig)

with tab2:
    agg = (
        view.groupby(["AGE_GROUP","GENDER"])["TOTAL_SALES"]
        .mean().reset_index()
    )
    fig,ax = plt.subplots(figsize=(8,4))
    sns.barplot(data=agg, x="AGE_GROUP", y="TOTAL_SALES", hue="GENDER",
                palette={"M":"#3498db","F":"#e75480"}, ax=ax)
    st.pyplot(fig)

with tab3:
    x_axis = st.selectbox("X축 변수", ["엔터전체매출","소비활력지수","유입지수","엔터전체방문자수"])
    y_axis = st.selectbox("Y축 변수", ["FEEL_IDX","엔터활동밀도","엔터매출비율"])
    fig,ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=view, x=x_axis, y=y_axis,
                    hue="FEEL_IDX", palette="viridis", alpha=0.6, ax=ax)
    st.pyplot(fig)

st.divider()
st.caption("데이터 출처 · Snowflake Marketplace")
