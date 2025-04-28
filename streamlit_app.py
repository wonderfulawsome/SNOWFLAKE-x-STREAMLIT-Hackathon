import os, streamlit as st, snowflake.connector as sf
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler; from sklearn.decomposition import PCA

os.environ["STREAMLIT_SERVER_HEALTH_CHECK_ENABLED"] = "false"
st.set_page_config(page_title="서울시 감성 지수 대시보드", layout="wide"); sns.set_style("whitegrid")
import matplotlib; matplotlib.rcParams["font.family"]="Malgun Gothic"; matplotlib.rcParams["axes.unicode_minus"]=False
st.markdown('<style>*{font-family:"Malgun Gothic",sans-serif!important;}</style>',unsafe_allow_html=True)
st.write("🚀 Streamlit 앱 시작!")

def get_conn():
    return sf.connect(user=st.secrets["snowflake"]["user"],
                      password=st.secrets["snowflake"]["password"],
                      account=st.secrets["snowflake"]["account"],
                      warehouse="COMPUTE_WH",
                      ocsp_fail_open=True,insecure_mode=True)

@st.cache_data(show_spinner=False)
def load(q):
    with get_conn() as c:
        cur=c.cursor(); cur.execute(q)
        df=pd.DataFrame(cur.fetchall(),columns=[x[0] for x in cur.description]); cur.close()
    return df

BASE="SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA"
Q_FP   =f"SELECT * FROM {BASE}.FLOATING_POPULATION_INFO"
Q_CARD =f"SELECT * FROM {BASE}.CARD_SALES_INFO"
Q_ASSET=f"SELECT * FROM {BASE}.ASSET_INCOME_INFO"
Q_SCCO =f"SELECT * FROM {BASE}.M_SCCO_MST"

@st.cache_data(show_spinner=True)
def preprocess():
    fp,card,asset,scco=load(Q_FP),load(Q_CARD),load(Q_ASSET),load(Q_SCCO)
    fp=fp.sample(frac=0.05,random_state=42)                      # 5 %만 메모리 탑재
    fp_card=pd.merge(fp,card,on=[
        "STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER",
        "TIME_SLOT","WEEKDAY_WEEKEND"],how="left")               # ← left join
    data=pd.merge(fp_card,asset,
        on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER"],
        how="left")                                              # ← left join

    scco_map=scco.drop_duplicates("DISTRICT_CODE").set_index("DISTRICT_CODE")["DISTRICT_KOR_NAME"]
    data["DISTRICT_KOR_NAME"]=data["DISTRICT_CODE"].map(scco_map)

    data["전체인구"]=data["RESIDENTIAL_POPULATION"]+data["WORKING_POPULATION"]+data["VISITING_POPULATION"]
    data["엔터전체매출"]=(
        data["FOOD_SALES"]+data["COFFEE_SALES"]+data["BEAUTY_SALES"]+
        data["ENTERTAINMENT_SALES"]+data["SPORTS_CULTURE_LEISURE_SALES"]+
        data["TRAVEL_SALES"]+data["CLOTHING_ACCESSORIES_SALES"])
    data["소비활력지수"]=data["엔터전체매출"]/data["전체인구"].replace(0,np.nan)
    data["유입지수"]=data["VISITING_POPULATION"]/(
        data["RESIDENTIAL_POPULATION"]+data["WORKING_POPULATION"]).replace(0,np.nan)
    data["엔터매출비율"]=data["엔터전체매출"]/data["TOTAL_SALES"].replace(0,np.nan)

    cnt_cols=[
        "FOOD_COUNT","COFFEE_COUNT","BEAUTY_COUNT","ENTERTAINMENT_COUNT",
        "SPORTS_CULTURE_LEISURE_COUNT","TRAVEL_COUNT","CLOTHING_ACCESSORIES_COUNT"]
    data["엔터전체방문자수"]=data[cnt_cols].sum(axis=1)
    data["엔터방문자비율"]=data["엔터전체방문자수"]/data["TOTAL_COUNT"].replace(0,np.nan)
    data["엔터활동밀도"]=data["엔터전체매출"]/data["전체인구"].replace(0,np.nan)
    data["엔터매출밀도"]=data["엔터전체매출"]/data["엔터전체방문자수"].replace(0,np.nan)

    emo_vars=["엔터전체매출","소비활력지수","유입지수","엔터매출비율",
              "엔터전체방문자수","엔터방문자비율","엔터활동밀도","엔터매출밀도"]
    X=(data[emo_vars].apply(pd.to_numeric,errors="coerce")
         .replace([np.inf,-np.inf],np.nan).dropna())
    if not X.empty:
        pc1=PCA(n_components=1).fit_transform(StandardScaler().fit_transform(X))
        data.loc[X.index,"FEEL_IDX"]=(pc1-pc1.min())/(pc1.max()-pc1.min()+1e-9)
    return data

data=preprocess()
st.title("서울시 인스타 감성 지수 분석")

if data.empty:
    st.error("데이터를 찾을 수 없습니다.")
    st.stop()

with st.sidebar:
    districts=st.multiselect("행정동",sorted(data["DISTRICT_KOR_NAME"].dropna().unique()),[])
    age_groups=st.multiselect("연령대",sorted(data["AGE_GROUP"].unique()),[])
    gender=st.multiselect("성별",["M","F"],[])

mask=((data["DISTRICT_KOR_NAME"].isin(districts) if districts else True) &
      (data["AGE_GROUP"].isin(age_groups)        if age_groups else True) &
      (data["GENDER"].isin(gender)              if gender else True))
view=data.loc[mask]

c1,c2,c3=st.columns(3)
c1.metric("평균 FEEL_IDX",f"{view['FEEL_IDX'].mean():.2f}")
c2.metric("평균 소비활력지수",f"{view['소비활력지수'].mean():.2f}")
c3.metric("평균 유입지수",f"{view['유입지수'].mean():.2f}")

tab1,tab2,tab3=st.tabs(["지수 상위 지역","성별·연령 분석","산점도"])

with tab1:
    top=view.groupby("DISTRICT_KOR_NAME")["소비활력지수"].mean().nlargest(20)
    fig,ax=plt.subplots(figsize=(10,5))
    sns.barplot(x=top.index,y=top.values,palette="rocket",ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha="right")
    ax.set_xlabel("행정동"); ax.set_ylabel("소비활력지수"); st.pyplot(fig)

with tab2:
    agg=view.groupby(["AGE_GROUP","GENDER"])["TOTAL_SALES"].mean().reset_index()
    fig,ax=plt.subplots(figsize=(8,4))
    sns.barplot(data=agg,x="AGE_GROUP",y="TOTAL_SALES",hue="GENDER",
                palette={"M":"#3498db","F":"#e75480"},ax=ax); st.pyplot(fig)

with tab3:
    x=st.selectbox("X축",["엔터전체매출","소비활력지수","유입지수","엔터전체방문자수"])
    y=st.selectbox("Y축",["FEEL_IDX","엔터활동밀도","엔터매출비율"])
    fig,ax=plt.subplots(figsize=(6,4))
    sns.scatterplot(data=view,x=x,y=y,hue="FEEL_IDX",palette="viridis",alpha=0.6,ax=ax)
    st.pyplot(fig)

st.divider(); st.caption("데이터 출처 · Snowflake Marketplace")
