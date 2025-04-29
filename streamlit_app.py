import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st
import matplotlib.font_manager as fm
from datetime import datetime

# 테스트 모드인지 확인 (secrets.toml에서 설정)
USE_MOCK_DATA = False
try:
    USE_MOCK_DATA = st.secrets["config"]["use_mock_data"]
except:
    pass

st.set_page_config(page_title="서울시 감성 지수 대시보드", layout="wide")
sns.set_style("whitegrid")

# 한글 폰트 설정 (배포 환경 대응)
try:
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# Streamlit에서 한글 폰트 문제를 회피하기 위한 대체 방법
def korean_font():
    try:
        # 로컬 환경
        font_path = "C:/Windows/Fonts/malgun.ttf"
        if os.path.exists(font_path):
            plt.rc('font', family=fm.FontProperties(fname=font_path).get_name())
    except:
        # 배포 환경에서는 기본 폰트 사용
        try:
            plt.rcParams['font.family'] = 'DejaVu Sans'
        except:
            pass

korean_font()

# 사용 가능한 폰트 확인 및 설정
def check_fonts():
    fonts = [f.name for f in fm.fontManager.ttflist]
    for font in ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'Arial Unicode MS']:
        if font in fonts:
            plt.rcParams['font.family'] = font
            return
    # 폰트가 없으면 기본 폰트 사용
    plt.rcParams['font.family'] = 'sans-serif'

check_fonts()

# 목업 데이터 생성 함수
def create_mock_data():
    st.info("테스트 모드: 목업 데이터를 사용합니다.")
    
    # 공통 데이터 생성
    districts = ['강남구', '서초구', '마포구', '종로구', '중구', '용산구', '성동구', '영등포구', '동작구', '광진구']
    ages = ['20대', '30대', '40대', '50대', '60대 이상']
    genders = ['M', 'F']
    time_slots = ['오전', '오후', '저녁', '심야']
    weekdays = ['평일', '주말']
    n_samples = 1000
    
    # 공통 데이터
    np.random.seed(42)
    data = {
        'STANDARD_YEAR_MONTH': [datetime.now().strftime('%Y%m') for _ in range(n_samples)],
        'DISTRICT_CODE': np.random.randint(1, 26, n_samples),
        'AGE_GROUP': np.random.choice(ages, n_samples),
        'GENDER': np.random.choice(genders, n_samples),
        'TIME_SLOT': np.random.choice(time_slots, n_samples),
        'WEEKDAY_WEEKEND': np.random.choice(weekdays, n_samples),
    }
    
    # fp_df: 유동인구 데이터
    fp_df = pd.DataFrame(data.copy())
    fp_df['RESIDENTIAL_POPULATION'] = np.random.randint(100, 10000, n_samples)
    fp_df['WORKING_POPULATION'] = np.random.randint(100, 5000, n_samples)
    fp_df['VISITING_POPULATION'] = np.random.randint(50, 2000, n_samples)
    
    # card_df: 카드 매출 데이터
    card_df = pd.DataFrame(data.copy())
    sales_categories = ['FOOD', 'COFFEE', 'BEAUTY', 'ENTERTAINMENT', 
                       'SPORTS_CULTURE_LEISURE', 'TRAVEL', 'CLOTHING_ACCESSORIES']
    
    for cat in sales_categories:
        card_df[f'{cat}_SALES'] = np.random.randint(10000, 1000000, n_samples)
        card_df[f'{cat}_COUNT'] = np.random.randint(10, 1000, n_samples)
    
    card_df['TOTAL_SALES'] = card_df[[f'{cat}_SALES' for cat in sales_categories]].sum(axis=1)
    card_df['TOTAL_COUNT'] = card_df[[f'{cat}_COUNT' for cat in sales_categories]].sum(axis=1)
    
    # asset_df: 자산 소득 데이터
    asset_df = pd.DataFrame({
        'STANDARD_YEAR_MONTH': data['STANDARD_YEAR_MONTH'],
        'DISTRICT_CODE': data['DISTRICT_CODE'],
        'AGE_GROUP': data['AGE_GROUP'],
        'GENDER': data['GENDER'],
        'ASSET_VALUE': np.random.randint(10000000, 1000000000, n_samples),
        'INCOME_VALUE': np.random.randint(2000000, 10000000, n_samples)
    })
    
    # scco_df: 지역 코드 데이터
    district_codes = list(set(data['DISTRICT_CODE']))
    scco_df = pd.DataFrame({
        'DISTRICT_CODE': district_codes,
        'DISTRICT_KOR_NAME': [districts[i % len(districts)] for i in range(len(district_codes))],
        'GEOMETRY': ['POLYGON(...)' for _ in range(len(district_codes))]
    })
    
    return fp_df, card_df, asset_df, scco_df

# Snowflake 연결 (테스트 모드가 아닐 때만)
if not USE_MOCK_DATA:
    try:
        import snowflake.connector
        
        def get_conn():
            try:
                return snowflake.connector.connect(
                    user=st.secrets["snowflake"]["user"],
                    password=st.secrets["snowflake"]["password"],
                    account=st.secrets["snowflake"]["account"],
                    warehouse=st.secrets["snowflake"]["warehouse"],
                    ocsp_fail_open=True,
                    insecure_mode=True,
                )
            except Exception as e:
                st.error(f"Snowflake 연결 오류: {e}")
                st.stop()

        @st.cache_data(show_spinner=True, ttl=3600)
        def load_df(q: str) -> pd.DataFrame:
            try:
                conn = get_conn()
                cur = conn.cursor(); cur.execute(q)
                df = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
                cur.close(); conn.close()
                # 샘플 비율을 0.01로 높여 충분한 데이터 확보
                return df.sample(frac=0.01, random_state=42) if not df.empty else df
            except Exception as e:
                st.error(f"데이터 로딩 오류: {e}")
                return pd.DataFrame()
    except ImportError:
        st.error("snowflake-connector-python 패키지가 설치되어 있지 않습니다.")
        USE_MOCK_DATA = True

# 시작 시 안내 메시지
st.title("서울시 감성 지수 대시보드")
st.markdown("""
이 대시보드는 서울시의 감성 지수를 시각화합니다.
데이터가 로드되는 동안 잠시만 기다려주세요.
""")

# 데이터 로딩
if USE_MOCK_DATA:
    # 목업 데이터 사용
    fp_df, card_df, asset_df, scco_df = create_mock_data()
else:
    # Snowflake 쿼리
    Q = {
        "fp":    "SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.FLOATING_POPULATION_INFO",
        "card":  "SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.CARD_SALES_INFO",
        "asset": "SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.ASSET_INCOME_INFO",
        "scco":  "SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.M_SCCO_MST",
    }

    # 데이터 로딩 - try/except로 오류 처리
    try:
        with st.spinner("데이터 로딩 중…"):
            fp_df    = load_df(Q["fp"])
            card_df  = load_df(Q["card"])
            asset_df = load_df(Q["asset"])
            scco_df  = load_df(Q["scco"]).sample(frac=0.01, random_state=42)

        # 데이터가 비어있는지 확인
        if fp_df.empty or card_df.empty or asset_df.empty or scco_df.empty:
            st.warning("일부 데이터를 로드할 수 없습니다. 테스트 모드로 전환합니다.")
            fp_df, card_df, asset_df, scco_df = create_mock_data()
    except Exception as e:
        st.error(f"데이터 로딩 중 오류가 발생했습니다: {e}")
        st.warning("테스트 모드로 전환합니다.")
        fp_df, card_df, asset_df, scco_df = create_mock_data()

# 전처리 함수
def preprocess():
    try:
        df = (
            fp_df.merge(card_df, on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER","TIME_SLOT","WEEKDAY_WEEKEND"], how="inner")
                .merge(asset_df, on=["STANDARD_YEAR_MONTH","DISTRICT_CODE","AGE_GROUP","GENDER"], how="inner")
                .merge(scco_df, on="DISTRICT_CODE", how="inner")
        )

        if df.empty:
            st.warning("데이터 병합 후 결과가 비어있습니다.")
            return pd.DataFrame()

        # 중복 컬럼 처리
        dup = [c for c in df.columns if c.endswith("_y")]
        if dup:
            df = df.drop(columns=dup)
            rename_cols = {c: c.replace("_x", "") for c in df.columns if c.endswith("_x")}
            df = df.rename(columns=rename_cols)

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
        if not X.empty and X.shape[0] > 1:  # PCA에는 최소 2개 이상의 샘플이 필요
            pc1 = PCA(n_components=1).fit_transform(StandardScaler().fit_transform(X))
            df.loc[X.index, "FEEL_IDX"] = (pc1 - pc1.min())/(pc1.max()-pc1.min()+1e-9)

        return df
    except Exception as e:
        st.error(f"데이터 전처리 중 오류 발생: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=True, ttl=3600)
def get_data():
    return preprocess()

# 데이터 전처리 실행
df_all = get_data()

if df_all.empty:
    st.error("처리된 데이터가 비어있습니다. 데이터를 확인해주세요.")
    st.stop()

# 공통 바 차트 함수
PALETTE = "rocket"

def bar(df, x, y, ttl):
    if df.empty:
        st.warning("차트를 그릴 데이터가 없습니다.")
        return
    
    try:
        fig, ax = plt.subplots(figsize=(14,7))
        sns.barplot(data=df, x=x, y=y, palette=PALETTE, ax=ax)
        ax.set_title(ttl)
        ax.set_xlabel("")
        ax.set_ylabel(y)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"차트 생성 오류: {e}")

# 사이드바
st.sidebar.header("메뉴")
page = st.sidebar.selectbox("페이지", ("방문자수 TOP","방문자 1인당 매출","유입지수 TOP","성별·연령 소비","상관관계"))

a = df_all  # alias

try:
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
    else:  # 상관관계
        cols = ["엔터전체매출","유입지수","엔터매출비율","엔터전체방문자수","엔터방문자비율","엔터활동밀도","엔터매출밀도","FEEL_IDX"]
        cdf = a[cols].dropna()
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
except Exception as e:
    st.error(f"차트 생성 중 오류가 발생했습니다: {e}")

st.success("완료"); 
