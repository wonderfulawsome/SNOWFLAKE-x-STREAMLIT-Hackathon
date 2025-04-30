콘텐츠로 건너뛰기
나의 말:

import snowflake.connector

# snowflake 데이터베이스 연결
conn = snowflake.connector.connect(
    user='___',
    password='___',
    account='___',  
    warehouse='___',
)
import pandas as pd

####################### 1. 백화점 데이터
# 커서 생성 및 쿼리 실행
cursor = conn.cursor()
cursor.execute("SELECT * FROM DEPARTMENT_STORE_FOOT_TRAFFIC_FOR_SNOWFLAKE_STREAMLIT_HACKATHON.PUBLIC.SNOWFLAKE_STREAMLIT_HACKATHON_LOPLAT_DEPARTMENT_STORE_DATA")
# 전체 데이터를 DataFrame으로 저장
department_store_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])


####################### 2. 서울 기온 및 강수량 데이터
# 커서 생성 및 쿼리 실행
cursor = conn.cursor()
cursor.execute("SELECT * FROM SEOUL_TEMPERATURE__RAINFALL_FOR_SNOWFLAKE_STREAMLIT_HACKATHON.PUBLIC.SNOWFLAKE_STREAMLIT_HACKATHON_LOPLAT_SEOUL_TEMPERATURE_RAINFALL")
# 전체 데이터를 DataFrame으로 저장
seoul_weather_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])


####################### 3. 주거지/근무지 데이터
# 커서 생성 및 쿼리 실행
cursor = conn.cursor()
cursor.execute("SELECT * FROM RESIDENTIAL__WORKPLACE_TRAFFIC_PATTERNS_FOR_SNOWFLAKE_STREAMLIT_HACKATHON.PUBLIC.SNOWFLAKE_STREAMLIT_HACKATHON_LOPLAT_HOME_OFFICE_RATIO;")
# 전체 데이터를 DataFrame으로 저장
home_office_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])





####################### 1. 유동 인구 데이터
# 커서 생성 및 쿼리 실행
cursor = conn.cursor()
cursor.execute("SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.FLOATING_POPULATION_INFO")
# 전체 데이터를 DataFrame으로 저장
floating_population_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])


####################### 2. 신한카드 사용 데이터
# 커서 생성 및 쿼리 실행
cursor = conn.cursor()
cursor.execute("SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.CARD_SALES_INFO")
# 전체 데이터를 DataFrame으로 저장
card_sales_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])


####################### 3. 소득 데이터
# 커서 생성 및 쿼리 실행
cursor = conn.cursor()
cursor.execute("SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.ASSET_INCOME_INFO;")
# 전체 데이터를 DataFrame으로 저장
asset_income_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])


####################### 4. 행정동 경계 데이터
# 커서 생성 및 쿼리 실행
cursor = conn.cursor()
cursor.execute("SELECT * FROM SEOUL_DISTRICTLEVEL_DATA_FLOATING_POPULATION_CONSUMPTION_AND_ASSETS.GRANDATA.M_SCCO_MST;")
# 전체 데이터를 DataFrame으로 저장
scco_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
스노우플레이크 커넥트 라이브러리를 활용하여 데이터베이스와 연결하여 쿼리문을통해 데이터를 변수명에 저장

ds_weather_df = pd.merge(
    department_store_df,
    seoul_weather_df,
    on='DATE_KST',
    how='left'
)
fp_card_df = pd.merge(
    floating_population_df,
    card_sales_df,
    on=['STANDARD_YEAR_MONTH', 'DISTRICT_CODE', 'AGE_GROUP', 'GENDER', 'TIME_SLOT', 'WEEKDAY_WEEKEND'],
    how='inner'
)
fp_card_asset = pd.merge(
    fp_card_df,
    asset_income_df,
    on=['STANDARD_YEAR_MONTH', 'DISTRICT_CODE', 'AGE_GROUP', 'GENDER'],
    how='inner'
)
fp_card_scco_asset = pd.merge(
    fp_card_asset,
    scco_df,
    on=['DISTRICT_CODE'],
    how='inner'
)
C:\Users\82106\AppData\Local\Temp\ipykernel_24760\263647221.py:1: FutureWarning: Passing 'suffixes' which cause duplicate columns {'PROVINCE_CODE_x', 'CITY_CODE_x'} in the result is deprecated and will raise a MergeError in a future version.
  fp_card_scco_asset = pd.merge(
공통 칼럼을 활용하여, 데이터 병합을 진행

# 필요 없는 컬럼 삭제
fp_card_scco_asset = fp_card_scco_asset.drop(columns=['PROVINCE_CODE_y', 'CITY_CODE_y'])
# 컬럼명 변경
fp_card_scco_asset = fp_card_scco_asset.rename(columns={
    'PROVINCE_CODE_x': 'PROVINCE_CODE',
    'CITY_CODE_x': 'CITY_CODE'
})
중복되는 칼럼을 삭제하고 칼럼명을 변경해준다.

전체인구, 엔터전체매출, 소비활력지수 계산
#한글폰트
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 경로 설정
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
import numpy as np

# 인구 합계 컬럼 생성 (거주+근무+방문 인구)
fp_card_scco_asset['전체인구'] = (
    fp_card_scco_asset['RESIDENTIAL_POPULATION'] +
    fp_card_scco_asset['WORKING_POPULATION'] +
    fp_card_scco_asset['VISITING_POPULATION']
)

# 인스타 감성 소비 매출 합계 컬럼 생성 (FOOD + COFFEE + BEAUTY + ENTERTAINMENT + SPORTS/CULTURE/LEISURE + TRAVEL)
fp_card_scco_asset['엔터전체매출'] = (
    fp_card_scco_asset['FOOD_SALES'] +
    fp_card_scco_asset['COFFEE_SALES'] + 
    fp_card_scco_asset['BEAUTY_SALES'] +
    fp_card_scco_asset['ENTERTAINMENT_SALES'] +
    fp_card_scco_asset['SPORTS_CULTURE_LEISURE_SALES'] +
    fp_card_scco_asset['TRAVEL_SALES']+
    fp_card_scco_asset['CLOTHING_ACCESSORIES_SALES']
)
소비활력지수 = 엔터매출 / 전체인구 = 한 사람당 평균적으로 얼마나 엔터테인먼트에 소비했는지

# 엔터 관련 count 칼럼만 선택
enter_count_columns = [
    'FOOD_COUNT',
    'COFFEE_COUNT',
    'BEAUTY_COUNT',
    'ENTERTAINMENT_COUNT',
    'SPORTS_CULTURE_LEISURE_COUNT',
    'TRAVEL_COUNT',
    'CLOTHING_ACCESSORIES_COUNT'
]

# 엔터 관련 방문횟수 합계 컬럼 생성
fp_card_scco_asset['엔터전체방문자수'] = fp_card_scco_asset[enter_count_columns].sum(axis=1)
# 2) 활동밀도(1인당 감성 소비액) 계산
fp_card_scco_asset['엔터활동밀도'] = (
    fp_card_scco_asset['엔터전체매출'] /
    fp_card_scco_asset['전체인구'].replace(0, np.nan)  # 0으로 나누기 방지
)
지역별 방문자수 순위
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 지역별 방문자수 합계 집계
agg = fp_card_scco_asset.groupby('DISTRICT_KOR_NAME')['엔터전체방문자수'].sum().reset_index()

# 2. 방문자수 내림차순 정렬
agg = agg.sort_values('엔터전체방문자수', ascending=False).head(30)

# 3. 바차트 시각화
plt.figure(figsize=(14, 7))
sns.barplot(data=agg, x='DISTRICT_KOR_NAME', y='엔터전체방문자수', palette='rocket')
plt.title('지역별 방문자수(엔터전체방문자수) 상위 지역')
plt.xlabel('행정동명')
plt.ylabel('엔터전체방문자수')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import json
from shapely.geometry import shape

# 1. GeoDataFrame 생성 (GeoJSON 형식일 경우)
def geojson_to_shape(geojson_str):
    return shape(json.loads(geojson_str))

gdf = fp_card_scco_asset[['DISTRICT_KOR_NAME', '엔터전체방문자수', 'DISTRICT_GEOM']].drop_duplicates('DISTRICT_KOR_NAME').copy()
gdf['geometry'] = gdf['DISTRICT_GEOM'].apply(geojson_to_shape)
gdf = gpd.GeoDataFrame(gdf, geometry='geometry')

# 2. 좌표계 지정 (위경도라고 가정)
gdf.set_crs(epsg=4326, inplace=True)

# 3. 지도 시각화 (GeoPandas + matplotlib)
plt.figure(figsize=(12, 10))
gdf.plot(
    column='엔터전체방문자수',
    cmap='Blues',
    legend=True,
    edgecolor='black',
    linewidth=0.5
)
plt.title('행정동별 방문자수(엔터전체방문자수) 지도 시각화')
plt.axis('off')
plt.show()
<Figure size 864x720 with 0 Axes>

지역별 방문자1인당 엔터매출
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 엔터 관련 방문횟수 합계 컬럼 생성
fp_card_scco_asset['엔터전체방문자수'] = fp_card_scco_asset[enter_count_columns].sum(axis=1)

# 1. 지역별로 엔터전체매출, 엔터전체방문자수 합계 집계
agg = fp_card_scco_asset.groupby('DISTRICT_KOR_NAME')[['엔터전체매출', '엔터전체방문자수']].sum().reset_index()

# 2. 방문자 1인당 엔터매출 계산
agg['방문자1인당엔터매출'] = agg['엔터전체매출'] / agg['엔터전체방문자수'].replace(0, np.nan)

# 3. 내림차순 정렬
agg = agg.sort_values('방문자1인당엔터매출', ascending=False).head(30)

# 4. 바차트 시각화
plt.figure(figsize=(14, 7))
sns.barplot(data=agg, x='DISTRICT_KOR_NAME', y='방문자1인당엔터매출', palette='rocket')
plt.title('지역별 방문자 1인당 엔터매출')
plt.xlabel('행정동명')
plt.ylabel('방문자 1인당 엔터매출')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 1. GeoDataFrame 생성 (GeoJSON 형식일 경우)
def geojson_to_shape(geojson_str):
    return shape(json.loads(geojson_str))

gdf = fp_card_scco_asset[['DISTRICT_KOR_NAME', '엔터전체매출', '엔터전체방문자수', 'DISTRICT_GEOM']].drop_duplicates('DISTRICT_KOR_NAME').copy()
gdf['geometry'] = gdf['DISTRICT_GEOM'].apply(geojson_to_shape)
gdf = gpd.GeoDataFrame(gdf, geometry='geometry')

# 2. 좌표계 지정 (위경도라고 가정)
gdf.set_crs(epsg=4326, inplace=True)

# 3. 방문자 1인당 엔터매출 계산
gdf['방문자1인당엔터매출'] = gdf['엔터전체매출'] / gdf['엔터전체방문자수'].replace(0, np.nan)

# 4. 지도 시각화
plt.figure(figsize=(12, 10))
gdf.plot(
    column='방문자1인당엔터매출',
    cmap='YlOrRd',
    legend=True,
    edgecolor='black',
    linewidth=0.5
)
plt.title('지역별 방문자 1인당 엔터매출 지도 시각화')
plt.axis('off')
plt.show()
<Figure size 864x720 with 0 Axes>

방문자1인당엔터매출 = 엔터전체매출 / 엔터전체방문자수

유입 지수 계산
유입지수 = 방문자수 / 주거자수 + 근로자수

import seaborn as sns

#유입지수 계산
fp_card_scco_asset['유입지수'] = fp_card_scco_asset['VISITING_POPULATION'] / (
    fp_card_scco_asset['RESIDENTIAL_POPULATION'] + 
    fp_card_scco_asset['WORKING_POPULATION']
).replace(0, np.nan)

# DISTRICT_KOR_NAME별 유입지수 평균 집계
agg_inflow = fp_card_scco_asset.groupby('DISTRICT_KOR_NAME')['유입지수'].mean().reset_index()

# 유입지수 내림차순 정렬
agg_inflow_head = agg_inflow.sort_values('유입지수', ascending=False).head(20)

# 더 세련된 스타일 설정
plt.style.use('seaborn')
plt.figure(figsize=(14, 7))

#한글폰트
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 경로 설정
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)

# 바 플롯 생성 및 커스터마이징
ax = sns.barplot(data=agg_inflow_head, x='DISTRICT_KOR_NAME', y='유입지수', 
                 palette='rocket', alpha=0.8)

# 바 위에 값 표시
for i, v in enumerate(agg_inflow_head['유입지수']):
    ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')

# 제목과 레이블 스타일링
plt.title('행정동별 유입지수 평균', pad=20, fontsize=15, fontweight='bold')
plt.xlabel('행정동명', fontsize=12)
plt.ylabel('유입지수(평균)', fontsize=12)

# x축 레이블 회전 및 그리드 설정
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 테두리 스타일링
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

성별 연령별 1일평균 소비량
# 1. GENDER, AGE_GROUP, 날짜별 1인당 소비지수 평균 집계
daily_group = fp_card_scco_asset.groupby(['GENDER', 'AGE_GROUP', 'STANDARD_YEAR_MONTH'])['TOTAL_SALES'].mean().reset_index()


# 2. GENDER, AGE_GROUP별 1일 평균(즉, 일별 평균의 평균)
agg_gender_age = daily_group.groupby(['GENDER', 'AGE_GROUP'])['TOTAL_SALES'].mean().reset_index()
#한글폰트
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 경로 설정
font_path = 'C:/Windows/Fonts/malgun.ttf'
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)

# 남자는 파란색, 여자는 핑크색으로 컬러 지정
gender_palette = {'M': '#3498db', 'F': '#e75480'}  # 남: 파랑, 여: 핑크

plt.figure(figsize=(12, 6))
sns.barplot(data=agg_gender_age, x='AGE_GROUP', y='TOTAL_SALES', hue='GENDER', palette=gender_palette)
plt.title('성별·연령대별 1일 평균 1인당 매출')
plt.xlabel('연령대')
plt.ylabel('1일 평균 1인당 매출량')
plt.legend(title='성별')
plt.tight_layout()
plt.show()

전체 매출대비 엔터 매출 비율 시간별 추이
엔터매출비율 = 엔터전체매출 / TOTAL_SALES

import matplotlib.pyplot as plt
import seaborn as sns

# 1. ENTER_RATE 계산 및 컬럼 추가
fp_card_scco_asset['엔터매출비율'] = fp_card_scco_asset['엔터전체매출'] / fp_card_scco_asset['TOTAL_SALES'].replace(0, np.nan)

# 2. NaN이 있는 행 제거
df = fp_card_scco_asset.dropna(subset=['엔터매출비율', 'TIME_SLOT'])

# 3. 시간대별 ENTER_RATE 평균 집계
timeslot = df.groupby('TIME_SLOT')['엔터매출비율'].mean().reset_index()

# 4. TIME_SLOT이 숫자라면 정렬
try:
    timeslot['TIME_SLOT'] = timeslot['TIME_SLOT'].astype(int)
    timeslot = timeslot.sort_values('TIME_SLOT')
except:
    timeslot = timeslot.sort_values('TIME_SLOT')

# 5. 시각화
plt.figure(figsize=(10, 5))
sns.lineplot(data=timeslot, x='TIME_SLOT', y='엔터매출비율', marker='o')
plt.title('시간대별 ENTER_RATE 추이')
plt.xlabel('TIME_SLOT')
plt.ylabel('ENTER_RATE')
plt.tight_layout()
plt.show()

전체 방문자수 대비 엔터 방문자수 비율 시간별 추이
엔터방문자비율 = 엔터전체방문자수 / TOTAL_COUNT

fp_card_scco_asset['엔터방문자비율'] = fp_card_scco_asset['엔터전체방문자수'] / fp_card_scco_asset['TOTAL_COUNT']

fp_card_scco_asset['INFLOW_IDX'] = fp_card_scco_asset['VISITING_POPULATION'] / (
                       fp_card_scco_asset['RESIDENTIAL_POPULATION'] + fp_card_scco_asset['WORKING_POPULATION'] + 1e-9)
연령별/성별 ENTER 매출 합계
import seaborn as sns

# 연령대+성별 그룹별 ENTER_TOTAL_SALES 합계
age_gender_sales = fp_card_scco_asset.groupby(['AGE_GROUP', 'GENDER'])['엔터전체매출'].sum().reset_index()

# 남자는 파란색, 여자는 핑크색으로 컬러 지정
gender_palette = {'M': '#3498db', 'F': '#e75480'} 

plt.figure(figsize=(10, 6))
sns.barplot(data=age_gender_sales, x='AGE_GROUP', y='엔터전체매출', hue='GENDER', palette=gender_palette)
plt.title('연령대/성별 엔터테인먼트 소비 매출 합계')
plt.ylabel('엔터전체매출')
plt.xlabel('AGE_GROUP')
plt.legend(title='GENDER')
plt.tight_layout()
plt.show()

연령별/성별 1인당 엔터 소비액
# 2) 활동밀도(1인당 감성 소비액) 계산
fp_card_scco_asset['엔터활동밀도'] = (
    fp_card_scco_asset['엔터전체매출'] /
    fp_card_scco_asset['전체인구'].replace(0, np.nan)  # 0으로 나누기 방지
)

# 연령대+성별 그룹별 ENTER_ACTIVITY_DENSITY 평균 계산 및 시각화
age_gender_density = fp_card_scco_asset.groupby(['AGE_GROUP', 'GENDER'])['엔터활동밀도'].mean().reset_index()

# 남자는 파란색, 여자는 핑크색으로 컬러 지정
gender_palette = {'M': '#3498db', 'F': '#e75480'} 

plt.figure(figsize=(10, 6))
sns.barplot(data=age_gender_density, x='AGE_GROUP', y='엔터활동밀도', hue='GENDER', palette=gender_palette)
plt.title('연령대/성별 엔터테인먼트 활동 밀도(1인당 감성 소비액)')
plt.ylabel('엔터활동밀도')
plt.xlabel('AGE_GROUP')
plt.legend(title='GENDER')
plt.tight_layout()
plt.show()

연령/성별 전체 방문횟수 합계
import seaborn as sns

# 연령대+성별 그룹별 ENTER_TOTAL_SALES 합계
age_gender_sales = fp_card_scco_asset.groupby(['AGE_GROUP', 'GENDER'])['VISITING_POPULATION'].sum().reset_index()

# 남자는 파란색, 여자는 핑크색으로 컬러 지정
gender_palette = {'M': '#3498db', 'F': '#e75480'} 

plt.figure(figsize=(10, 6))
sns.barplot(data=age_gender_sales, x='AGE_GROUP', y='VISITING_POPULATION', hue='GENDER', palette=gender_palette)
plt.title('연령대/성별 전체 방문횟수 합계')
plt.ylabel('VISITING_POPULATION')
plt.xlabel('AGE_GROUP')
plt.legend(title='GENDER')
plt.tight_layout()
plt.show()

연령/성별 엔터관련 방문횟수 합계
# 연령/성별별 엔터 관련 방문횟수 합계 집계
grouped = fp_card_scco_asset.groupby(['AGE_GROUP', 'GENDER'])['엔터전체방문자수'].sum().reset_index()

# 남자는 파란색, 여자는 핑크색으로 컬러 지정
gender_palette = {'M': '#3498db', 'F': '#e75480'} 

# 시각화
plt.figure(figsize=(12, 6))
sns.barplot(data=grouped, x='AGE_GROUP', y='엔터전체방문자수', hue='GENDER', palette=gender_palette)
plt.title('연령/성별별 엔터 관련 방문횟수 합계')
plt.xlabel('AGE_GROUP')
plt.ylabel('ENTER 관련 방문횟수 합계')
plt.legend(title='GENDER')
plt.tight_layout()
plt.show()

엔터 매출 밀도(방문 1명당 매출)
엔터매출밀도 = 엔터전체매출 / 엔터전체방문자수

# 엔터 매출 밀도(방문 1회당 매출) 컬럼 생성
fp_card_scco_asset['엔터매출밀도'] = fp_card_scco_asset['엔터전체매출'] / fp_card_scco_asset['엔터전체방문자수'].replace(0, np.nan)

# 연령/성별별 엔터 매출 밀도 평균 집계
grouped = fp_card_scco_asset.groupby(['AGE_GROUP', 'GENDER'])['엔터매출밀도'].mean().reset_index()


# 남자는 파란색, 여자는 핑크색으로 컬러 지정
gender_palette = {'M': '#3498db', 'F': '#e75480'} 

# 시각화
plt.figure(figsize=(12, 6))
sns.barplot(data=grouped, x='AGE_GROUP', y='엔터매출밀도', hue='GENDER', palette=gender_palette)
plt.title('연령/성별 엔터 매출 밀도(방문 1회당 매출)')
plt.xlabel('AGE_GROUP')
plt.ylabel('방문 1회당 엔터 매출(밀도)')
plt.legend(title='GENDER')
plt.tight_layout()
plt.show()

방문1회당 매출이 높은건 남성이 더 높은거로 나타났다.

비교적, 남성은 주로 방문보다는 구매에 초점이 맞춰져있고,

여성은 구매 보다는 방문에 초점이 맞춰져있다는것을 알수있다.

또한, 연령이 높아질수록 두 성별모두 구매에 초점을맞추는 경향이 짙어진다는것도 알수있다.

엔터 지수 예측 모델
PCA(주성분 분석) 통해 가중치 조합 찾기
FEEL_IDX : 종합감성지수 = 엔터소비활동지수
- 즉시 소비력, 소비의 활력
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

vars_emo = [
    '엔터전체매출',
    '유입지수',
    '엔터매출비율',
    '엔터전체방문자수',
    '엔터방문자비율',
    '엔터활동밀도',
    '엔터매출밀도'
]

# NaN이 있는 행 제거
X = fp_card_scco_asset[vars_emo].dropna()

# ① 표준화
X_std = StandardScaler().fit_transform(X)

# ② 첫 번째 주성분 추출
pc1 = PCA(n_components=1).fit_transform(X_std)

# ③ 0~1 스케일로 변환해 감성지수로 저장
pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-9)

# 인덱스를 맞춰서 FEEL_IDX 할당
fp_card_scco_asset.loc[X.index, 'FEEL_IDX'] = pc1_norm
# ② PCA 객체 생성 및 학습
pca = PCA(n_components=1)
pca.fit(X_std)

# ③ 첫 번째 주성분의 loading(가중치) 추출
loadings = pca.components_[0]

# ④ 보기 좋게 데이터프레임으로 출력
weights_df = pd.DataFrame({'칼럼명': vars_emo, '가중치(loading)': loadings})
print(weights_df)
        칼럼명  가중치(loading)
0    엔터전체매출      0.564407
1      유입지수      0.047140
2    엔터매출비율      0.104359
3  엔터전체방문자수      0.617971
4   엔터방문자비율      0.120349
5    엔터활동밀도      0.517641
6    엔터매출밀도      0.063302
weights_df_sorted = weights_df.reindex(weights_df['가중치(loading)'].abs().sort_values(ascending=False).index)

plt.figure(figsize=(10, 6))
sns.barplot(data=weights_df_sorted, x='가중치(loading)', y='칼럼명', palette='rocket')
plt.title('PCA 첫 번째 주성분 가중치(loading) (절대값 내림차순)')
plt.xlabel('가중치(loading)')
plt.ylabel('칼럼명')
plt.tight_layout()
plt.show()

fp_card_scco_asset['FEEL_IDX']
0          0.006936
1          0.006936
2          0.001663
3          0.001663
4          0.000691
             ...   
6454697    0.013696
6454698    0.001997
6454699         NaN
6454700    0.010246
6454701         NaN
Name: FEEL_IDX, Length: 6454702, dtype: float64
감성지수와 각 변수별 산점도 시각화
# sales_columns 예시 (원하는 컬럼명으로 수정)
sales_columns = [
    'FOOD_SALES',
    'COFFEE_SALES',
    'BEAUTY_SALES',
    'ENTERTAINMENT_SALES',
    'SPORTS_CULTURE_LEISURE_SALES',
    'TRAVEL_SALES',
    'CLOTHING_ACCESSORIES_SALES'
]

# FEEL_IDX와 sales_columns만 추출
corr_df = fp_card_scco_asset[['FEEL_IDX'] + sales_columns].dropna()

# 상관계수 계산
corr = corr_df.corr()

# FEEL_IDX와 각 sales 컬럼의 상관계수만 추출
feel_corr = corr.loc[sales_columns, 'FEEL_IDX']

# 1. 바차트로 시각화
plt.figure(figsize=(8, 5))
sns.barplot(x=feel_corr.values, y=feel_corr.index, palette='rocket')
plt.title('FEEL_IDX와 업종별 매출 상관계수')
plt.xlabel('상관계수')
plt.ylabel('매출 컬럼')
plt.xlim(-1, 1)
plt.tight_layout()
plt.show()
C:\Users\82106\AppData\Local\Temp\ipykernel_24760\1710878054.py:28: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from current font.
  plt.tight_layout()
c:\Users\82106\anaconda3\lib\site-packages\IPython\core\pylabtools.py:151: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from current font.
  fig.canvas.print_figure(bytes_io, **kw)

sales_columns = [
    'FOOD_SALES',
    'COFFEE_SALES',
    'BEAUTY_SALES',
    'ENTERTAINMENT_SALES',
    'SPORTS_CULTURE_LEISURE_SALES',
    'TRAVEL_SALES',
    'CLOTHING_ACCESSORIES_SALES'
]

n_cols = 3
n_rows = (len(sales_columns) + n_cols - 1) // n_cols

plt.figure(figsize=(6 * n_cols, 5 * n_rows))

for i, col in enumerate(sales_columns):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.scatter(fp_card_scco_asset[col], fp_card_scco_asset['FEEL_IDX'], 
                c=fp_card_scco_asset['FEEL_IDX'], 
                cmap='viridis',
                alpha=0.5)
    plt.xlabel(col)
    plt.ylabel('FEEL_IDX')
    plt.title(f'FEEL_IDX vs {col}')

plt.tight_layout()
plt.show()

# 분석할 칼럼 리스트
columns = [
    'FEEL_IDX',
    '엔터전체매출',
    '유입지수',
    '엔터매출비율',
    '엔터전체방문자수',
    '엔터방문자비율',
    '엔터활동밀도',
    '엔터매출밀도'
]

# 결측치 제거
corr_df = fp_card_scco_asset[columns].dropna()

# 상관계수 계산
corr = corr_df.corr()

# 1. 히트맵 시각화
plt.figure(figsize=(8, 7))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title('FEEL_IDX 및 주요 지표 간 상관관계 히트맵')
plt.tight_layout()
plt.show()

# 2. FEEL_IDX와 나머지 칼럼의 상관계수만 바차트로
feel_corr = corr.loc[columns[1:], 'FEEL_IDX']

plt.figure(figsize=(7, 5))
sns.barplot(x=feel_corr.values, y=feel_corr.index, palette='rocket')
plt.title('FEEL_IDX와 주요 지표별 상관계수')
plt.xlabel('상관계수')
plt.ylabel('지표명')
plt.xlim(-1, 1)
plt.tight_layout()
plt.show()
c:\Users\82106\anaconda3\lib\site-packages\seaborn\utils.py:95: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from current font.
  fig.canvas.draw()
C:\Users\82106\AppData\Local\Temp\ipykernel_24760\719483456.py:23: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from current font.
  plt.tight_layout()
c:\Users\82106\anaconda3\lib\site-packages\IPython\core\pylabtools.py:151: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from current font.
  fig.canvas.print_figure(bytes_io, **kw)

C:\Users\82106\AppData\Local\Temp\ipykernel_24760\719483456.py:35: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from current font.
  plt.tight_layout()
c:\Users\82106\anaconda3\lib\site-packages\IPython\core\pylabtools.py:151: UserWarning: Glyph 8722 (\N{MINUS SIGN}) missing from current font.
  fig.canvas.print_figure(bytes_io, **kw)

columns = [
    '엔터전체매출',
    '유입지수',
    '엔터매출비율',
    '엔터전체방문자수',
    '엔터방문자비율',
    '엔터활동밀도',
    '엔터매출밀도'
]

n_cols = 3
n_rows = (len(columns) + n_cols - 1) // n_cols

plt.figure(figsize=(6 * n_cols, 5 * n_rows))

for i, col in enumerate(columns):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.scatter(fp_card_scco_asset[col], fp_card_scco_asset['FEEL_IDX'],
                c=fp_card_scco_asset['FEEL_IDX'], 
                cmap='viridis',
                alpha=0.5)
    plt.xlabel(col)
    plt.ylabel('FEEL_IDX')
    plt.title(f'FEEL_IDX vs {col}')

plt.tight_layout()
plt.show()


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 기본 세팅
st.set_page_config(page_title="서울시 감성 지수 대시보드", layout="wide")
sns.set_style("whitegrid")
            
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
import streamlit as st

# ── 1. NanumGothic 폰트 로드 (리포지토리 fonts 폴더)
font_path = os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic.TTF")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    nanum_font_name = fm.FontProperties(fname=font_path).get_name()
    matplotlib.rcParams['font.family'] = nanum_font_name
    matplotlib.rcParams['axes.unicode_minus'] = False
else:
    st.warning("❌ NanumGothic.TTF 파일이 fonts 폴더에 없습니다.")

# ── 2. Streamlit UI CSS로도 강제 적용
st.markdown(f"""
<style>
@font-face {{
    font-family: 'NanumGothic';
    src: url('fonts/NanumGothic.TTF') format('truetype');
}}
html, body, [class*="css"] {{
    font-family: 'NanumGothic', 'sans-serif';
}}
</style>
""", unsafe_allow_html=True)
            
# 데이터 로드
DATA_DIR = Path(__file__).parent / "data"

@st.cache_data(show_spinner=False)
def load_data():
    fp = pd.read_csv(DATA_DIR / "floating population.csv")
    card = pd.read_csv(DATA_DIR / "Shinhan card sales.csv")
    scco = pd.read_csv(DATA_DIR / "scco.csv")
    return fp, card, scco

fp_df, card_df, scco_df = load_data()

# 병합
merge_keys = ["PROVINCE_CODE", "CITY_CODE", "DISTRICT_CODE"]
df = pd.merge(fp_df, scco_df[merge_keys + ["DISTRICT_KOR_NAME"]], on=merge_keys, how="left")

# 파생변수
df["전체인구"] = df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"] + df["VISITING_POPULATION"]
df["소비활력지수"] = df["VISITING_POPULATION"] / df["전체인구"].replace(0, np.nan)
df["유입지수"] = df["VISITING_POPULATION"] / (df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"]).replace(0, np.nan)

# FEEL_IDX 계산
X = df[["RESIDENTIAL_POPULATION", "WORKING_POPULATION", "VISITING_POPULATION", "소비활력지수", "유입지수"]].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=1)
pc1 = pca.fit_transform(X_scaled).ravel()
pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-9)

df.loc[X.index, "FEEL_IDX"] = pc1_norm

# 샘플링
data = df.sample(frac=0.01, random_state=42)

# UI
st.title("서울시 감성 지수 매출분석")

top_districts = data["DISTRICT_KOR_NAME"].value_counts().head(10).index.tolist()

with st.sidebar:
    st.markdown("## 🔎 필터")
    districts = st.multiselect("행정동 (상위 10개)", options=top_districts, default=top_districts)
    age_groups = st.multiselect("연령대", sorted(data["AGE_GROUP"].dropna().unique()), default=sorted(data["AGE_GROUP"].dropna().unique()))
    gender = st.multiselect("성별", ["M", "F"], default=["M", "F"])

districts_mask = data["DISTRICT_KOR_NAME"].isin(districts)
age_groups_mask = data["AGE_GROUP"].isin(age_groups)
gender_mask = data["GENDER"].isin(gender)
mask = districts_mask & age_groups_mask & gender_mask
view = data.loc[mask]

# 요약 지표
st.subheader("요약 지표")
c1, c2, c3 = st.columns(3)

if not view.empty:
    c1.metric("평균 FEEL_IDX", f"{view['FEEL_IDX'].mean():.2f}")
    c2.metric("평균 소비활력지수", f"{view['소비활력지수'].mean():.2f}")
    c3.metric("평균 유입지수", f"{view['유입지수'].mean():.2f}")
else:
    c1.metric("평균 FEEL_IDX", "-")
    c2.metric("평균 소비활력지수", "-")
    c3.metric("평균 유입지수", "-")

# 탭
tab1, tab2, tab3 = st.tabs(["지수 상위 지역", "성별·연령 분석", "산점도"])

with tab1:
    st.subheader("지수 상위 지역 Top 10")
    if not view.empty:
        top = view.groupby("DISTRICT_KOR_NAME")["FEEL_IDX"].mean().sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top.values, y=top.index, palette="rocket", ax=ax)
        st.pyplot(fig)
    else:
        st.info("선택된 데이터가 없습니다.")

with tab2:
    st.subheader("성별 · 연령대별 FEEL_IDX")
    if not view.empty:
        grp = view.groupby(["AGE_GROUP", "GENDER"])["FEEL_IDX"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=grp, x="AGE_GROUP", y="FEEL_IDX", hue="GENDER", palette={"M": "skyblue", "F": "lightpink"}, ax=ax)
        st.pyplot(fig)
    else:
        st.info("선택된 데이터가 없습니다.")

with tab3:
    st.subheader("산점도 분석")
    if not view.empty:
        x_axis = st.selectbox("X축 변수", ["소비활력지수", "유입지수"])
        y_axis = st.selectbox("Y축 변수", ["FEEL_IDX"])
        if all(col in view.columns for col in [x_axis, y_axis]):
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=view, x=x_axis, y=y_axis, hue="FEEL_IDX", palette="viridis", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("선택한 컬럼이 데이터에 없습니다.")
    else:
        st.info("선택된 데이터가 없습니다.")

st.divider()
st.caption("© 2025 데이터 출처: snowflake park")


파생변수가 모두 생성이 안되는거 같은데, 
엔터전체방문자
수
방문자수 합계
엔터활동밀도 인구 1인당 엔터매출
방문자 1인당 
엔터매출
유입지수
방문자 1명당 발생한 엔터매출
방문자수가 주거자수 + 근로자수 대비 
얼마나 많은지
엔터매출비율 전체 매출 중 엔터매출 비중
FOOD_COUNT + COFFEE_COUNT + BEAUTY_COUNT + ENTERTAINMENT_COUNT + 
SPORTS_CULTURE_LEISURE_COUNT + TRAVEL_COUNT + CLOTHING_ACCESSORIES_COUNT
엔터전체매출 ÷ 전체인구
엔터전체매출 ÷ 엔터전체방문자수
방문자수 ÷ (주거자수 + 근로자수)
엔터전체매출 ÷ TOTAL_SALES
엔터방문자비율 전체 방문자 중 엔터 관련 방문자 비율 엔터전체방문자수 ÷ TOTAL_COUNT
엔터매출밀도 방문자 1명당 엔터매출
엔터전체매출 ÷ 엔터전체방문자수


이 7개를 생성해봐 다시 코드를 작성해봐
ChatGPT의 말:
아래는 요청하신 7개 파생변수를 생성하는 코드입니다. fp_card_scco_asset DataFrame을 기준으로 작성되어 있습니다:

import numpy as np

# 엔터전체방문자수
enter_count_columns = [
    'FOOD_COUNT', 'COFFEE_COUNT', 'BEAUTY_COUNT',
    'ENTERTAINMENT_COUNT', 'SPORTS_CULTURE_LEISURE_COUNT',
    'TRAVEL_COUNT', 'CLOTHING_ACCESSORIES_COUNT'
]
fp_card_scco_asset['엔터전체방문자수'] = fp_card_scco_asset[enter_count_columns].sum(axis=1)

# 엔터활동밀도 = 엔터전체매출 ÷ 전체인구
fp_card_scco_asset['엔터활동밀도'] = fp_card_scco_asset['엔터전체매출'] / fp_card_scco_asset['전체인구'].replace(0, np.nan)

# 방문자 1인당 엔터매출 = 엔터전체매출 ÷ 엔터전체방문자수
fp_card_scco_asset['방문자1인당엔터매출'] = fp_card_scco_asset['엔터전체매출'] / fp_card_scco_asset['엔터전체방문자수'].replace(0, np.nan)

# 유입지수 = 방문자수 ÷ (주거자수 + 근로자수)
fp_card_scco_asset['유입지수'] = fp_card_scco_asset['VISITING_POPULATION'] / (
    fp_card_scco_asset['RESIDENTIAL_POPULATION'] + fp_card_scco_asset['WORKING_POPULATION']
).replace(0, np.nan)

# 엔터매출비율 = 엔터전체매출 ÷ TOTAL_SALES
fp_card_scco_asset['엔터매출비율'] = fp_card_scco_asset['엔터전체매출'] / fp_card_scco_asset['TOTAL_SALES'].replace(0, np.nan)

# 엔터방문자비율 = 엔터전체방문자수 ÷ TOTAL_COUNT
fp_card_scco_asset['엔터방문자비율'] = fp_card_scco_asset['엔터전체방문자수'] / fp_card_scco_asset['TOTAL_COUNT'].replace(0, np.nan)

# 엔터매출밀도 = 엔터전체매출 ÷ 엔터전체방문자수 (중복 정의 시 갱신)
fp_card_scco_asset['엔터매출밀도'] = fp_card_scco_asset['엔터전체매출'] / fp_card_scco_asset['엔터전체방문자수'].replace(0, np.nan)
이 코드는 결측값 방지를 위해 replace(0, np.nan) 처리를 포함합니다.
추가로 TOTAL_SALES, TOTAL_COUNT 등의 컬럼이 있어야 하며, 누락 시 사전에 확인이 필요합니다.

이제 어떤 파생변수를 시각화하거나 분석에 사용할까요?

나의 말:
전체 코드를 작성해봐
ChatGPT의 말:
아래는 요청하신 7개 파생변수를 포함하여 fp_card_scco_asset 기준으로 전체 계산을 수행하는 코드입니다:

import numpy as np

# 엔터 관련 방문자수 컬럼 합계
enter_count_columns = [
    'FOOD_COUNT', 'COFFEE_COUNT', 'BEAUTY_COUNT',
    'ENTERTAINMENT_COUNT', 'SPORTS_CULTURE_LEISURE_COUNT',
    'TRAVEL_COUNT', 'CLOTHING_ACCESSORIES_COUNT'
]

# 전체 인구 = 주거 + 근무 + 방문 인구
fp_card_scco_asset['전체인구'] = (
    fp_card_scco_asset['RESIDENTIAL_POPULATION'] +
    fp_card_scco_asset['WORKING_POPULATION'] +
    fp_card_scco_asset['VISITING_POPULATION']
)

# 엔터전체매출 = 관련 업종 매출 합계
fp_card_scco_asset['엔터전체매출'] = (
    fp_card_scco_asset['FOOD_SALES'] +
    fp_card_scco_asset['COFFEE_SALES'] +
    fp_card_scco_asset['BEAUTY_SALES'] +
    fp_card_scco_asset['ENTERTAINMENT_SALES'] +
    fp_card_scco_asset['SPORTS_CULTURE_LEISURE_SALES'] +
    fp_card_scco_asset['TRAVEL_SALES'] +
    fp_card_scco_asset['CLOTHING_ACCESSORIES_SALES']
)

# 엔터전체방문자수 = 관련 업종 방문자수 합계
fp_card_scco_asset['엔터전체방문자수'] = fp_card_scco_asset[enter_count_columns].sum(axis=1)

# 엔터활동밀도 = 엔터전체매출 ÷ 전체인구
fp_card_scco_asset['엔터활동밀도'] = fp_card_scco_asset['엔터전체매출'] / fp_card_scco_asset['전체인구'].replace(0, np.nan)

# 방문자 1인당 엔터매출 = 엔터전체매출 ÷ 엔터전체방문자수
fp_card_scco_asset['방문자1인당엔터매출'] = fp_card_scco_asset['엔터전체매출'] / fp_card_scco_asset['엔터전체방문자수'].replace(0, np.nan)

# 유입지수 = 방문자수 ÷ (주거자수 + 근로자수)
fp_card_scco_asset['유입지수'] = fp_card_scco_asset['VISITING_POPULATION'] / (
    fp_card_scco_asset['RESIDENTIAL_POPULATION'] + fp_card_scco_asset['WORKING_POPULATION']
).replace(0, np.nan)

# 엔터매출비율 = 엔터전체매출 ÷ TOTAL_SALES
fp_card_scco_asset['엔터매출비율'] = fp_card_scco_asset['엔터전체매출'] / fp_card_scco_asset['TOTAL_SALES'].replace(0, np.nan)

# 엔터방문자비율 = 엔터전체방문자수 ÷ TOTAL_COUNT
fp_card_scco_asset['엔터방문자비율'] = fp_card_scco_asset['엔터전체방문자수'] / fp_card_scco_asset['TOTAL_COUNT'].replace(0, np.nan)

# 엔터매출밀도 = 엔터전체매출 ÷ 엔터전체방문자수
fp_card_scco_asset['엔터매출밀도'] = fp_card_scco_asset['엔터전체매출'] / fp_card_scco_asset['엔터전체방문자수'].replace(0, np.nan)
이 코드는 fp_card_scco_asset이 이미 병합 및 전처리된 상태라는 가정 하에 동작합니다.
필요한 컬럼(*_SALES, *_COUNT, TOTAL_SALES, TOTAL_COUNT 등)이 존재하는지 꼭 확인해 주세요.

추가로 이 변수들을 활용한 시각화도 같이 만들까요?

나의 말:
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 기본 세팅
st.set_page_config(page_title="서울시 감성 지수 대시보드", layout="wide")
sns.set_style("whitegrid")
            
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
import streamlit as st

# ── 1. NanumGothic 폰트 로드 (리포지토리 fonts 폴더)
font_path = os.path.join(os.path.dirname(__file__), "fonts", "NanumGothic.TTF")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    nanum_font_name = fm.FontProperties(fname=font_path).get_name()
    matplotlib.rcParams['font.family'] = nanum_font_name
    matplotlib.rcParams['axes.unicode_minus'] = False
else:
    st.warning("❌ NanumGothic.TTF 파일이 fonts 폴더에 없습니다.")

# ── 2. Streamlit UI CSS로도 강제 적용
st.markdown(f"""
<style>
@font-face {{
    font-family: 'NanumGothic';
    src: url('fonts/NanumGothic.TTF') format('truetype');
}}
html, body, [class*="css"] {{
    font-family: 'NanumGothic', 'sans-serif';
}}
</style>
""", unsafe_allow_html=True)
            
# 데이터 로드
DATA_DIR = Path(__file__).parent / "data"

@st.cache_data(show_spinner=False)
def load_data():
    fp = pd.read_csv(DATA_DIR / "floating population.csv")
    card = pd.read_csv(DATA_DIR / "Shinhan card sales.csv")
    scco = pd.read_csv(DATA_DIR / "scco.csv")
    return fp, card, scco

fp_df, card_df, scco_df = load_data()

# 병합
merge_keys = ["PROVINCE_CODE", "CITY_CODE", "DISTRICT_CODE"]
df = pd.merge(fp_df, scco_df[merge_keys + ["DISTRICT_KOR_NAME"]], on=merge_keys, how="left")

# 파생변수
df["전체인구"] = df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"] + df["VISITING_POPULATION"]
df["소비활력지수"] = df["VISITING_POPULATION"] / df["전체인구"].replace(0, np.nan)
df["유입지수"] = df["VISITING_POPULATION"] / (df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"]).replace(0, np.nan)

# FEEL_IDX 계산
X = df[["RESIDENTIAL_POPULATION", "WORKING_POPULATION", "VISITING_POPULATION", "소비활력지수", "유입지수"]].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=1)
pc1 = pca.fit_transform(X_scaled).ravel()
pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-9)

df.loc[X.index, "FEEL_IDX"] = pc1_norm

# 샘플링
data = df.sample(frac=0.01, random_state=42)

# UI
st.title("서울시 감성 지수 매출분석")

top_districts = data["DISTRICT_KOR_NAME"].value_counts().head(10).index.tolist()

with st.sidebar:
    st.markdown("## 🔎 필터")
    districts = st.multiselect("행정동 (상위 10개)", options=top_districts, default=top_districts)
    age_groups = st.multiselect("연령대", sorted(data["AGE_GROUP"].dropna().unique()), default=sorted(data["AGE_GROUP"].dropna().unique()))
    gender = st.multiselect("성별", ["M", "F"], default=["M", "F"])

districts_mask = data["DISTRICT_KOR_NAME"].isin(districts)
age_groups_mask = data["AGE_GROUP"].isin(age_groups)
gender_mask = data["GENDER"].isin(gender)
mask = districts_mask & age_groups_mask & gender_mask
view = data.loc[mask]

# 요약 지표
st.subheader("요약 지표")
c1, c2, c3 = st.columns(3)

if not view.empty:
    c1.metric("평균 FEEL_IDX", f"{view['FEEL_IDX'].mean():.2f}")
    c2.metric("평균 소비활력지수", f"{view['소비활력지수'].mean():.2f}")
    c3.metric("평균 유입지수", f"{view['유입지수'].mean():.2f}")
else:
    c1.metric("평균 FEEL_IDX", "-")
    c2.metric("평균 소비활력지수", "-")
    c3.metric("평균 유입지수", "-")

# 탭
tab1, tab2, tab3 = st.tabs(["지수 상위 지역", "성별·연령 분석", "산점도"])

with tab1:
    st.subheader("지수 상위 지역 Top 10")
    if not view.empty:
        top = view.groupby("DISTRICT_KOR_NAME")["FEEL_IDX"].mean().sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top.values, y=top.index, palette="rocket", ax=ax)
        st.pyplot(fig)
    else:
        st.info("선택된 데이터가 없습니다.")

with tab2:
    st.subheader("성별 · 연령대별 FEEL_IDX")
    if not view.empty:
        grp = view.groupby(["AGE_GROUP", "GENDER"])["FEEL_IDX"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=grp, x="AGE_GROUP", y="FEEL_IDX", hue="GENDER", palette={"M": "skyblue", "F": "lightpink"}, ax=ax)
        st.pyplot(fig)
    else:
        st.info("선택된 데이터가 없습니다.")

with tab3:
    st.subheader("산점도 분석")
    if not view.empty:
        x_axis = st.selectbox("X축 변수", ["소비활력지수", "유입지수"])
        y_axis = st.selectbox("Y축 변수", ["FEEL_IDX"])
        if all(col in view.columns for col in [x_axis, y_axis]):
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=view, x=x_axis, y=y_axis, hue="FEEL_IDX", palette="viridis", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("선택한 컬럼이 데이터에 없습니다.")
    else:
        st.info("선택된 데이터가 없습니다.")

st.divider()
st.caption("© 2025 데이터 출처: snowflake park")


이거를 알맞게 수정해서, 다시 수정된 전체코드작성
ChatGPT의 말:
전체 Streamlit 코드에 요청하신 파생변수를 반영해 수정 완료했습니다.
FEEL_IDX, 유입지수, 방문자당 엔터매출 등 모든 파생 지표를 포함하고 필터링 및 시각화 UI도 구성되어 있습니다.

추가적으로 어떤 시각화나 기능을 넣고 싶으세요?

나의 말:
KeyError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/home/adminuser/venv/lib/python3.12/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 534, in _run_script
    exec(code, module.__dict__)
File "/mount/src/snowflake-x-streamlit-hackathon/streamlit_app.py", line 68, in <module>
    df["전체인구"] = df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"] + df["VISITING_POPULATION"]
                     ~~^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/pandas/core/frame.py", line 4102, in __getitem__
    indexer = self.columns.get_loc(key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.12/site-packages/pandas/core/indexes/base.py", line 3812, in get_loc
    raise KeyError(key) from err
ChatGPT의 말:
오류 메시지로 보아 df["RESIDENTIAL_POPULATION"] 컬럼이 존재하지 않아 KeyError가 발생했습니다. 이는 fp_df, card_df, scco_df 병합 후 df에 해당 컬럼이 포함되지 않았기 때문입니다.

🔍 원인 점검 포인트
fp_df에 RESIDENTIAL_POPULATION, WORKING_POPULATION, VISITING_POPULATION 컬럼이 존재하는지 확인

card_df와 병합 시 해당 컬럼들이 손실되지 않았는지 확인

병합 키에 STANDARD_YEAR_MONTH, AGE_GROUP, GENDER 등이 없으면 inner join에서 행이 모두 사라질 수도 있음

✅ 해결 방법
아래 코드처럼 병합 순서를 조정하거나 join 키를 줄이기:

# 먼저 fp와 scco만 병합해 인구 관련 컬럼 보존
df = pd.merge(fp_df, scco_df[merge_keys + ["DISTRICT_KOR_NAME"]], on=merge_keys, how="left")

# 이후 카드 데이터 병합 (연령/성별/날짜 조건 없이)
df = pd.merge(df, card_df, on=merge_keys, how="left")
또는, 병합 후 컬럼 존재 여부 확인:

print("RESIDENTIAL_POPULATION" in df.columns)
print(df.columns.tolist())
🛠 빠른 수정 제안
해당 줄을 아래처럼 수정해 컬럼 존재 시에만 파생변수 생성하도록 방어 코드를 추가할 수 있어요:

if all(col in df.columns for col in ["RESIDENTIAL_POPULATION", "WORKING_POPULATION", "VISITING_POPULATION"]):
    df["전체인구"] = df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"] + df["VISITING_POPULATION"]
else:
    st.error("❌ 인구 관련 컬럼이 누락되어 있습니다. 병합 순서 또는 데이터셋을 확인하세요.")
원본 fp_df.head() 구조 보여주면 더 정확히 도와줄 수 있어. 샘플 보여줄래?

나의 말:
다시 전체코드를 작성
ChatGPT의 말:

대신 채팅에서 답합니다
전체 코드를 수정해 `










ChatGPT는 실수를 할 수 있습니다. 중요한 정보는 재차 확인하세요.
?


Seoul Feel Index App
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
# 병합
merge_keys = ["PROVINCE_CODE", "CITY_CODE", "DISTRICT_CODE"]
df = pd.merge(fp_df, scco_df[merge_keys + ["DISTRICT_KOR_NAME"]], on=merge_keys, how="left")
df = pd.merge(df, card_df, on=merge_keys + ["AGE_GROUP", "GENDER", "STANDARD_YEAR_MONTH"], how="left")

# 필수 컬럼 확인
required_cols = ["RESIDENTIAL_POPULATION", "WORKING_POPULATION", "VISITING_POPULATION"]
if not all(col in df.columns for col in required_cols):
    st.stop("❌ 인구 관련 컬럼이 누락되어 분석을 진행할 수 없습니다.")

# 파생변수 생성
enter_count_columns = [
    'FOOD_COUNT', 'COFFEE_COUNT', 'BEAUTY_COUNT', 'ENTERTAINMENT_COUNT',
    'SPORTS_CULTURE_LEISURE_COUNT', 'TRAVEL_COUNT', 'CLOTHING_ACCESSORIES_COUNT'
]

sales_columns = [
    'FOOD_SALES', 'COFFEE_SALES', 'BEAUTY_SALES', 'ENTERTAINMENT_SALES',
    'SPORTS_CULTURE_LEISURE_SALES', 'TRAVEL_SALES', 'CLOTHING_ACCESSORIES_SALES'
]

df["전체인구"] = df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"] + df["VISITING_POPULATION"]
df["엔터전체매출"] = df[sales_columns].sum(axis=1)
df["엔터전체방문자수"] = df[enter_count_columns].sum(axis=1)
df["엔터활동밀도"] = df["엔터전체매출"] / df["전체인구"].replace(0, np.nan)
df["방문자1인당엔터매출"] = df["엔터전체매출"] / df["엔터전체방문자수"].replace(0, np.nan)
df["유입지수"] = df["VISITING_POPULATION"] / (df["RESIDENTIAL_POPULATION"] + df["WORKING_POPULATION"]).replace(0, np.nan)
df["엔터매출비율"] = df["엔터전체매출"] / df["TOTAL_SALES"].replace(0, np.nan)
df["엔터방문자비율"] = df["엔터전체방문자수"] / df["TOTAL_COUNT"].replace(0, np.nan)
df["엔터매출밀도"] = df["엔터전체매출"] / df["엔터전체방문자수"].replace(0, np.nan)
df["소비활력지수"] = df["VISITING_POPULATION"] / df["전체인구"].replace(0, np.nan)

# FEEL_IDX 계산
pca_vars = ["엔터전체매출", "유입지수", "엔터매출비율", "엔터전체방문자수", "엔터방문자비율", "엔터활동밀도", "엔터매출밀도"]
X = df[pca_vars].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=1)
pc1 = pca.fit_transform(X_scaled).ravel()
pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-9)
df.loc[X.index, "FEEL_IDX"] = pc1_norm

# 샘플링
data = df.sample(frac=0.01, random_state=42)

# UI
st.title("서울시 감성 지수 매출분석")

top_districts = data["DISTRICT_KOR_NAME"].value_counts().head(10).index.tolist()

with st.sidebar:
    st.markdown("## 🔎 필터")
    districts = st.multiselect("행정동 (상위 10개)", options=top_districts, default=top_districts)
    age_groups = st.multiselect("연령대", sorted(data["AGE_GROUP"].dropna().unique()), default=sorted(data["AGE_GROUP"].dropna().unique()))
    gender = st.multiselect("성별", ["M", "F"], default=["M", "F"])

mask = (
    data["DISTRICT_KOR_NAME"].isin(districts) &
    data["AGE_GROUP"].isin(age_groups) &
    data["GENDER"].isin(gender)
)
view = data.loc[mask]

# 요약 지표
st.subheader("요약 지표")
c1, c2, c3 = st.columns(3)

if not view.empty:
    c1.metric("평균 FEEL_IDX", f"{view['FEEL_IDX'].mean():.2f}")
    c2.metric("평균 소비활력지수", f"{view['소비활력지수'].mean():.2f}")
    c3.metric("평균 유입지수", f"{view['유입지수'].mean():.2f}")
else:
    c1.metric("평균 FEEL_IDX", "-")
    c2.metric("평균 소비활력지수", "-")
    c3.metric("평균 유입지수", "-")

# 탭 시각화
with st.tabs(["지수 상위 지역", "성별·연령 분석", "산점도"]) as (tab1, tab2, tab3):
    with tab1:
        st.subheader("지수 상위 지역 Top 10")
        if not view.empty:
            top = view.groupby("DISTRICT_KOR_NAME")["FEEL_IDX"].mean().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=top.values, y=top.index, palette="rocket", ax=ax)
            st.pyplot(fig)
        else:
            st.info("선택된 데이터가 없습니다.")

    with tab2:
        st.subheader("성별 · 연령대별 FEEL_IDX")
        if not view.empty:
            grp = view.groupby(["AGE_GROUP", "GENDER"])["FEEL_IDX"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=grp, x="AGE_GROUP", y="FEEL_IDX", hue="GENDER", palette={"M": "skyblue", "F": "lightpink"}, ax=ax)
            st.pyplot(fig)
        else:
            st.info("선택된 데이터가 없습니다.")

    with tab3:
        st.subheader("산점도 분석")
        if not view.empty:
            x_axis = st.selectbox("X축 변수", ["소비활력지수", "유입지수"])
            y_axis = st.selectbox("Y축 변수", ["FEEL_IDX"])
            if all(col in view.columns for col in [x_axis, y_axis]):
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=view, x=x_axis, y=y_axis, hue="FEEL_IDX", palette="viridis", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("선택한 컬럼이 데이터에 없습니다.")
        else:
            st.info("선택된 데이터가 없습니다.")

st.divider()
st.caption("© 2025 데이터 출처: snowflake park")

