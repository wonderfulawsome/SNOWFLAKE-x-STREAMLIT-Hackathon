# 📊 Snowflake + Streamlit: 서울 엔터테인먼트 소비 분석

## 프로젝트 소개
- **목표**: 서울시 엔터테인먼트 소비활동 분석 및 감성지수(FEEL_IDX) 개발
- **데이터**: Snowflake Database 연결하여 다양한 테이블 통합
- **사용 기술**: Python, Pandas, Streamlit, Matplotlib, Seaborn, GeoPandas, Scikit-learn

---

## 데이터 연결 및 수집
- **Snowflake** 연결 (`snowflake-connector-python` 사용)
- 수집 테이블
  - 백화점 방문 데이터 (`department_store_df`)
  - 서울 기상 데이터 (`seoul_weather_df`)
  - 주거지/근무지 데이터 (`home_office_df`)
  - 유동 인구 데이터 (`floating_population_df`)
  - 카드 매출 데이터 (`card_sales_df`)
  - 소득 데이터 (`asset_income_df`)
  - 행정동 경계 데이터 (`scco_df`)

---

## 데이터 전처리
- 공통 컬럼으로 테이블 병합
- 중복 컬럼 제거 및 컬럼명 통일
- 주요 파생 컬럼 생성
  - `전체인구 = 거주 + 근무 + 방문 인구`
  - `엔터전체매출 = 감성 관련 업종 매출 합산`
  - `소비활력지수 = 엔터전체매출 / 전체인구`

---

## 주요 파생지표
| 지표명 | 설명 |
|:------|:----|
| 유입지수 | 방문자수 / (거주자수 + 근로자수) |
| 엔터활동밀도 | 엔터매출 / 전체인구 |
| 엔터매출비율 | 엔터매출 / 총매출 |
| 엔터매출밀도 | 엔터매출 / 엔터방문자수 |
| FEEL_IDX | 종합 감성 소비 지수 (PCA 기반) |

---

## 데이터 시각화
- **Bar Chart**
  - 지역별 방문자수, 방문자 1인당 엔터매출
  - 연령/성별별 매출 비교, 방문횟수 비교
- **Geo Map**
  - 행정동별 방문자수, 1인당 매출 지도 시각화
- **Scatter Plot**
  - 감성지수(FEEL_IDX) vs 매출 변수 산점도
- **Heatmap**
  - 주요 변수 간 상관관계 분석

---

## 감성지수 (FEEL_IDX) 개발
- 7개 주요 변수를 **표준화(StandardScaler)** 후
- **PCA** 수행 → 첫 번째 주성분을 감성지수로 설정
- 변수별 가중치(loading) 추출 및 시각화
- 감성지수 상관분석: 어떤 매출 항목이 감성지수에 영향을 미치는지 분석

---

## 설치 방법
```bash
pip install snowflake-connector-python pandas matplotlib seaborn geopandas scikit-learn
```

---

## 실행 방법
1. Snowflake 연결정보(user, password, account) 입력
2. Python 코드 실행
3. 결과 그래프 및 지표 확인

---

## 주요 인사이트
- 남성 방문자의 1회당 매출 기여도가 여성보다 높음
- 연령이 높을수록 방문보다는 구매 중심 소비
- 특정 지역은 유입지수가 매우 높음 (방문 특화)
- 엔터활동밀도와 전체매출이 감성지수에 강한 영향

---

# 🏙️ 서울시 감성 지수 대시보드 - Streamlit App

## 프로젝트 소개
- **목표**: 서울시 인스타 감성 소비 활동을 분석하고 시각화하는 대시보드 개발
- **데이터 소스**: Snowflake Marketplace 데이터 (서울시 유동인구, 카드 매출, 자산/소득 정보 등)
- **사용 기술**: Streamlit, Python, Snowflake Connector, Pandas, Seaborn, Matplotlib, Scikit-learn

---

## 주요 기능
- **Snowflake 연결** 및 데이터 조회
- **전처리**: 공통 컬럼 병합, 파생변수 생성 (소비활력지수, 유입지수 등)
- **FEEL_IDX 감성 지수**: PCA 분석 기반 종합 소비활동 지수 계산
- **Streamlit 대시보드**:
  - 지역별 소비활력지수 상위 순위
  - 성별·연령대별 평균 매출 분석
  - 변수 간 산점도 시각화
- **한글 폰트 지원**: matplotlib + Streamlit CSS 설정

---

## 설치 방법
```bash
pip install streamlit snowflake-connector-python pandas numpy matplotlib seaborn scikit-learn
```

---

## 실행 방법
1. `.streamlit/secrets.toml` 파일에 Snowflake 연결 정보 입력
```toml
[user]
user = "YOUR_USER"
password = "YOUR_PASSWORD"
account = "YOUR_ACCOUNT"
```

2. Streamlit 앱 실행
```bash
streamlit run app.py
```

---

## 주요 코드 흐름
- **Snowflake 연결**
  - `snowflake.connector.connect()`로 데이터 조회
- **데이터 전처리**
  - 4개 테이블 병합 후 파생 컬럼 생성
  - 표준화(StandardScaler) + PCA 분석하여 감성지수 도출
- **대시보드 구성**
  - Sidebar: 지역, 연령대, 성별 필터
  - Main:
    - 요약 지표 (FEEL_IDX, 소비활력지수, 유입지수)
    - 지역별 소비활력지수 Top20 바차트
    - 성별·연령대별 평균 매출
    - 자유로운 변수 선택 산점도

---

## 주요 파생변수
| 변수명 | 설명 |
|:---|:---|
| 전체인구 | 거주 + 근무 + 방문 인구 합 |
| 엔터전체매출 | FOOD, COFFEE, BEAUTY, ENTERTAINMENT 등 합산 |
| 소비활력지수 | 엔터전체매출 / 전체인구 |
| 유입지수 | 방문인구 / (거주인구 + 근로인구) |
| 엔터매출비율 | 엔터전체매출 / 전체매출 |
| FEEL_IDX | PCA 기반 종합 감성 소비 지수 |

---

## 주의사항
- 최초 실행 시 Streamlit 캐시로 인해 약간의 로딩이 필요할 수 있음
- Snowflake 쿼리 결과에 따라 데이터양이 많으면 샘플링 적용됨 (`1%`)


---

