import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─────────────────────────────
# 0. 기본 세팅
# ─────────────────────────────
st.set_page_config(page_title="서울시 감성 지수 대시보드", layout="wide")
sns.set_style("whitegrid")

# ──── 한글 폰트 설정 ────
import matplotlib
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
st.markdown(
    """
    <style>
      * { font-family: "Malgun Gothic", sans-serif !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────
DATA_DIR = Path(__file__).parent / "data"

@st.cache_data(show_spinner=False)
def load_data():
    fp   = pd.read_csv(DATA_DIR / "floating population.csv")
    card = pd.read_csv(DATA_DIR / "Shinhan card sales.csv")
    scco = pd.read_csv(DATA_DIR / "scco.csv")
    return fp, card, scco

try:
    fp_df, card_df, scco_df = load_data()
except FileNotFoundError as e:
    st.error(f"데이터 파일을 찾을 수 없습니다: {e}")
    st.stop()

# ─────────────────────────────
# 2. 병합 & 매핑
# ─────────────────────────────
# 병합 키
merge_keys = [
    "PROVINCE_CODE", "CITY_CODE", "DISTRICT_CODE",
    "STANDARD_YEAR_MONTH", "WEEKDAY_WEEKEND",
    "GENDER", "AGE_GROUP", "TIME_SLOT"
]
df = pd.merge(fp_df, card_df, on=merge_keys, how="inner", validate="m:m")

# 행정동 한글 이름 매핑
if "DISTRICT_KOR_NAME" in scco_df.columns:
    name_map = scco_df.set_index("DISTRICT_CODE")["DISTRICT_KOR_NAME"].to_dict()
    df["DISTRICT_KOR_NAME"] = df["DISTRICT_CODE"].map(name_map)
else:
    df["DISTRICT_KOR_NAME"] = df["DISTRICT_CODE"].astype(str)

# ─────────────────────────────
# 3. 파생변수 & FEEL_IDX
# ─────────────────────────────
# 숫자형 피처만 골라서 표준화 → PCA(1) → 0–1 정규화
group_cols = merge_keys + ["DISTRICT_KOR_NAME"]
numeric_cols = (
    df.select_dtypes(include="number")
      .columns
      .difference(["PROVINCE_CODE","CITY_CODE","DISTRICT_CODE"])
)
features = [c for c in numeric_cols if c not in group_cols]

# 결측치 제거 후 스케일링 & PCA
X = df[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=1)
pc1 = pca.fit_transform(X_scaled).ravel()
# 0–1 정규화
pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-9)
df.loc[X.index, "FEEL_IDX"] = pc1_norm

# 샘플링
data = df.sample(frac=0.01, random_state=42)

# ─────────────────────────────
# 4. UI: 필터링
# ─────────────────────────────
st.title("서울시 인스타 감성 지수 대시보드")
with st.sidebar:
    sel_d = st.multiselect("행정동", sorted(data["DISTRICT_KOR_NAME"].dropna().unique()))
    sel_a = st.multiselect("연령대", sorted(data["AGE_GROUP"].unique()))
    sel_g = st.multiselect("성별", ["M","F"])

mask = (
    (data["DISTRICT_KOR_NAME"].isin(sel_d) if sel_d else True) &
    (data["AGE_GROUP"].isin(sel_a) if sel_a else True) &
    (data["GENDER"].isin(sel_g) if sel_g else True)
)
view = data[mask]

# ─────────────────────────────
# 5. 요약 지표
# ─────────────────────────────
st.subheader("요약 지표")
avg_idx = view["FEEL_IDX"].mean()
st.metric("평균 FEEL_IDX", f"{avg_idx:.2f}")

# ─────────────────────────────
# 6. 상위 지역 Bar 차트
# ─────────────────────────────
st.subheader("FEEL_IDX 상위 10개 행정동")
top10 = view.groupby("DISTRICT_KOR_NAME")["FEEL_IDX"].mean().nlargest(10)
fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(x=top10.values, y=top10.index, palette="rocket", ax=ax)
ax.set_xlabel("FEEL_IDX")
st.pyplot(fig)

# ─────────────────────────────
# 7. 성별·연령별 분석
# ─────────────────────────────
st.subheader("연령·성별별 FEEL_IDX")
grp = view.groupby(["AGE_GROUP","GENDER"])["FEEL_IDX"].mean().reset_index()
fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(data=grp, x="AGE_GROUP", y="FEEL_IDX", hue="GENDER", ax=ax)
st.pyplot(fig)

# ─────────────────────────────
# 8. 산점도
# ─────────────────────────────
st.subheader("FEEL_IDX vs 기타 변수")
scatter_vars = ["FEEL_IDX"] + features[:3]  # 첫 3개 변수를 예시로
x_var = st.selectbox("X축", scatter_vars[1:], index=0)
fig, ax = plt.subplots(figsize=(6,4))
sns.scatterplot(data=view, x=x_var, y="FEEL_IDX", alpha=0.6, ax=ax)
st.pyplot(fig)

st.sidebar.caption("© 2025 Park Sungsoo")
