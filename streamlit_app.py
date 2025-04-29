import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# 설정 --------------------------------------------------
st.set_page_config(page_title="Henry Hub Gas Dashboard", layout="wide")
sns.set_style("whitegrid")

# 폰트 --------------------------------------------------
def set_korean_font():
    local = Path(__file__).parent / "assets" / "NanumGothic.ttf"
    if local.exists():
        plt.rc("font", family=fm.FontProperties(fname=str(local)).get_name())
        return
    for cand in ["NanumGothic", "Noto Sans KR", "AppleGothic"]:
        if any(cand in fp for fp in fm.findSystemFonts()):
            plt.rc("font", family=cand)
            return
    plt.rc("font", family="DejaVu Sans")

set_korean_font()
plt.rcParams["axes.unicode_minus"] = False

TARGET = "Natural_Gas_US_Henry_Hub_Gas"
DATA_DIR = Path(__file__).parent / "data"

# 데이터 -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    train   = pd.read_csv(DATA_DIR / "department store .csv")
    test    = pd.read_csv(DATA_DIR / "floating population.csv")
    card    = pd.read_csv(DATA_DIR / "Shinhan card sales.csv")
    scco    = pd.read_csv(DATA_DIR / "scco.csv")
    return train, test, card, scco

try:
    train_df, test_df, card_df, scco_df = load_data()
except FileNotFoundError as e:
    st.error(f"데이터 파일을 찾을 수 없습니다: {e}")
    st.stop()

if TARGET not in train_df.columns:
    st.error(f"‘{TARGET}’ 컬럼이 없습니다.\n컬럼 목록: {train_df.columns.tolist()}")
    st.stop()

# 사이드바 ----------------------------------------------
section = st.sidebar.selectbox("메뉴", ["상관관계", "피처 중요도", "시계열 비교", "산점도"])

# 상관관계 ----------------------------------------------
if section == "상관관계":
    st.subheader("전체 변수 히트맵")
    corr = train_df.drop(columns=["date"]).corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": .5}, ax=ax)
    st.pyplot(fig)

    st.divider()
    st.subheader("Henry Hub vs 변수")
    target_corr = corr[TARGET].sort_values(ascending=False)
    st.dataframe(target_corr.to_frame("corr"))

    vars_kor = [
        "원유 지수", "가스 굴착기 수", "세계경기 지수", "가스 지수", "가스 수입량",
        "철강 YoY", "캐나다 가스 수입량", "PMI 중간재 물가", "기준금리", "제조업 경기예상"
    ]
    corrs = [0.8767, 0.8132, 0.7175, 0.7158, 0.7107,
             0.6722, 0.6632, 0.6500, 0.6013, 0.5865]
    df_bar = pd.DataFrame({"var": vars_kor, "corr": corrs}).sort_values("corr", ascending=False)
    df_bar["color"] = np.where(df_bar.corr > 0, "steelblue", "crimson")
    fig_bar = go.Figure(go.Bar(x=df_bar.var, y=df_bar.corr,
                               marker_color=df_bar.color,
                               text=df_bar.corr.round(3),
                               textposition="outside"))
    fig_bar.update_layout(height=500, margin=dict(t=40, b=120))
    st.plotly_chart(fig_bar, use_container_width=True)

# 피처 중요도 --------------------------------------------
elif section == "피처 중요도":
    st.subheader("XGBoost Feature Importance")
    X = train_df.drop(columns=["date", TARGET])
    y = train_df[TARGET]
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(random_state=42)
    model.fit(X_tr, y_tr)
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(imp.index[::-1], imp.values[::-1], color="steelblue")
    ax.set_xlabel("중요도")
    st.pyplot(fig)

# 시계열 비교 --------------------------------------------
elif section == "시계열 비교":
    st.subheader("표준화 시계열 비교")
    cols = [
        "Natural_Gas_Rotary_Rig_Count_USA",
        "Kilian_Global_Economy_Index_WORLD",
        "Natural_Gas_Imports_USA",
        "BCOMCL_INDX",
        "Crude_Steel_Accumulated_YoY",
        "PPI_Mining_Sector_USA",
        "DXY_INDX",
    ]
    missing = [c for c in cols if c not in train_df.columns]
    if missing:
        st.warning(f"다음 컬럼 없음: {missing}")
    else:
        scale_cols = cols + [TARGET]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(train_df[scale_cols])
        s_df = pd.DataFrame(scaled, columns=scale_cols)
        s_df["date"] = pd.to_datetime(train_df["date"])

        n_cols = 2
        n_rows = int(np.ceil(len(cols) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            axes[i].plot(s_df.date, s_df[col], label=col, color="steelblue")
            axes[i].plot(s_df.date, s_df[TARGET], label="Henry Hub", color="darkred")
            axes[i].legend(fontsize=8)
            axes[i].tick_params(axis="x", rotation=45, labelsize=7)

        for j in range(len(cols), len(axes)):
            fig.delaxes(axes[j])

        st.pyplot(fig)

# 산점도 --------------------------------------------------
else:
    st.subheader("변수별 산점도")
    cols = [
        "Natural_Gas_Rotary_Rig_Count_USA",
        "Kilian_Global_Economy_Index_WORLD",
        "BCOMCL_INDX",
        "Crude_Steel_Accumulated_YoY",
        "PPI_Mining_Sector_USA",
        "DXY_INDX",
    ]
    valid_cols = [c for c in cols if c in train_df.columns]
    n_cols = 2
    n_rows = int(np.ceil(len(valid_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(valid_cols):
        sns.regplot(data=train_df, x=col, y=TARGET,
                    ax=axes[i], fit_reg=False, scatter_kws={"alpha": .6})
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Henry Hub")

    for j in range(len(valid_cols), len(axes)):
        fig.delaxes(axes[j])

    st.pyplot(fig)

st.sidebar.caption("© 2025 Park Sungsoo")
