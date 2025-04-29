# ─────────────────────────────
# 4. UI 구성
# ─────────────────────────────
st.title("서울시 인스타 감성 지수 분석")

# 상위 등장 행정동 10개 뽑기
top_districts = (
    data["DISTRICT_KOR_NAME"].value_counts()
    .head(10)
    .index
    .tolist()
)

with st.sidebar:
    st.markdown("## 🔎 필터")

    default_districts = top_districts if len(top_districts) > 0 else data["DISTRICT_KOR_NAME"].dropna().unique().tolist()
    default_ages = data["AGE_GROUP"].dropna().unique().tolist()
    default_gender = ["M", "F"]

    districts = st.multiselect("행정동 (상위 10)", options=top_districts, default=top_districts)
    age_groups = st.multiselect("연령대", options=sorted(data["AGE_GROUP"].unique()), default=default_ages)
    gender = st.multiselect("성별", ["M", "F"], default=default_gender)

# 필터링
districts_mask = data["DISTRICT_KOR_NAME"].isin(districts) if districts else pd.Series([True] * len(data), index=data.index)
age_groups_mask = data["AGE_GROUP"].isin(age_groups) if age_groups else pd.Series([True] * len(data), index=data.index)
gender_mask = data["GENDER"].isin(gender) if gender else pd.Series([True] * len(data), index=data.index)

mask = districts_mask & age_groups_mask & gender_mask
view = data.loc[mask]

# ─────────────────────────────
# 5. 요약 지표
# ─────────────────────────────
st.subheader("요약 지표")
c1, c2, c3 = st.columns(3)

if not view.empty and all(col in view.columns for col in ["FEEL_IDX", "소비활력지수", "유입지수"]):
    c1.metric("평균 FEEL_IDX", f"{view['FEEL_IDX'].mean():.2f}")
    c2.metric("평균 소비활력지수", f"{view['소비활력지수'].mean():.2f}")
    c3.metric("평균 유입지수", f"{view['유입지수'].mean():.2f}")
else:
    c1.metric("평균 FEEL_IDX", "-")
    c2.metric("평균 소비활력지수", "-")
    c3.metric("평균 유입지수", "-")

# ─────────────────────────────
# 6. 탭 형태 분석 화면
# ─────────────────────────────
tab1, tab2, tab3 = st.tabs(["지수 상위 지역", "성별·연령 분석", "산점도"])

with tab1:
    st.subheader("지수 상위 지역 Top 20")
    if not view.empty:
        top = (
            view.groupby("DISTRICT_KOR_NAME")["FEEL_IDX"]
            .mean()
            .sort_values(ascending=False)
            .head(20)
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top.values, y=top.index, palette="rocket", ax=ax)
        ax.set_xlabel("평균 FEEL_IDX")
        ax.set_ylabel("행정동")
        st.pyplot(fig)
    else:
        st.info("선택된 데이터가 없습니다.")

with tab2:
    st.subheader("성별 · 연령대별 FEEL_IDX")
    if not view.empty:
        grp = (
            view.groupby(["AGE_GROUP", "GENDER"])["FEEL_IDX"]
            .mean()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=grp, x="AGE_GROUP", y="FEEL_IDX", hue="GENDER",
                    palette={"M": "skyblue", "F": "lightpink"}, ax=ax)
        st.pyplot(fig)
    else:
        st.info("선택된 데이터가 없습니다.")

with tab3:
    st.subheader("산점도 분석")
    if not view.empty:
        x_axis = st.selectbox("X축 변수", ["엔터전체매출", "소비활력지수", "유입지수", "엔터전체방문자수"])
        y_axis = st.selectbox("Y축 변수", ["FEEL_IDX", "엔터매출비율"])
        required_cols = [x_axis, y_axis, "FEEL_IDX"]
        if all(col in view.columns for col in required_cols):
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                data=view, x=x_axis, y=y_axis,
                hue="FEEL_IDX", palette="viridis", alpha=0.6, ax=ax
            )
            st.pyplot(fig)
        else:
            st.warning("선택한 컬럼이 데이터에 없습니다.")
    else:
        st.info("선택된 데이터가 없습니다.")
