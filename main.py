    df["연도"] = df["날짜"].dt.year

    # 2025년까지의 자료만 사용하며, 연도별 유효 관측일 수가 300일 이상인 해만 남깁니다.
    df = df[df["연도"] <= CUTOFF_YEAR]
    yearly = (
        df.groupby("연도", as_index=False)
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
    )
    yearly = yearly[yearly["관측일수"] >= MIN_OBSERVATION_DAYS].sort_values("연도")
    return yearly.reset_index(drop=True)


try:
    yearly = load_yearly_data(DATA_URL)
except Exception as exc:
    st.error("기온 데이터를 불러오거나 처리하지 못했습니다.")
    st.exception(exc)
    st.stop()

if len(yearly) < 2:
    st.error("회귀 분석을 수행하기에 유효한 연도가 부족합니다.")
    st.stop()

years = yearly["연도"].to_numpy(dtype=float)
temps = yearly["연평균기온"].to_numpy(dtype=float)

# 최소제곱법으로 직선의 기울기와 절편을 구합니다.
slope, intercept = np.polyfit(years, temps, 1)
correlation = float(np.corrcoef(years, temps)[0, 1])
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
year_count = len(yearly)

selected_year = st.slider(
    "예측할 연도 선택",
    min_value=1900,
    max_value=2100,
    value=2050,
    step=1,
    help="1900년부터 2100년까지 선택할 수 있습니다. 학습 기간 바깥의 값은 외삽 예측입니다.",
)
predicted_temp = slope * selected_year + intercept

st.subheader(f"{selected_year}년 예상 연평균기온")
st.metric(label="선형회귀 예측값", value=f"{predicted_temp:.2f} °C")
if selected_year < start_year or selected_year > end_year:
    st.warning(
        f"{selected_year}년은 학습 기간({start_year}~{end_year}년) 바깥입니다. "
        "이 값은 관측 범위를 벗어난 외삽 결과이므로 불확실성이 큽니다."
    )

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("회귀에 사용한 연도 수", f"{year_count}년")
metric2.metric("시작 연도", f"{start_year}년")
metric3.metric("끝 연도", f"{end_year}년")
metric4.metric("상관계수 (r)", f"{correlation:.4f}")

st.subheader("연도별 평균기온과 회귀 직선")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker={"size": 7},
        customdata=yearly[["관측일수"]],
        hovertemplate=(
            "연도: %{x}년<br>연평균기온: %{y:.2f} °C"
            "<br>관측일수: %{customdata[0]}일<extra></extra>"
        ),
    )
)

line_years = np.linspace(start_year, end_year, 300)
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=slope * line_years + intercept,
        mode="lines",
        name="선형회귀 직선",
        line={"width": 3},
        hovertemplate="연도: %{x:.0f}년<br>회귀값: %{y:.2f} °C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예측",
        marker={"size": 13, "symbol": "diamond"},
        hovertemplate="%{x:.0f}년 예측: %{y:.2f} °C<extra></extra>",
    )
)
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend_title_text="범례",
    margin={"l": 20, "r": 20, "t": 25, "b": 20},
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"학습 자료: {start_year}~{end_year}년 중 유효 관측일이 "
    f"{MIN_OBSERVATION_DAYS}일 이상인 연도. "
    f"회귀식: 예상 기온 = {slope:.5f} × 연도 + ({intercept:.3f})."
)
st.caption(
    "주의: 이 직선은 과거 연도와 연평균기온 사이의 선형 관계를 요약한 교육용 모델입니다. "
    "실제 미래 기온을 확정적으로 예측하지 않으며, 장기 외삽은 특히 불확실합니다."
)

with st.expander("분석에 사용한 연도별 데이터 보기"):
    display_df = yearly.copy()
    display_df["연평균기온"] = display_df["연평균기온"].round(2)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
