import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
MAX_YEAR = 2025
MIN_DAYS = 300

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")
st.title("🌡️ 기온 예측기")
st.write("서울의 연평균기온을 산점도와 선형회귀 직선으로 살펴봅니다.")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required = ["날짜", "평균기온"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"필수 열이 없습니다: {', '.join(missing)}")

    df = df[required].copy()
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])

    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= MAX_YEAR]

    yearly = (
        df.groupby("연도", as_index=False)
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
    )
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS]
    return yearly.sort_values("연도").reset_index(drop=True)


try:
    yearly = load_data()
except Exception as exc:
    st.error("서울 기온 데이터를 불러오지 못했습니다.")
    st.exception(exc)
    st.stop()

if len(yearly) < 2:
    st.error("회귀 분석을 하기에 유효한 연도가 부족합니다.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x, y, 1)
correlation = float(np.corrcoef(x, y)[0, 1])

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
year_count = len(yearly)

selected_year = st.slider(
    "예측할 연도",
    min_value=1900,
    max_value=2100,
    value=2050,
    step=1,
)
prediction = slope * selected_year + intercept

st.subheader(f"{selected_year}년 예상 평균기온")
st.markdown(
    f"<div style='font-size:3rem;font-weight:700;line-height:1.2'>{prediction:.2f} °C</div>",
    unsafe_allow_html=True,
)

if selected_year < start_year or selected_year > end_year:
    st.warning(
        f"선택한 {selected_year}년은 학습 기간({start_year}~{end_year}년) 밖입니다. "
        "과거 자료 범위를 벗어난 외삽이므로 불확실성이 큽니다."
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("회귀에 사용한 해의 개수", f"{year_count}년")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
c4.metric("상관계수", f"{correlation:.4f}")

st.subheader("연도별 평균기온과 회귀 직선")
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=7),
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>평균기온: %{y:.2f} °C"
            "<br>관측일수: %{customdata[0]}일<extra></extra>"
        ),
    )
)

line_x = np.linspace(start_year, end_year, 300)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=slope * line_x + intercept,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3),
        hovertemplate="연도: %{x:.0f}년<br>회귀값: %{y:.2f} °C<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[prediction],
        mode="markers",
        name="선택한 연도의 예상값",
        marker=dict(size=13, symbol="diamond"),
        hovertemplate="%{x}년 예상 기온: %{y:.2f} °C<extra></extra>",
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend_title="범례",
    margin=dict(l=20, r=20, t=25, b=20),
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"회귀 학습 자료: {start_year}~{end_year}년. "
    f"2025년 이하 자료 중 유효 관측일이 {MIN_DAYS}일 이상인 해만 사용했습니다."
)
st.caption(
    f"회귀식: 예상 평균기온 = {slope:.5f} × 연도 + ({intercept:.3f}). "
    "교육용 단순 선형 모델이며 실제 미래 기온을 보장하지 않습니다."
)

with st.expander("분석에 사용한 연도별 데이터"):
    table = yearly.copy()
    table["연평균기온"] = table["연평균기온"].round(2)
    st.dataframe(table, use_container_width=True, hide_index=True)
