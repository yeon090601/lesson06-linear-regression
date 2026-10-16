import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
MAX_YEAR = 2025
MIN_DAYS = 300
RECENT_YEARS = 20

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")
st.title("🌡️ 기온 예측기")
st.write("서울의 연평균기온으로 장기 추세와 최근 추세를 비교합니다.")


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
slope_all, intercept_all = np.polyfit(x, y, 1)
correlation = float(np.corrcoef(x, y)[0, 1])

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
year_count = len(yearly)

# 최근 20년: 필터링된 데이터에서 가장 최근 연도부터 20년 구간을 사용합니다.
recent_start_year = end_year - RECENT_YEARS + 1
recent = yearly[yearly["연도"] >= recent_start_year]
if len(recent) >= 2:
    slope_recent, intercept_recent = np.polyfit(
        recent["연도"].to_numpy(dtype=float),
        recent["연평균기온"].to_numpy(dtype=float),
        1,
    )
    recent_fit_ok = True
else:
    slope_recent, intercept_recent = np.nan, np.nan
    recent_fit_ok = False

# 기울기(°C/년)에 100을 곱해 100년당 변화량으로 변환합니다.
rise_100_all = slope_all * 100
rise_100_recent = slope_recent * 100 if recent_fit_ok else np.nan

st.subheader("100년당 기온 변화 비교")
metric_cols = st.columns(2)
with metric_cols[0]:
    st.metric(
        label=f"전체 기간 ({start_year}~{end_year}년)",
        value=f"{rise_100_all:+.2f} °C / 100년",
        help="전체 분석 기간의 선형회귀 기울기에 100을 곱한 값입니다.",
    )
with metric_cols[1]:
    if recent_fit_ok:
        st.metric(
            label=f"최근 20년 ({int(recent['연도'].min())}~{int(recent['연도'].max())}년)",
            value=f"{rise_100_recent:+.2f} °C / 100년",
            help="최근 구간의 연평균기온으로 별도의 선형회귀를 계산했습니다.",
        )
    else:
        st.metric(label="최근 20년", value="계산할 데이터 부족")

st.caption(
    "양수는 기온 상승 추세, 음수는 하강 추세를 뜻합니다. "
    "최근 20년 계산은 관측일 300일 이상 조건을 통과한 연도 중 가장 최근 연도를 기준으로 합니다."
)

selected_year = st.slider(
    "예측할 연도",
    min_value=1900,
    max_value=2100,
    value=2050,
    step=1,
)
prediction = slope_all * selected_year + intercept_all
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
        y=slope_all * line_x + intercept_all,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(width=3),
        hovertemplate="연도: %{x:.0f}년<br>회귀값: %{y:.2f} °C<extra></extra>",
    )
)
if recent_fit_ok:
    recent_x = np.linspace(int(recent["연도"].min()), int(recent["연도"].max()), 100)
    fig.add_trace(
        go.Scatter(
            x=recent_x,
            y=slope_recent * recent_x + intercept_recent,
            mode="lines",
            name="최근 20년 회귀 직선",
            line=dict(width=3, dash="dash"),
            hovertemplate="연도: %{x:.0f}년<br>최근 회귀값: %{y:.2f} °C<extra></extra>",
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
    f"전체 회귀식: 예상 평균기온 = {slope_all:.5f} × 연도 + ({intercept_all:.3f})."
)
if recent_fit_ok:
    st.caption(
        f"최근 구간 회귀식: 예상 평균기온 = {slope_recent:.5f} × 연도 + ({intercept_recent:.3f})."
    )
st.caption(
    f"2025년 이하 자료 중 유효 관측일이 {MIN_DAYS}일 이상인 해만 사용했습니다. "
    "회귀 추세는 교육용 통계 모델이며 실제 미래 기온을 보장하지 않습니다."
)

with st.expander("분석에 사용한 연도별 데이터"):
    table = yearly.copy()
    table["연평균기온"] = table["연평균기온"].round(2)
    st.dataframe(table, use_container_width=True, hide_index=True)
