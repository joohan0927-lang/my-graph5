import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

END_YEAR = 2025
MIN_DAYS = 300
REGRESSION_START_YEAR = 1908


# ---------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 숫자형 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 유효한 자료만 사용
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


# ---------------------------------------------------------
# 연도별 평균기온 계산
# ---------------------------------------------------------
@st.cache_data
def make_annual_data(df):
    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )

    # 2025년까지 + 관측일수가 300일 이상인 해만 사용
    annual = annual[
        (annual["연도"] <= END_YEAR)
        & (annual["관측일수"] >= MIN_DAYS)
    ].copy()

    # 회귀 분석은 1908년부터
    annual = annual[annual["연도"] >= REGRESSION_START_YEAR].copy()

    annual = annual.sort_values("연도").reset_index(drop=True)

    return annual


# ---------------------------------------------------------
# 선형회귀
# 독립변수 = 1908년부터 지난 연수
# x = 연도 - 1908
# ---------------------------------------------------------
def fit_regression(annual):
    x = (annual["연도"] - REGRESSION_START_YEAR).to_numpy(dtype=float)
    y = annual["연평균기온"].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)

    annual = annual.copy()
    annual["예측기온"] = intercept + slope * x

    # 상관계수
    correlation = np.corrcoef(x, y)[0, 1]

    # R²
    ss_res = np.sum((y - annual["예측기온"].to_numpy()) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1 - ss_res / ss_tot

    return annual, slope, intercept, correlation, r2


# ---------------------------------------------------------
# 앱
# ---------------------------------------------------------
st.title("🌡️ 기온 예측기")

st.write(
    "서울의 일별 평균기온 자료를 이용하여 연평균기온을 계산하고, "
    "1908년부터의 경과 연수를 독립 변수로 하는 선형회귀 모델을 만듭니다."
)

# 데이터 로드
try:
    raw_df = load_data()
    annual = make_annual_data(raw_df)
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

if annual.empty:
    st.error("조건을 만족하는 연도별 데이터가 없습니다.")
    st.stop()

# 회귀
annual, slope, intercept, correlation, r2 = fit_regression(annual)

start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())
n_years = len(annual)

# ---------------------------------------------------------
# 주요 정보
# ---------------------------------------------------------
st.subheader("회귀 분석 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("회귀에 사용한 연도 수", f"{n_years}년")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")

with col4:
    st.metric("상관계수", f"{correlation:.4f}")

st.info(
    f"관측일수가 {MIN_DAYS}일 미만인 해와 {END_YEAR}년 이후의 자료를 제외했습니다. "
    f"최종적으로 {start_year}~{end_year}년의 {n_years}개 연도가 회귀선 작성에 사용되었습니다."
)

# ---------------------------------------------------------
# 회귀식
# ---------------------------------------------------------
st.subheader("회귀식")

st.latex(
    rf"\hat{{T}} = {intercept:.4f} + {slope:.4f}(연도 - 1908)"
)

st.write(
    f"회귀선의 기울기: **{slope:.4f} ℃/년** "
    f"= **{slope * 10:.4f} ℃/10년**"
)

# ---------------------------------------------------------
# 연도 슬라이더
# ---------------------------------------------------------
st.subheader("기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

selected_x = selected_year - REGRESSION_START_YEAR
predicted_temp = intercept + slope * selected_x

st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temp:.2f} ℃",
)

# ---------------------------------------------------------
# Plotly 그래프
# ---------------------------------------------------------
st.subheader("연도별 연평균기온과 회귀선")

# 1900~2100 전체 범위에 대한 회귀선
line_years = np.arange(1900, 2101)
line_x = line_years - REGRESSION_START_YEAR
line_temps = intercept + slope * line_x

fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(
            size=7,
            color="#1f77b4",
            opacity=0.75,
        ),
        customdata=annual["관측일수"],
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} ℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        ),
    )
)

# 회귀선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="회귀 직선",
        line=dict(
            color="#d62728",
            width=3,
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 예측: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)

# 선택한 연도의 예측값
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name="선택 연도 예측",
        marker=dict(
            size=14,
            color="#ff7f0e",
            symbol="star",
        ),
        hovertemplate=(
            f"{selected_year}년<br>"
            f"예상 연평균기온: {predicted_temp:.2f} ℃"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified",
    template="plotly_white",
    height=600,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
)

# x축은 연도를 그대로 표시
fig.update_xaxes(
    range=[1900, 2100],
    dtick=10,
    tickformat="d",
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# 데이터 표
# ---------------------------------------------------------
with st.expander("사용된 연도별 데이터 보기"):
    display_df = annual.copy()
    display_df["연평균기온"] = display_df["연평균기온"].round(2)
    display_df["예측기온"] = display_df["예측기온"].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------
# 안내
# ---------------------------------------------------------
st.caption(
    "데이터 출처: 서울 기온 일자료(seoul.csv). "
    "2025년까지의 자료 중 연간 관측일수가 300일 이상인 연도만 사용했습니다."
)
