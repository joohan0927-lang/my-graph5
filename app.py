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
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="기온 예측기 - 다항회귀",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

END_YEAR = 2025
MIN_DAYS = 300

# 2005년 이전 = 훈련
# 2005년부터 = 테스트
SPLIT_YEAR = 2005

# 회귀 계산용 기준 연도
# 실제 연도 대신 (연도 - 2005)를 사용
CENTER_YEAR = 2005


# ---------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜", "평균기온"]).copy()
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

    # 2025년까지 + 관측일수 300일 이상
    annual = annual[
        (annual["연도"] <= END_YEAR)
        & (annual["관측일수"] >= MIN_DAYS)
    ].copy()

    annual = annual.sort_values("연도").reset_index(drop=True)

    return annual


# ---------------------------------------------------------
# 다항회귀 모델
# ---------------------------------------------------------
def fit_polynomial(annual, degree):
    train = annual[annual["연도"] < SPLIT_YEAR].copy()
    test = annual[annual["연도"] >= SPLIT_YEAR].copy()

    # 큰 연도 숫자를 그대로 사용하지 않고
    # 2005년을 0으로 만드는 방식으로 중심화
    x_train = (train["연도"] - CENTER_YEAR).to_numpy(dtype=float)
    y_train = train["연평균기온"].to_numpy(dtype=float)

    x_test = (test["연도"] - CENTER_YEAR).to_numpy(dtype=float)
    y_test = test["연평균기온"].to_numpy(dtype=float)

    # 다항식 계수 계산
    coefficients = np.polyfit(x_train, y_train, degree)
    polynomial = np.poly1d(coefficients)

    # 테스트 데이터 예측
    test_prediction = polynomial(x_test)

    # 평균 절대 오차(MAE)
    mae = np.mean(np.abs(y_test - test_prediction))

    # 2050년 예측
    x_2050 = 2050 - CENTER_YEAR
    prediction_2050 = float(polynomial(x_2050))

    # 학습 데이터 예측
    train_prediction = polynomial(x_train)

    # 학습 R²
    ss_res = np.sum((y_train - train_prediction) ** 2)
    ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
    train_r2 = 1 - ss_res / ss_tot

    return {
        "degree": degree,
        "polynomial": polynomial,
        "coefficients": coefficients,
        "train": train,
        "test": test,
        "test_prediction": test_prediction,
        "mae": mae,
        "prediction_2050": prediction_2050,
        "train_r2": train_r2,
    }


# ---------------------------------------------------------
# 앱 시작
# ---------------------------------------------------------
st.title("🌡️ 기온 예측기 — 다항회귀 비교")

st.write(
    "서울의 연평균기온을 이용해 1차, 3차, 9차 다항회귀를 비교합니다. "
    "모델은 2005년 이전 자료만 학습하고, 2005년 이후 자료로 성능을 평가합니다."
)

try:
    raw_df = load_data()
    annual = make_annual_data(raw_df)

except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


if annual.empty:
    st.error("조건을 만족하는 연도별 데이터가 없습니다.")
    st.stop()


# ---------------------------------------------------------
# 훈련/테스트 데이터 분리
# ---------------------------------------------------------
train = annual[annual["연도"] < SPLIT_YEAR].copy()
test = annual[annual["연도"] >= SPLIT_YEAR].copy()

train_start = int(train["연도"].min())
train_end = int(train["연도"].max())

test_start = int(test["연도"].min())
test_end = int(test["연도"].max())


# ---------------------------------------------------------
# 데이터 개수 표시
# ---------------------------------------------------------
st.subheader("훈련용 / 테스트용 데이터")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "훈련용 연도 수",
        f"{len(train)}개",
    )
    st.write(
        f"**{train_start}~{train_end}년** "
        f"(2005년 이전)"
    )

with col2:
    st.metric(
        "테스트용 연도 수",
        f"{len(test)}개",
    )
    st.write(
        f"**{test_start}~{test_end}년** "
        f"(2005년부터)"
    )

st.info(
    "중요: 회귀식은 훈련용 데이터만 사용하여 만들고, "
    "MAE는 모델이 학습할 때 사용하지 않은 테스트용 데이터에 대해서만 계산합니다."
)


# ---------------------------------------------------------
# 모델 학습
# ---------------------------------------------------------
degrees = [1, 3, 9]

models = {}

for degree in degrees:
    models[degree] = fit_polynomial(annual, degree)


# ---------------------------------------------------------
# 결과 표
# ---------------------------------------------------------
st.subheader("모델별 테스트 성능 및 2050년 예측")

results = []

for degree in degrees:
    model = models[degree]

    results.append(
        {
            "모델": f"{degree}차",
            "테스트 MAE (℃)": model["mae"],
            "2050년 예상기온 (℃)": model["prediction_2050"],
        }
    )

result_df = pd.DataFrame(results)

st.dataframe(
    result_df.style.format(
        {
            "테스트 MAE (℃)": "{:.3f}",
            "2050년 예상기온 (℃)": "{:.2f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# 모델 설명
# ---------------------------------------------------------
st.caption(
    "MAE가 작을수록 학습에 사용하지 않은 테스트 데이터의 실제 기온을 "
    "평균적으로 적게 빗나간다는 뜻입니다."
)

st.write(
    f"회귀 계산에서는 실제 연도 대신 **연도 − {CENTER_YEAR}**를 사용했습니다. "
    f"따라서 2005년은 0, 2004년은 -1, 2025년은 +20으로 변환되어 "
    f"고차 다항식의 수치적 불안정성을 줄였습니다."
)


# ---------------------------------------------------------
# Plotly 그래프
# ---------------------------------------------------------
st.subheader("훈련 데이터, 테스트 데이터와 다항회귀 곡선")

fig = go.Figure()

# 훈련 데이터
fig.add_trace(
    go.Scatter(
        x=train["연도"],
        y=train["연평균기온"],
        mode="markers",
        name="훈련용 데이터",
        marker=dict(
            size=6,
            color="#1f77b4",
            opacity=0.65,
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)

# 테스트 데이터
fig.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="markers",
        name="테스트용 데이터",
        marker=dict(
            size=8,
            color="#ff7f0e",
            symbol="diamond",
            opacity=0.9,
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "실제 연평균기온: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)


# ---------------------------------------------------------
# 회귀 곡선
# ---------------------------------------------------------
# 1900~2100 범위에서 곡선을 그린다.
curve_years = np.linspace(1900, 2100, 1000)
curve_x = curve_years - CENTER_YEAR

colors = {
    1: "#d62728",
    3: "#2ca02c",
    9: "#9467bd",
}

for degree in degrees:
    model = models[degree]
    polynomial = model["polynomial"]

    curve_temperature = polynomial(curve_x)

    fig.add_trace(
        go.Scatter(
            x=curve_years,
            y=curve_temperature,
            mode="lines",
            name=f"{degree}차 회귀",
            line=dict(
                color=colors[degree],
                width=3 if degree == 1 else 2,
            ),
            hovertemplate=(
                f"{degree}차 회귀<br>"
                "연도: %{x:.0f}년<br>"
                "예측기온: %{y:.2f} ℃"
                "<extra></extra>"
            ),
        )
    )


# 2005년 경계선
fig.add_vline(
    x=SPLIT_YEAR,
    line_dash="dash",
    line_color="black",
    annotation_text="2005년: 학습 → 테스트",
    annotation_position="top",
)


fig.update_layout(
    template="plotly_white",
    height=650,
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
)

# 연도 자체를 x축에 표시
fig.update_xaxes(
    range=[1900, 2100],
    dtick=10,
    tickformat="d",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# ---------------------------------------------------------
# 모델별 상세 정보
# ---------------------------------------------------------
st.subheader("모델별 상세 정보")

for degree in degrees:
    model = models[degree]

    with st.expander(f"{degree}차 회귀 자세히 보기"):
        st.write(
            f"- 학습 데이터: {len(model['train'])}개 연도"
        )
        st.write(
            f"- 테스트 데이터: {len(model['test'])}개 연도"
        )
        st.write(
            f"- 테스트 MAE: **{model['mae']:.3f} ℃**"
        )
        st.write(
            f"- 학습 데이터 R²: **{model['train_r2']:.4f}**"
        )
        st.write(
            f"- 2050년 예측: **{model['prediction_2050']:.2f} ℃**"
        )


# ---------------------------------------------------------
# 데이터 보기
# ---------------------------------------------------------
with st.expander("연도별 데이터 보기"):
    display_df = annual.copy()

    display_df["구분"] = np.where(
        display_df["연도"] < SPLIT_YEAR,
        "훈련",
        "테스트",
    )

    display_df["연평균기온"] = display_df["연평균기온"].round(2)

    st.dataframe(
        display_df[
            ["연도", "연평균기온", "관측일수", "구분"]
        ],
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# 데이터 출처
# ---------------------------------------------------------
st.caption(
    "데이터 출처: 서울 기온 일자료(seoul.csv). "
    "2025년까지의 자료 중 연간 관측일수가 300일 이상인 연도만 사용했습니다."
)
