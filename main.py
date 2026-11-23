import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
import os

# 리눅스(스트림릿 클라우드) 및 로컬 환경 한글 폰트 설정
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')
else:
    # 로컬 PC(Windows/Mac) 실행 시 대비
    plt.rc('font', family=['Malgun Gothic', 'AppleGothic', 'sans-serif'])

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 페이지 기본 설정
st.set_page_config(
    page_title="사교육비와 진학률 회귀 분석",
    page_icon="📊",
    layout="wide"
)

# 1. 제목 및 프로젝트 개요 (평가 기준표 요약)
st.title("📊 사교육비가 높을수록 진학률은 정말 높아질까?")
st.markdown("""
- **분석 방법:** 단순 선형 회귀 분석 (Linear Regression)
- **데이터 출처:** 통계청 국가통계포털(KOSIS) 사교육비 조사 & 교육통계서비스(KESS) 교육기본통계
""")
st.divider()

# 2. 사이드바: 데이터 선택 및 옵션 설정
st.sidebar.header("⚙️ 데이터 및 그래프 설정")
data_option = st.sidebar.radio(
    "사용할 데이터를 선택하세요:",
    ("기본 제공 데이터 (예시)", "내 CSV 파일 직접 업로드")
)

# 기본 예시 데이터 생성 함수
@st.cache_data
def load_default_data():
    data = {
        '시도': ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', 
               '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'],
        '월평균_사교육비_만원': [62.8, 41.2, 45.6, 40.5, 42.1, 44.3, 41.8, 48.5, 
                         47.9, 34.2, 35.8, 36.4, 34.9, 32.5, 36.1, 37.8, 38.9],
        '진학률_퍼센트': [76.5, 73.2, 75.8, 72.1, 77.4, 76.9, 74.1, 78.2, 
                     74.5, 72.8, 75.1, 73.9, 76.2, 74.8, 75.5, 74.9, 73.6]
    }
    return pd.DataFrame(data)

# 데이터 로드 분기 처리
if data_option == "기본 제공 데이터 (예시)":
    df = load_default_data()
else:
    uploaded_file = st.sidebar.file_uploader("병합한 CSV 파일을 업로드하세요", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='cp949')  # 윈도우 엑셀 CSV 인코딩 대응
    else:
        st.info("👈 왼쪽 사이드바에서 CSV 파일을 업로드하거나 '기본 제공 데이터'를 선택해 주세요.")
        st.stop()

# 열(변수) 선택 기능 (업로드한 파일의 열 이름이 달라도 동작하도록 구성)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
text_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    st.error("회귀 분석을 위해서는 숫자 형태의 열(변수)이 최소 2개 이상 필요합니다.")
    st.stop()

label_col = st.sidebar.selectbox("구분(라벨) 열 선택", text_cols if text_cols else df.columns)
x_col = st.sidebar.selectbox("독립변수 (X축: 원인)", numeric_cols, index=0)
y_col = st.sidebar.selectbox("종속변수 (Y축: 결과)", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

show_labels = st.sidebar.checkbox("그래프에 지역명 표시하기", value=True)
show_ci = st.sidebar.checkbox("95% 신뢰구간 음영 표시", value=True)

# 3. 회귀 분석 계산
x = df[x_col]
y = df[y_col]
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
r_squared = r_value ** 2

# 4. 핵심 통계 지표 카드 (Metric)
col1, col2, col3, col4 = st.columns(4)
col1.metric("회귀식 기울기", f"{slope:.4f}")
col2.metric("상관계수 (r)", f"{r_value:.4f}")
col3.metric("결정계수 (R²)", f"{r_squared:.4f}")
col4.metric("유의확률 (p-value)", f"{p_value:.4f}")

# 5. 메인 화면 분할: 왼쪽(회귀 그래프) / 오른쪽(데이터 표 & 시뮬레이터)
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("📈 선형 회귀 분석 그래프")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.regplot(
        data=df,
        x=x_col,
        y=y_col,
        ci=95 if show_ci else None,
        scatter_kws={'s': 90, 'color': '#2b5c8f', 'alpha': 0.85},
        line_kws={'color': '#d9534f', 'linewidth': 2.5, 'label': f'회귀선 (y = {slope:.2f}x + {intercept:.2f})'},
        ax=ax
    )

    if show_labels:
        for i in range(len(df)):
            ax.text(
                df[x_col].iloc[i] + (x.max() - x.min()) * 0.012,
                df[y_col].iloc[i],
                str(df[label_col].iloc[i]),
                fontsize=9.5,
                color='#222222'
            )

    ax.set_title(f"{x_col}에 따른 {y_col} 회귀 분석", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)
    
    st.pyplot(fig)

with right_col:
    st.subheader("🔮 회귀식 예측 시뮬레이터")
    st.caption("회귀 모델을 바탕으로 사교육비에 따른 예상 진학률을 계산해 봅니다.")
    
    input_x = st.slider(
        f"{x_col} 입력",
        min_value=float(x.min() * 0.8),
        max_value=float(x.max() * 1.2),
        value=float(x.mean()),
        step=0.5
    )
    predicted_y = slope * input_x + intercept
    st.success(f"**예상 {y_col}:** `{predicted_y:.2f}`")
    
    st.subheader("📋 분석 데이터 미리보기")
    st.dataframe(df, use_container_width=True, height=280)

# 6. 자동 결과 해석 박스
st.divider()
st.subheader("💡 분석 결과 자동 해석")
if p_value < 0.05:
    st.write(f"✅ 유의확률(p-value)이 **{p_value:.4f}**로 0.05보다 작아, 두 변수 간의 관계가 **통계적으로 유의미**합니다. "
             f"상관계수(r)는 **{r_value:.2f}**이며, 설명력(R²)은 **{r_squared*100:.1f}%**입니다.")
else:
    st.write(f"⚠️ 유의확률(p-value)이 **{p_value:.4f}**로 0.05 이상이므로, 사교육비만으로 진학률 변화를 통계적으로 뚜렷하게 설명하기 어렵습니다. "
             f"(상관계수 r = **{r_value:.2f}**, 설명력 R² = **{r_squared*100:.1f}%**)")
