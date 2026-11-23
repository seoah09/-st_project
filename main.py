import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
import os

# 1. 리눅스(스트림릿 클라우드) 및 로컬 환경 한글 폰트 설정
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')
else:
    plt.rc('font', family=['Malgun Gothic', 'AppleGothic', 'sans-serif'])

plt.rcParams['axes.unicode_minus'] = False

# 페이지 기본 설정
st.set_page_config(
    page_title="사교육비와 진학률 데이터 분석",
    page_icon="📊",
    layout="wide"
)

# 2. 상단 제목 및 팀 질문
st.title("📊 사교육비가 높을수록 진학률은 정말 높아질까?")
st.markdown("""
- **팀 핵심 질문:** *"지역별 학생 1인당 월평균 사교육비가 높을수록 고등학교 졸업생 진학률은 정말 높아지는가?"*
- **분석 방법:** 단순 선형 회귀 분석 & 지역별 비교 시각화
- **1차 출처:** 통계청 국가통계포털(KOSIS) 사교육비 조사 & 교육통계서비스(KESS) 교육기본통계
""")
st.divider()

# 3. 사이드바: 데이터 선택 및 전처리 옵션
st.sidebar.header("⚙️ 데이터 및 전처리 설정")
data_option = st.sidebar.radio(
    "1) 데이터 소스 선택",
    ("기본 제공 데이터 (예시)", "내 CSV 파일 직접 업로드")
)

@st.cache_data
def load_default_data():
    # 전처리 시연이 가능하도록 예시 데이터 구성
    data = {
        '시도': ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', 
               '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'],
        '월평균_사교육비_만원': [62.8, 41.2, 45.6, 40.5, 42.1, 44.3, 41.8, 48.5, 
                         47.9, 34.2, 35.8, 36.4, 34.9, 32.5, 36.1, 37.8, 38.9],
        '진학률_퍼센트': [76.5, 73.2, 75.8, 72.1, 77.4, 76.9, 74.1, 78.2, 
                     74.5, 72.8, 75.1, 73.9, 76.2, 74.8, 75.5, 74.9, 73.6]
    }
    return pd.DataFrame(data)

if data_option == "기본 제공 데이터 (예시)":
    raw_df = load_default_data()
else:
    uploaded_file = st.sidebar.file_uploader("병합한 CSV 파일을 업로드하세요", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            raw_df = pd.read_csv(uploaded_file, encoding='cp949')
    else:
        st.info("👈 왼쪽 사이드바에서 CSV 파일을 업로드하거나 '기본 제공 데이터'를 선택해 주세요.")
        st.stop()

numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
text_cols = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    st.error("회귀 분석을 위해서는 숫자 형태의 열(변수)이 최소 2개 이상 필요합니다.")
    st.stop()

label_col = st.sidebar.selectbox("2) 구분(지역명) 열", text_cols if text_cols else raw_df.columns)
x_col = st.sidebar.selectbox("3) 독립변수 (X축: 사교육비)", numeric_cols, index=0)
y_col = st.sidebar.selectbox("4) 종속변수 (Y축: 진학률)", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

st.sidebar.markdown("---")
st.sidebar.subheader("🧹 전처리 옵션 (이상치 처리)")
remove_outliers = st.sidebar.checkbox("IQR 기준 이상치 제거 적용하기", value=False)

# ==============================================================================
# [평가 항목 1] 데이터 감식 - 데이터 개요 (행·열 수, 자료형, 빈 값)
# ==============================================================================
st.header("1. 데이터 개요 및 전처리 과정 (데이터 감식)")

col_info1, col_info2 = st.columns([1, 1])

with col_info1:
    st.subheader("📋 원본 데이터 구조 요약")
    st.write(f"- **원본 데이터 크기:** 총 **{raw_df.shape[0]}행** × **{raw_df.shape[1]}열**")
    
    # 열별 자료형 및 결측치(빈 값) 요약 테이블 생성
    summary_df = pd.DataFrame({
        '열 이름': raw_df.columns,
        '자료형 (Data Type)': [str(dtype) for dtype in raw_df.dtypes],
        '빈 값 (결측치 수)': raw_df.isnull().sum().values,
        '유효 데이터 수': raw_df.notnull().sum().values
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ==============================================================================
# [평가 항목 3] 전처리의 앱 반영 - 결측치/이상치 기준, 전후 행 수, 편향(치우침)
# ==============================================================================
# 1단계: 결측치 처리 (분석 핵심 열에 빈 값이 있는 행 제거)
df_no_na = raw_df.dropna(subset=[x_col, y_col]).copy()
na_removed_count = len(raw_df) - len(df_no_na)

# 2단계: 이상치 처리 (IQR 사분위수 기준: Q1 - 1.5*IQR 미만 또는 Q3 + 1.5*IQR 초과)
Q1 = df_no_na[x_col].quantile(0.25)
Q3 = df_no_na[x_col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_df = df_no_na[(df_no_na[x_col] < lower_bound) | (df_no_na[x_col] > upper_bound)]

if remove_outliers:
    clean_df = df_no_na[(df_no_na[x_col] >= lower_bound) & (df_no_na[x_col] <= upper_bound)].copy()
else:
    clean_df = df_no_na.copy()

with col_info2:
    st.subheader("🧹 결측치·이상치 처리 및 편향 점검")
    
    # 처리 전후 행 수 비교 지표
    m1, m2, m3 = st.columns(3)
    m1.metric("처리 전 행 수", f"{len(raw_df)}행")
    m2.metric("처리 후 행 수", f"{len(clean_df)}행", delta=f"{len(clean_df) - len(raw_df)}행")
    m3.metric("탐지된 이상치", f"{len(outliers_df)}건")
    
    st.markdown(f"""
    - **결측치 처리 기준:** `{x_col}` 또는 `{y_col}`에 빈 값(NaN)이 있는 행은 평균 대체 시 지역 왜곡이 생기므로 제거함 (**{na_removed_count}행 제거**).
    - **이상치 처리 기준:** 사분위수 범위(IQR) 기준 정상 범위(`{lower_bound:.1f}` ~ `{upper_bound:.1f}`)를 벗어나는 극단값을 이상치로 판별함.
      {f"(탐지된 지역: {', '.join(outliers_df[label_col].astype(str).tolist())})" if len(outliers_df) > 0 else "(탐지된 이상치 없음)"}
    - **처리로 생길 수 있는 치우침(편향):** 사교육비가 압도적으로 높은 대도시(예: 서울)를 이상치로 제거할 경우, **고액 사교육비 구간의 데이터가 사라져 회귀선의 기울기(상관관계)가 실제보다 완만하게 축소되는 표본 선택 편향**이 발생할 수 있습니다.
    """)

st.divider()

# ==============================================================================
# [평가 항목 1] 팀 질문과 연결된 그래프 2개 이상 (제목 + 축 이름 필수 포함)
# ==============================================================================
st.header("2. 시각화 분석 (질문 검증 그래프 2종)")

# 회귀 통계량 계산
x = clean_df[x_col]
y = clean_df[y_col]
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
r_squared = r_value ** 2

g_col1, g_col2 = st.columns(2)

# [그래프 1] 선형 회귀 산점도
with g_col1:
    st.subheader("그래프 ① 사교육비와 진학률의 선형 회귀 분석")
    fig1, ax1 = plt.subplots(figsize=(8, 5.5))
    
    sns.regplot(
        data=clean_df,
        x=x_col,
        y=y_col,
        scatter_kws={'s': 85, 'color': '#2b5c8f', 'alpha': 0.85},
        line_kws={'color': '#d9534f', 'linewidth': 2.2, 'label': f'회귀선 (y={slope:.2f}x+{intercept:.2f}, R²={r_squared:.2f})'},
        ax=ax1
    )
    
    for i in range(len(clean_df)):
        ax1.text(
            clean_df[x_col].iloc[i] + (x.max() - x.min()) * 0.015,
            clean_df[y_col].iloc[i],
            str(clean_df[label_col].iloc[i]),
            fontsize=9,
            color='#222222'
        )
        
    ax1.set_title(f"시·도별 {x_col}와 {y_col}의 상관관계", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel(f"{x_col} (독립변수)", fontsize=11)
    ax1.set_ylabel(f"{y_col} (종속변수)", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', fontsize=9.5)
    st.pyplot(fig1)

# [그래프 2] 지역별 사교육비 순위 대비 진학률 비교 막대·선 복합 그래프 (예외 지역 탐색용)
with g_col2:
    st.subheader("그래프 ② 지역별 사교육비 순위 및 진학률 비교")
    sorted_df = clean_df.sort_values(by=x_col, ascending=False).reset_index(drop=True)
    
    fig2, ax2 = plt.subplots(figsize=(8, 5.5))
    bars = ax2.bar(sorted_df[label_col], sorted_df[x_col], color='#6baed6', alpha=0.85, label=x_col)
    ax2.set_title(f"지역별 {x_col} 순위 및 {y_col} 비교", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("시·도 (지역명)", fontsize=11)
    ax2.set_ylabel(x_col, fontsize=11, color='#1f4e78')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    # 보조축으로 진학률 꺾은선 표시
    ax2_sub = ax2.twinx()
    ax2_sub.plot(sorted_df[label_col], sorted_df[y_col], color='#e6550d', marker='o', linewidth=2, label=y_col)
    ax2_sub.set_ylabel(y_col, fontsize=11, color='#a63603')
    
    # 범례 통합
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_sub.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9.5)
    
    st.pyplot(fig2)

st.divider()

# ==============================================================================
# [평가 항목 2] 찾아낸 인사이트 (근거 수치 + 질문 연결 + 예상과 다른 점 + 다음 분석 과제)
# ==============================================================================
st.header("3. 찾아낸 인사이트 및 결과 해석")

# 주요 근거 수치 자동 추출
max_x_row = clean_df.loc[clean_df[x_col].idxmax()]
min_x_row = clean_df.loc[clean_df[x_col].idxmin()]
max_y_row = clean_df.loc[clean_df[y_col].idxmax()]
min_y_row = clean_df.loc[clean_df[y_col].idxmin()]

ic1, ic2, ic3, ic4 = st.columns(4)
ic1.metric("상관계수 (r)", f"{r_value:.3f}")
ic2.metric("결정계수 (R²)", f"{r_squared*100:.1f}%")
ic3.metric("유의확률 (p-value)", f"{p_value:.4f}")
ic4.metric("회귀 기울기", f"+{slope:.3f}")

st.markdown(f"""
#### ① 그래프에서 찾아낸 핵심 사실 (근거 수치)
- **사교육비 격차:** 조사 대상 중 **{max_x_row[label_col]}**의 사교육비가 **{max_x_row[x_col]:.1f}**로 가장 높았고, **{min_x_row[label_col]}**이 **{min_x_row[x_col]:.1f}**로 가장 낮아 약 **{max_x_row[x_col] - min_x_row[x_col]:.1f}**의 차이를 보였습니다.
- **회귀 분석 결과:** 회귀식의 기울기는 **{slope:.3f}**로, 사교육비가 1단위 증가할 때 진학률이 평균 **{slope:.3f}%p** 상승하는 양(+)의 방향성을 보입니다. 상관계수($r$)는 **{r_value:.3f}**, 결정계수($R^2$)는 **{r_squared*100:.1f}%**입니다.

#### ② 팀 질문과의 연결 결론
- *"사교육비가 높을수록 진학률이 정말 높아지는가?"*라는 질문에 대해, 전체적인 추세선은 우상향하지만 **설명력($R^2$)이 {r_squared*100:.1f}%** 수준으로 사교육비만으로 각 지역의 진학률을 100% 설명하기는 어렵다는 결론을 얻었습니다.

#### ③ 예상과 다른 점 및 다음 분석에서 확인할 점
- **예상과 다른 반전 지역(그래프 ② 확인):** 사교육비가 가장 높은 **{max_x_row[label_col]}({max_x_row[x_col]:.1f})**보다, 사교육비가 중위권인 **{max_y_row[label_col]}({max_y_row[x_col]:.1f})**의 진학률이 **{max_y_row[y_col]:.1f}%**로 더 높게 나타났습니다. 이는 사교육비 외에 **지역별 특성화고·일반고 비율, 공교육(방과후학교) 지원 수준, 수시·정시 전형 비율** 등 다른 변수가 진학률에 큰 영향을 미침을 시사합니다.
- **다음 분석 과제:** 다음 단계에서는 단순 '전체 진학률' 대신 **'4년제 대학 진학률' 또는 '학업성취도 평가 점수'**로 종속변수를 세분화하고, **학교 수·교원 1인당 학생 수** 변수를 추가한 다중 회귀 분석을 진행할 계획입니다.
""")

with st.expander("🔍 전처리 완료된 전체 데이터 표 보기"):
    st.dataframe(clean_df, use_container_width=True)
