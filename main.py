import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import platform

# 1. 한글 폰트 설정 (그래프에서 한글 깨짐 방지)
system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin':  # Mac
    plt.rc('font', family='AppleGothic')
else:  # Linux / Google Colab
    plt.rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 2. 데이터 불러오기
# 실제 CSV 파일을 사용할 때는 아래 주석을 풀고 파일명을 입력하세요.
# df = pd.read_csv('education_data.csv', encoding='utf-8')

# [테스트용 예시 데이터] 실제 KOSIS/KESS 데이터로 교체하세요!
data = {
    '시도': ['서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', 
           '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주'],
    '월평균_사교육비_만원': [62.8, 41.2, 45.6, 40.5, 42.1, 44.3, 41.8, 48.5, 
                     47.9, 34.2, 35.8, 36.4, 34.9, 32.5, 36.1, 37.8, 38.9],
    '진학률_퍼센트': [76.5, 73.2, 75.8, 72.1, 77.4, 76.9, 74.1, 78.2, 
                 74.5, 72.8, 75.1, 73.9, 76.2, 74.8, 75.5, 74.9, 73.6]
}
df = pd.DataFrame(data)

# 3. 회귀 분석 통계량 계산 (기울기, 절편, 상관계수 r, p-value)
x = df['월평균_사교육비_만원']
y = df['진학률_퍼센트']
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
r_squared = r_value ** 2  # 결정계수 (R^2)

print(f"회귀식: y = {slope:.4f}x + {intercept:.4f}")
print(f"상관계수 (r): {r_value:.4f}")
print(f"결정계수 (R²): {r_squared:.4f}")
print(f"유의확률 (p-value): {p_value:.4f}")

# 4. 회귀 그래프 그리기
plt.figure(figsize=(10, 6))

# 산점도 + 회귀선 (ci=95는 95% 신뢰구간 음영 표시)
sns.regplot(
    data=df, 
    x='월평균_사교육비_만원', 
    y='진학률_퍼센트',
    scatter_kws={'s': 80, 'color': '#2b5c8f', 'alpha': 0.8},
    line_kws={'color': '#d9534f', 'linewidth': 2, 'label': f'회귀선 (R² = {r_squared:.2f})'}
)

# 각 점 옆에 시·도 이름 표시
for i in range(len(df)):
    plt.text(
        df['월평균_사교육비_만원'][i] + 0.4, 
        df['진학률_퍼센트'][i], 
        df['시도'][i], 
        fontsize=9, 
        color='#333333'
    )

# 그래프 제목 및 축 라벨 설정
plt.title('시·도별 학생 1인당 월평균 사교육비와 대입 진학률의 관계', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('학생 1인당 월평균 사교육비 (단위: 만원)', fontsize=12)
plt.ylabel('고등학교 졸업생 진학률 (단위: %)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='
