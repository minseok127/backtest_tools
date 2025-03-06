import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV 파일 경로와 분석할 필드 이름들을 지정합니다.
csv_file = "C:\\Users\\Public\\Desktop\\kosdaq_data_stat_5min.csv"  # 분석할 CSV 파일의 경로를 입력하세요.
fields = ["sellday_morning_high_per_buyday_close"]  # 분석하고자 하는 필드 이름들을 입력하세요.

# CSV 파일 읽기
df = pd.read_csv(csv_file)

# 데이터를 정렬한 후, 각 값에서 1을 빼서 새로운 변수 Y (즉, Y = X - 1)를 생성하고,
# 이에 대한 생존 함수 P(Y ≥ y)를 계산하는 함수
def compute_survival_function(data):
    # 데이터를 오름차순으로 정렬 후 Y = X - 1 적용
    x_sorted = np.sort(data)
    y_vals = x_sorted - 1
    n = len(y_vals)
    # 각 y 값에 대해 y 이상인 값의 비율을 계산 (생존 함수)
    survival = np.arange(n, 0, -1) / n
    return y_vals, survival

# ----------------------------
# 1. 각 필드의 CDF (P(Y ≥ y)) 그래프들을 하나의 화면에 나란히 그리기
# ----------------------------
num_fields = len(fields)
fig, axs = plt.subplots(1, num_fields, figsize=(5 * num_fields, 4))
if num_fields == 1:
    axs = [axs]  # subplot이 하나일 경우 리스트로 변환

for ax, field in zip(axs, fields):
    # 해당 필드의 데이터 가져오기 (결측값 제거)
    data = df[field].dropna().values
    y_vals, survival = compute_survival_function(data)
    
    # step plot으로 생존 함수 P(Y ≥ y)를 그림
    ax.step(y_vals, survival, where='post', label='P(Y ≥ y)')
    ax.set_title(f"{field}의 CDF (Y = X - 1 적용)")
    ax.set_xlabel("Y (X - 1)")
    ax.set_ylabel("P(Y ≥ y)")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# ----------------------------
# 2. 각 필드에 대해 y 값과 CDF 값의 곱 (y * P(Y ≥ y)) 그래프들을 하나의 화면에 나란히 그리기
# ----------------------------
fig2, axs2 = plt.subplots(1, num_fields, figsize=(5 * num_fields, 4))
if num_fields == 1:
    axs2 = [axs2]

for ax, field in zip(axs2, fields):
    data = df[field].dropna().values
    y_vals, survival = compute_survival_function(data)

    mask = y_vals >= -0.3
    y_vals = y_vals[mask]
    survival = survival[mask]
    
    product = y_vals * survival  # y 값과 P(Y ≥ y)의 곱 계산
    ax.step(y_vals, product, where='post', label='y * P(Y ≥ y)')
    ax.set_title(f"{field}의 y * CDF 그래프 (Y = X - 1 적용)")
    ax.set_xlabel("Y (X - 1)")
    ax.set_ylabel("y * P(Y ≥ y)")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()