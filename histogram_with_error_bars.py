import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 파일 경로 지정
file_path = "C:\\Users\\Public\\Desktop\\kosdaq_data_stat_5min.csv"
data = pd.read_csv(file_path)

# x축과 y축으로 사용할 필드명을 변수로 지정 (필요에 따라 변경)
x_field = "sellday_open_per_buyday_close"  # 예: "some_numeric_field1" 또는 "last_change"
y_field = "sellday_morning_high_per_buyday_close"  # 예: "some_numeric_field2" 또는 "buyday_close_per_last_vwap"

# 예시로 필드명을 원하는 값으로 설정하세요.
# 예시:
# x_field = "last_change"
# y_field = "buyday_close_per_last_vwap"

# x_field와 y_field에 결측치가 있는 행 제거
data = data.dropna(subset=[x_field, y_field])

# x_field를 100개의 bin으로 나누어 각 bin의 y_field 평균과 표준편차 계산 함수
def calculate_means_and_stds_for_bins(data, x_field, y_field, bins=100):
    # x_field에 대해 히스토그램 계산 (각 bin의 데이터 개수)
    hist, bin_edges = np.histogram(data[x_field], bins=bins)
    means, stds = [], []
    for i in range(len(bin_edges) - 1):
        # 각 bin에 해당하는 데이터 추출
        bin_data = data[(data[x_field] >= bin_edges[i]) & (data[x_field] < bin_edges[i+1])]
        if not bin_data.empty:
            means.append(bin_data[y_field].mean())
            stds.append(bin_data[y_field].std())
        else:
            means.append(np.nan)
            stds.append(np.nan)
    return hist, bin_edges, means, stds

# 함수 실행하여 계산
hist, bin_edges, means, stds = calculate_means_and_stds_for_bins(data, x_field, y_field, bins=100)

# 각 bin의 중심값 계산
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 그래프 그리기
fig, ax1 = plt.subplots(figsize=(12, 8))

# 왼쪽 y축: 보라색 히스토그램 (x_field의 분포)
ax1.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), color="purple", alpha=0.7, label="Histogram")
ax1.set_xlabel(x_field, fontsize=12)
ax1.set_ylabel("Frequency", fontsize=12)
ax1.legend(loc="upper left")
ax1.grid(False, axis="y")  # 히스토그램 y축 그리드 제거

# 오른쪽 y축: 에러바로 y_field의 평균 및 ±1 표준편차 표시
ax2 = ax1.twinx()
ax2.errorbar(bin_centers, means, yerr=stds, fmt='o', color='blue', 
             label=f"Mean ± 1 Std Dev of {y_field}")
ax2.set_ylabel(y_field, fontsize=12)
ax2.legend(loc="upper right")

plt.title(f"Relationship between {x_field} and {y_field}", fontsize=14)
plt.tight_layout()
plt.show()