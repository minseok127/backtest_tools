import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 경로
csv_file = "C:\\Users\\Public\\Desktop\\kosdaq_data_stat_5min.csv"

# 히스토그램을 그릴 필드 이름 (원하는 필드명으로 변경)
field = "buyday_1430_1500_high_per_buyday_close"

# CSV 파일 읽기
try:
    data = pd.read_csv(csv_file)
except Exception as e:
    print(f"CSV 파일을 읽는 중 에러 발생: {e}")
    exit()

# 필드가 존재하는지 확인
if field not in data.columns:
    print(f"'{field}' 컬럼이 CSV 파일에 존재하지 않습니다. 사용 가능한 컬럼: {data.columns.tolist()}")
    exit()

# 히스토그램 그리기 (NaN 값은 제외)
plt.figure(figsize=(10, 6))
plt.hist(data[field].dropna(), bins=100, color='skyblue', edgecolor='black')
plt.title(f"Histogram of {field}")
plt.xlabel(field)
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
