import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. CSV 파일 읽기
csv_path = "C:\\Users\\Public\\Desktop\\kosdaq_data_stat_5min.csv"
df = pd.read_csv(csv_path)

# 열 이름(가독성을 위해 변수 할당)
COL_O  = "sellday_open_per_buyday_close"          # O(t+1) / C(t)
COL_MH = "sellday_morning_high_per_buyday_close"  # MH(t+1) / C(t)
COL_MC = "sellday_morning_close_per_buyday_close" # MC(t+1) / C(t)

# 2. x값의 범위 지정 (예: 0.9 ~ 1.2 구간을 50 등분)
x_values = np.linspace(0.7, 1.3, 100)

# 결과를 담을 리스트
expected_returns = []

for x in x_values:
    # f(x) = P(MH >= x)
    f_x = (df[COL_MH] >= x).mean()
    
    # g(x) = P(O >= x)
    g_x = (df[COL_O]  >= x).mean()
    
    # E[O | O >= x]
    mask_o = (df[COL_O] >= x)
    if mask_o.any():  # 조건을 만족하는 행이 하나라도 있다면
        E_O_cond = df.loc[mask_o, COL_O].mean()
    else:
        # 조건을 만족하는 데이터가 없을 경우 처리 (예: NaN)
        E_O_cond = np.nan
    
    # h(x) = E[MC | MH < x]
    mask_mh = (df[COL_MH] < x)
    if mask_mh.any():
        h_x = df.loc[mask_mh, COL_MC].mean()
    else:
        h_x = np.nan
    
    # E(Y/C(t)) = g(x)*E[O|O>=x] + (f(x)-g(x))*x + (1-f(x))*h(x)
    # (Case1: 시가>=x, Case2: 시가는 x 미만이나 MH>=x, Case3: MH<x)
    # 여기서 NaN이 껴 있을 경우를 어떻게 처리할지는 상황에 따라 결정
    if np.isnan(E_O_cond) or np.isnan(h_x):
        # 예시로, NaN이 나오면 해당 x의 기대값도 NaN 처리
        e_ratio = np.nan
    else:
        e_ratio = g_x * E_O_cond + (f_x - g_x)*x + (1 - f_x)*h_x
    
    # 기대 수익률 = E(Y/C(t)) - 1
    e_return = e_ratio - 1 if e_ratio is not np.nan else np.nan
    expected_returns.append(e_return)

# 3. 그래프 시각화
plt.plot(x_values, expected_returns)
plt.xlabel('x (목표가 / 전일 종가 비율)')
plt.ylabel('기대수익률 (E(Return))')
plt.title('x에 따른 기대수익률')
plt.show()
