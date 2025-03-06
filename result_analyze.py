import pandas as pd
import numpy as np

def analyze_field(series, field_name):
    total_count   = len(series)
    profit_mask   = series > 1.0
    loss_mask     = series < 1.0
    neutral_mask  = series == 1.0

    profit_count  = profit_mask.sum()
    loss_count    = loss_mask.sum()
    neutral_count = neutral_mask.sum()

    sum_profit = (series[profit_mask] - 1.0).sum()
    sum_loss   = (1.0 - series[loss_mask]).sum()

    win_probability = profit_count / total_count if total_count > 0 else 0.0
    avg_profit = sum_profit / profit_count if profit_count > 0 else 0.0
    avg_loss = sum_loss / loss_count if loss_count > 0 else 0.0
    profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0.0

    # 기하평균 계산: 모든 수익률의 로그 평균에 대해 지수함수 적용
    geom_mean = np.exp(np.log(series).mean()) if total_count > 0 else 0.0

    print(f"---- {field_name} 통계 ----")
    print(f"전체 트레이드 수: {total_count}")
    print(f" - 수익 트레이드 수: {profit_count}")
    print(f" - 손실 트레이드 수: {loss_count}")
    print(f" - 보합(1.0) 트레이드 수: {neutral_count}")
    print(f"수익 확률: {win_probability * 100:.2f}%")
    print(f"평균 이익 (수익난 경우): {avg_profit:.6f}")
    print(f"평균 손실 (손실난 경우): {avg_loss:.6f}")
    print(f"손익비: {profit_loss_ratio:.4f}")
    print(f"수익률 기하평균: {geom_mean:.6f}")
    print()

def analyze_timing(series, field_name):
    # 타이밍 값에 대한 매핑: 0: 시가, 1: 11시 이전, 2: 11시 이후, 3: 시장가매도
    mapping = {0: "시가", 1: "11시 이전", 2: "11시 이후", 3: "시장가매도"}
    total = len(series)
    print(f"---- {field_name} 타이밍 비율 ----")
    for key, label in mapping.items():
        count = (series == key).sum()
        ratio = count / total if total > 0 else 0.0
        print(f"{label} ({key}): {ratio * 100:.2f}%")
    print()

if __name__ == "__main__":
    # CSV 파일 읽기 (CSV 파일은 date, result, result_timing, result_new, result_new_timing 필드를 포함)
    df = pd.read_csv("C:\\Users\\Public\\Desktop\\kosdaq_data_stat_5min.csv")

    # result 필드와 그에 해당하는 타이밍(result_timing) 분석
    analyze_field(df["result"], "result")
    analyze_timing(df["result_timing"], "result_timing")

    # result_new 필드와 그에 해당하는 타이밍(result_new_timing) 분석
    analyze_field(df["result_new"], "result_new")
    analyze_timing(df["result_new_timing"], "result_new_timing")
