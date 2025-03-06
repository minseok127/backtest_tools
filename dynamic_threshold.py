import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import threading
import concurrent.futures

from itertools import product

def backtest_strategy(data, data_1500, n_days=55):
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
    data = data.sort_values(by='date')

    data_1500['date'] = pd.to_datetime(data_1500['date'], format='%Y%m%d')
    data_1500 = data_1500.sort_values(by='date')

    dates = data['date'].unique()
    cumulative_return_original = 1
    max_cumulative_return_original = 1
    mdd_original = 0
    daily_returns = []
    cumulative_returns = []
    avg_candidates_count_last = None
    std_candidates_count_last = None

    # 각 조건별 기하평균을 위한 리스트
    returns_above_std = []
    returns_within_std = []
    returns_below_std = []
    returns_below_avg_minus_std = []

    data_len_list = []
    results = []  # 종목 수와 수익률을 저장할 리스트

    for i in range(len(dates)):
        current_date = dates[i]

        current_data = data[data['date'] == current_date].copy()
        current_data_1500 = data_1500[data_1500['date'] == current_date].copy()
        
        portfolio = []  # 포트폴리오 리스트 초기화 (각 날짜마다 새로 시작)

        # 가장 작은 buy_order_time을 찾고 해당 시간대 종목 수를 data_len_list에 추가
        min_buy_order_time = 1300
        candidate_stocks = current_data[current_data['buy_order_time'] == min_buy_order_time]
        num_candidates = len(candidate_stocks)
        data_len_list.append(num_candidates)

        print(current_date)

        if i > n_days and len(data_len_list[i - n_days - 1:i - 1]) > 0:
            average_candidates_count = np.mean(data_len_list[i - n_days - 1:i - 1])
            std_candidates_count = np.std(data_len_list[i - n_days - 1:i - 1])
        else:
            average_candidates_count = 0
            std_candidates_count = 0

        avg_candidates_count_last = average_candidates_count
        std_candidates_count_last = std_candidates_count

        """
        print(f"{avg_candidates_count_last - std_candidates_count_last} ... {avg_candidates_count_last} ... {avg_candidates_count_last + std_candidates_count_last}")
        print(num_candidates)
        print(candidate_stocks)
        """

        if num_candidates >= average_candidates_count + std_candidates_count:
            sort_criterion = 'close_prev_day_close'
            category = "above_std"
        elif num_candidates >= average_candidates_count:
            sort_criterion = 'close_prev_day_close'
            category = "within_std"
        elif num_candidates >= average_candidates_count - std_candidates_count:
            sort_criterion = 'high_low'
            category = "below_std"
        else:
            sort_criterion = 'high_low'
            category = "below_avg_minus_std"

        category_thresholds = {
            "above_std": {
                "time_thresholds": [1315, 1330, 1345, 1400, 1415, 1430, 1500],
                "min_close_thresholds": [1.07, 1.07, 1.025, 1.025, 1.025, 1.025, 1.025],
                "max_close_thresholds": [1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25],
                "close_thresholds_for_alloc": [1.07, 1.07, 1.07, 1.07, 1.07, 1.07, 1.07],
                "alloc_below_thresholds": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                "alloc_above_thresholds": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
            },
            "within_std": {
                "time_thresholds": [1315, 1330, 1345, 1400, 1415, 1430, 1500],
                "min_close_thresholds": [1.07, 1.07, 1.025, 1.025, 1.025, 1.025, 1.025],
                "max_close_thresholds": [1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25],
                "close_thresholds_for_alloc": [1.07, 1.07, 1.07, 1.07, 1.07, 1.07, 1.07],
                "alloc_below_thresholds": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                "alloc_above_thresholds": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
            },
            "below_std": {
                "time_thresholds": [1315, 1330, 1345, 1400, 1415, 1430, 1500],
                "min_close_thresholds": [1.025, 1.025, 1.025, 1.025, 1.025, 1.025, 1.025],
                "max_close_thresholds": [1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25],
                "close_thresholds_for_alloc": [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
                "alloc_below_thresholds": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                "alloc_above_thresholds": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
            },
            "below_avg_minus_std": {
                "time_thresholds": [1315, 1330, 1345, 1400, 1415, 1430, 1500],
                "min_close_thresholds": [1.025, 1.025, 1.025, 1.025, 1.025, 1.025, 1.025],
                "max_close_thresholds": [1.25, 1.25, 1.25, 1.25, 1.25, 1.25, 1.25],
                "close_thresholds_for_alloc": [1.05, 1.05, 1.05, 1.05, 1.05, 1.05, 1.05],
                "alloc_below_thresholds": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
                "alloc_above_thresholds": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
            },
        }
            
        # 카테고리에 맞는 thresholds 불러오기
        thresholds = category_thresholds[category]
        time_thresholds = thresholds["time_thresholds"]
        min_close_thresholds = thresholds["min_close_thresholds"]
        max_close_thresholds = thresholds["max_close_thresholds"]   
        close_thresholds_for_alloc = thresholds["close_thresholds_for_alloc"]
        alloc_below_thresholds = thresholds["alloc_below_thresholds"]
        alloc_above_thresholds = thresholds["alloc_above_thresholds"]

        available_portion = 1.0

        # 각 시간대별 for문 실행
        time_list = []
        for hour in [13, 14, 15]:
            for minute in range(0, 60, 5):
                t = hour * 100 + minute
                if t > 1500:  # 15:00를 초과하면 그만
                    break
                time_list.append(t)
        
        for current_time in time_list:
            eligible_stocks = None
                
            # 포트폴리오 업데이트: 취소 시간이 도달한 종목 제거
            # current_time이 cancel_time을 넘어가야 취소함. 그전에는 포트에 체결 안된상태로 남아있도록함
            to_remove = []
            for item in portfolio:
                if item["buy_success"] == 0 and item["cancel_time"] == current_time:
                    available_portion += item["allocated_portion"]
                    to_remove.append(item)

            # 포트폴리오에서 제거
            for r in to_remove:
                portfolio.remove(r)

            thresdhold_time_idx = 0

            # 첫 번째로 충족하는 시간대 조건만 적용
            for idx, time_threshold in enumerate(time_thresholds):
                if current_time <= time_threshold:
                    eligible_stocks = current_data[
                        (current_data['buy_order_time'] == current_time) &
                        (current_data['close_prev_day_close'] >= min_close_thresholds[idx]) &
                        (current_data['close_prev_day_close'] <= max_close_thresholds[idx])
                    ]
                    thresdhold_time_idx = idx
                    break  # 조건을 충족하는 첫 번째 시간대에서 멈춤

            close_threshold_for_alloc = close_thresholds_for_alloc[thresdhold_time_idx]
            alloc_below = alloc_below_thresholds[thresdhold_time_idx]
            alloc_above = alloc_above_thresholds[thresdhold_time_idx]
            
            # 필터링 후 후보 종목이 없으면 다음 시간대로 건너뛰기
            if eligible_stocks is None or eligible_stocks.empty:
                continue

            # 설정된 정렬 기준에 따라 eligible_stocks 정렬
            if sort_criterion is not None:
                eligible_stocks = eligible_stocks.sort_values(by=sort_criterion, ascending=False)
            else:
                continue

            # 포트폴리오에 종목 추가 (num_stocks를 초과하지 않는 범위 내)
            for _, stock in eligible_stocks.iterrows():
                # 중복 여부를 확인하여 포트폴리오에 추가
                if any(p['code'] == stock['code'] for p in portfolio):
                    continue  # 이미 있다면 건너뜀
                
                # threshold 따라 할당 희망 비중
                close_ratio = stock['close_prev_day_close']
                if close_ratio < close_threshold_for_alloc:
                    proposed_alloc = alloc_below
                else:
                    proposed_alloc = alloc_above

                # 주문 들어감 → 비중 차감(실패/성공 구분 없이)
                alloc_for_this_order = min(proposed_alloc, available_portion)
                
                # 실제로 0 이하라면 더 이상 주문 불가
                if alloc_for_this_order <= 0:
                    continue

                available_portion -= alloc_for_this_order

                portfolio.append({'code': stock['code'], 'result': stock['result'], 'high_low': stock['high_low'], 
                                      'close_prev_day_close': stock['close_prev_day_close'], 'buy_order_time': stock['buy_order_time'], 
                                      'buy_price': stock['buy_price'], 'sell_price': stock['sell_price'], 
                                      'cancel_time': stock['cancel_time'], 'buy_success': stock['buy_success'],
                                      'allocated_portion': alloc_for_this_order})

        to_remove = []
        for item in portfolio:
            if item["buy_success"] == 0:
                available_portion += item["allocated_portion"]
                to_remove.append(item)

        # 포트폴리오에서 제거
        for r in to_remove:
            portfolio.remove(r)

         # 매수가 되지 않은 종목 수만큼 추가 매수를 시도
        if available_portion > 0:
            additional_candidates = current_data_1500[~current_data_1500['code'].isin([p['code'] for p in portfolio])]

            if sort_criterion is not None:
                additional_candidates = additional_candidates.sort_values(by=sort_criterion, ascending=False)
            
            for _, stock in additional_candidates.iterrows():
                # 중복 여부를 확인하여 포트폴리오에 추가
                if any(p['code'] == stock['code'] for p in portfolio):
                    continue  # 이미 있다면 건너뜀
                
                # threshold 따라 할당 희망 비중
                proposed_alloc = 0.25

                alloc_for_this_order = min(proposed_alloc, available_portion)

                # 실제로 0 이하라면 더 이상 주문 불가
                if alloc_for_this_order <= 0:
                    continue

                available_portion -= alloc_for_this_order

                portfolio.append({'code': stock['code'], 'result': stock['result'], 'high_low': stock['high_low'], 
                                      'close_prev_day_close': stock['close_prev_day_close'], 'buy_order_time': stock['buy_order_time'], 
                                      'buy_price': stock['buy_price'], 'sell_price': stock['sell_price'],
                                      'allocated_portion': alloc_for_this_order})

        # 포트폴리오가 확정되면 daily_return_original 계산
        daily_return_original = 0

        # 포트폴리오 내 확정된 종목들에 대해 수익률 계산
        for p in portfolio:
            code = p['code']
            result = p['result']
            buy_order_time = p['buy_order_time']
            buy_price = p['buy_price']
            sell_price = p['sell_price']
            allocated_portion = p['allocated_portion']

            if result != 0:
                daily_return_original += result * allocated_portion
                
            # 각 종목에 대한 정보 출력
            print(code)
            print(f"buy: {buy_price}")
            print(f"sell: {sell_price}")
            print(f"buy_order_time: {buy_order_time}")
            print(f"result: {result}")
            print(category)

        # 매수가 되지 않은 비중만큼 수익률 1을 추가
        daily_return_original += available_portion

        cumulative_return_original *= daily_return_original
        daily_returns.append(daily_return_original)
        cumulative_returns.append(cumulative_return_original)

        if cumulative_return_original < 0:
            print(daily_return_original)
            exit(1)

        # 종목 수와 수익률을 결과 리스트에 저장
        results.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'result': daily_return_original,
        })

        # 각 카테고리에 따른 daily_return_original 저장
        if category == "above_std":
            returns_above_std.append(daily_return_original)
        elif category == "within_std":
            returns_within_std.append(daily_return_original)
        elif category == "below_std":
            returns_below_std.append(daily_return_original)
        else:
            returns_below_avg_minus_std.append(daily_return_original)

        print(f"Date: {current_date.strftime('%Y-%m-%d')}, Daily Return: {daily_return_original:.4f}, Cumulative Return: {cumulative_return_original:.4f}")

        max_cumulative_return_original = max(max_cumulative_return_original, cumulative_return_original)
        drawdown_original = (max_cumulative_return_original - cumulative_return_original) / max_cumulative_return_original
        mdd_original = max(mdd_original, drawdown_original)

    if len(daily_returns) > 0:
        geometric_mean_return = np.exp(np.mean(np.log(daily_returns)))
    else:
        geometric_mean_return = 1

    avg_candidates_count_last = np.mean(data_len_list[-n_days:])
    std_candidates_count_last = np.std(data_len_list[-n_days:])
    print(f"\nFinal Average Candidates Count: {avg_candidates_count_last:.4f}")
    print(f"Final Std Candidates Count: {std_candidates_count_last:.4f}")
    print(f"{avg_candidates_count_last - std_candidates_count_last} ... {avg_candidates_count_last} ... {avg_candidates_count_last + std_candidates_count_last}")

    # 누적 수익률 그래프 그리기
    plt.figure(figsize=(12, 6))
    plt.plot(dates, cumulative_returns, label='Cumulative Return')
    plt.title('Cumulative Return Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 로그 변환된 누적 수익률 그래프 그리기
    log_cumulative_returns = np.log(cumulative_returns)
    plt.figure(figsize=(12, 6))
    plt.plot(dates, log_cumulative_returns, label='Log Transformed Cumulative Return', color='orange')
    plt.title('Log Transformed Cumulative Return Over Time')
    plt.xlabel('Date')
    plt.ylabel('Log(Cumulative Return)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 각 리스트에 대해 기하평균 계산
    def geometric_mean(returns):
        if len(returns) > 0:
            return np.exp(np.mean(np.log(returns)))
        return 1

    geom_mean_above_std = geometric_mean(returns_above_std)
    geom_mean_within_std = geometric_mean(returns_within_std)
    geom_mean_below_std = geometric_mean(returns_below_std)
    geom_mean_below_avg_minus_std = geometric_mean(returns_below_avg_minus_std)

    # 각 카테고리별 기하평균 출력
    print(f"\nGeometric Mean for len(current_data) >= average_candidates_count + std_candidates_count: {geom_mean_above_std:.4f}, len: {len(returns_above_std)}")
    print(f"Geometric Mean for len(current_data) between average_candidates_count and average_candidates_count + std_candidates_count: {geom_mean_within_std:.4f}, len: {len(returns_within_std)}")
    print(f"Geometric Mean for len(current_data) between average_candidates_count - std_candidates_count and average_candidates_count: {geom_mean_below_std:.4f}, len: {len(returns_below_std)}")
    print(f"Geometric Mean for len(current_data) < average_candidates_count - std_candidates_count: {geom_mean_below_avg_minus_std:.4f}, len: {len(returns_below_avg_minus_std)}")

    # 최종 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame(results)

    # 결과를 CSV 파일로 저장
    
    results_df.to_csv('ss_results.csv', index=False)
    print("ss_results.csv 파일이 생성되었습니다.")
    
    return cumulative_return_original, mdd_original, geometric_mean_return

# 데이터 불러오기
data = pd.read_csv("C:\\Users\\Public\\Desktop\\kosdaq_data_ss_5min.csv")
data_1500 = pd.read_csv("C:\\Users\\Public\\Desktop\\kosdaq_data_ss_30min_1500.csv")

# 백테스트 수행
cumulative_return, mdd, geometric_mean_return = backtest_strategy(data, data_1500)

print(f"\nFinal Cumulative Return: {cumulative_return:.4f}")
print(f"Final MDD: {mdd:.4f}")
print(f"Final Geometric Mean Return: {geometric_mean_return:.4f}")