############################################################
# dp_optimal_sell.py
#
# 전제: CSV 파일은 "preopen=0" 상태인 데이터만 담고 있어,
#       preopen_case라는 필드가 없다.
#
#       (9:30,10:00,10:30,11:00) 각 case_xxx=[1..4]
#       전이확률은 build_transition_probs_main()로 계산
#
#
# DP 순서:
#   1) (10:30->11:00), (10:00->10:30), (9:30->10:00)
#   2) (0->9:30)
############################################################

import pandas as pd
import numpy as np

def main():
    # 1) CSV 로드
    df = pd.read_csv("sell_data_with_case.csv") 
    # - 여기에는
    #   date, prevday_close, sellday_open, sellday_900_930_close, sellday_930_1000_close, ...
    #   case_930, case_1000, case_1030, case_1100,
    #   ratio_930_from_prevclose, ratio_1000_from_prevclose, ...
    #   등이 들어있고, preopen_case는 없음.

    # 날짜 정렬
    unique_dates = sorted(df["date"].unique())

    # DP 결과 저장
    all_results = []

    # (A) 정의
    time_preopen = 0
    cases_preopen = [0]          # preopen=0 하나만 존재
    times_main = [930, 1000, 1030, 1100]
    cases_main = [1,2,3,4]

    target_candidates_preopen = np.arange(-0.1, 0.1, 0.01)
    target_candidates = np.arange(0, 0.1, 0.005)

    for current_date in unique_dates:
        past_data = df[df["date"]<current_date]
        if len(past_data)==0:
            # 과거데이터 전무 -> 기본 DP
            dp, best_target = make_default_dp(time_preopen, cases_preopen, times_main, cases_main)
        else:
            # 1) build_transition_probs_main: 9:30->10:00, 10:00->10:30, 10:30->11:00
            trans_main = build_transition_probs_main(past_data)

            # 2) build_transition_probs_0_930: (0->9:30)
            trans_0_930 = build_transition_probs_0_930(past_data)

            # 3) 체결확률: preopen->9:30 / 9:30->10:00 / ...
            hit_prob = compute_hit_probabilities(
                past_data,
                time_preopen, cases_preopen,
                times_main, cases_main,
                target_candidates_preopen, target_candidates
            )

            # 4) 마지막 시점(11:00) 종가매도 수익률
            final_ratio_map = {}
            for c in cases_main:
                sub = past_data[past_data["case_1100"]==c]
                if len(sub)==0:
                    final_ratio_map[c] = 0.0
                else:
                    final_ratio_map[c] = sub["ratio_1100_from_prevclose"].mean()

            # 5) DP
            dp, best_target = run_dp(
                time_preopen, cases_preopen,
                times_main, cases_main,
                target_candidates_preopen, target_candidates,
                trans_main, trans_0_930,
                hit_prob,
                final_ratio_map,
                past_data
            )

        # (B) DP 결과 저장
        # preopen
        all_results.append({
            "date": current_date,
            "time": 0,
            "case": 0,
            "expected_return": dp[0][0],
            "best_target_ratio": best_target[0][0]
        })
        # main
        for t in times_main:
            for c in cases_main:
                all_results.append({
                    "date": current_date,
                    "time": t,
                    "case": c,
                    "expected_return": dp[t][c],
                    "best_target_ratio": best_target[t][c]
                })

    # 저장
    res_df = pd.DataFrame(all_results)
    res_df.to_csv("dp_result.csv", index=False)
    print("dp_result.csv 저장 완료")


############################################################
# DP 기본값
############################################################
def make_default_dp(time_preopen, cases_preopen, times_main, cases_main):
    dp={}
    best_target={}
    # preopen
    dp[time_preopen]={}
    best_target[time_preopen]={}
    for c0 in cases_preopen:
        dp[time_preopen][c0]=0.0
        best_target[time_preopen][c0]=None

    # main
    for t in times_main:
        dp[t]={}
        best_target[t]={}
        for c in cases_main:
            dp[t][c]=0.0
            best_target[t][c]=None
    return dp,best_target


############################################################
# 9:30->10:00 / 10:00->10:30 / 10:30->11:00 전이확률
############################################################
def build_transition_probs_main(past_data):
    times = [930, 1000, 1030]
    cases = [1,2,3,4]
    counts = {}
    for t in times:
        counts[t] = {c:{c2:0 for c2 in cases} for c in cases}

    for _,row in past_data.iterrows():
        c930 = row["case_930"]
        c1000= row["case_1000"]
        c1030= row["case_1030"]
        c1100= row["case_1100"]
        # 930->1000
        counts[930][c930][c1000]+=1
        # 1000->1030
        counts[1000][c1000][c1030]+=1
        # 1030->1100
        counts[1030][c1030][c1100]+=1

    trans={}
    for t in times:
        trans[t]={}
        for c in cases:
            total= sum(counts[t][c].values())
            if total==0:
                trans[t][c]={c2:1.0/len(cases) for c2 in cases}
            else:
                trans[t][c]={}
                for c2 in cases:
                    trans[t][c][c2] = counts[t][c][c2]/total
    return trans


############################################################
# (0->9:30) 전이확률: preopen 케이스마다 애초에 csv파일 구분됨
############################################################
def build_transition_probs_0_930(past_data):
    # case_930 in [1,2,3,4]
    cases_930 = [1,2,3,4]
    count_dict = {c2:0 for c2 in cases_930}

    for _, row in past_data.iterrows():
        c930 = row["case_930"]
        if c930 in cases_930:
            count_dict[c930]+=1

    total = sum(count_dict.values())
    trans_0_930={}
    # key=> preopen=0 (dict in dict)
    trans_0_930[0]={}
    if total==0:
        # 균등
        for c2 in cases_930:
            trans_0_930[0][c2] = 1.0/len(cases_930)
    else:
        for c2 in cases_930:
            trans_0_930[0][c2] = count_dict[c2]/total

    return trans_0_930


############################################################
# 체결 확률 (preopen & main)
############################################################
def compute_hit_probabilities(
    past_data,
    time_preopen, cases_preopen,
    times_main, cases_main,
    target_candidates_preopen, target_candidates
):
    hit_prob={}

    # preopen
    hit_prob[time_preopen]={}
    for c0 in cases_preopen:
        hit_prob[time_preopen][c0]={}
        for r in target_candidates_preopen:
            hit_prob[time_preopen][c0][r] = calc_preopen_hit_prob(past_data, r)

    # main
    for t in times_main:
        hit_prob[t]={}
        for c in cases_main:
            hit_prob[t][c]={}
            for r in target_candidates:
                p = calc_hit_probability_sub(
                    past_data, t, c, r
                )
                hit_prob[t][c][r]=p

    return hit_prob


def calc_preopen_hit_prob(past_data, r):
    if len(past_data)==0:
        return 0.0

    hits=0
    for _, row in past_data.iterrows():
        prev_close = row["prevday_close"]
        high_0930  = row["sellday_900_930_high"]

        target_price = prev_close*(1.0+r)
        if high_0930>=target_price:
            hits+=1
    return hits/len(past_data)

def calc_hit_probability_sub(past_data, time_start, case_label, target_ratio):
    """
    (9:30->10:00) 등에서 분봉 고가>= (직전분봉 close)*(1+target_ratio)
    """
    if time_start==930:
        curr_close_col="sellday_900_930_close"
        next_high_col ="sellday_930_1000_high"
    elif time_start==1000:
        curr_close_col="sellday_930_1000_close"
        next_high_col ="sellday_1000_1030_high"
    elif time_start==1030:
        curr_close_col="sellday_1000_1030_close"
        next_high_col ="sellday_1030_1100_high"
    else:
        return 0.0

    sub = past_data[past_data[f"case_{time_start}"]==case_label]
    if len(sub)==0:
        return 0.0

    hits=0
    for _, row in sub.iterrows():
        curr_price = row[curr_close_col]
        high_price = row[next_high_col]
        target_price= curr_price*(1.0+target_ratio)
        if high_price>=target_price:
            hits+=1
    return hits/len(sub)


############################################################
# run_dp: 
#   1) (10:30->11:00), (10:00->10:30), (9:30->10:00)
#   2) (0->9:30)
############################################################
def run_dp(
    time_preopen, cases_preopen,
    times_main, cases_main,
    target_candidates_preopen, target_candidates,
    trans_main, trans_0_930,
    hit_prob,
    final_ratio_map,
    past_data
):
    # DP init
    dp={}
    best_target={}

    # preopen=0
    dp[time_preopen]={}
    best_target[time_preopen]={}
    for c0 in cases_preopen:
        dp[time_preopen][c0]= -9999.0
        best_target[time_preopen][c0]= None

    # main
    for t in times_main:
        dp[t]={}
        best_target[t]={}
        if t==1100:
            # 종가 매도
            for c in cases_main:
                dp[t][c] = final_ratio_map[c]
                best_target[t][c] = None # 11시에는 어차피 목표가 재설정 안하고 바로 매도함
        else:
            for c in cases_main:
                dp[t][c] = -9999.0
                best_target[t][c] = None

    # (A) main loop: (10:30->11:00), (10:00->10:30), (9:30->10:00)
    main_order = [(1030,1100), (1000,1030), (930,1000)]
    for (tc,tn) in main_order:
        for c in cases_main:
            ratio_t = get_mean_ratio_of_time_case(past_data, tc, c)
            best_val=-9999.0
            best_r=None
            for r in target_candidates:
                p_hit= hit_prob[tc][c][r]
                # 체결시 payoff = ratio_t + r + ratio_t*r
                # (1 + ratio_t) * (1 + r) - 1
                payoff_if_hit= ratio_t + r + ratio_t*r

                # 미체결 => sum_{c2}(trans_main[tc][c][c2]* dp[tn][c2])
                exp_if_not_hit=0.0
                for c2, p_c2 in trans_main[tc][c].items():
                    exp_if_not_hit+= p_c2*dp[tn][c2]

                exp_val = p_hit*payoff_if_hit + (1-p_hit)*exp_if_not_hit
                if exp_val>best_val:
                    best_val=exp_val
                    best_r=r
            dp[tc][c]=best_val
            best_target[tc][c]=best_r

    # (B) preopen(0)->9:30
    for c0 in cases_preopen:
        best_val=-9999.0
        best_r=None
        for r in target_candidates_preopen:
            p_hit= hit_prob[0][c0][r]
            # 체결시 payoff를 가정
            # 예: 시가체결시 payoff = open/prev_close -1
            #     or 목표가체결 => r
            payoff_if_hit = get_preopen_avg_payoff_if_hit(past_data, r)

            # 미체결 => (9:30, c2)
            # 전이확률 trans_0_930[c0][c2]
            exp_if_not_hit=0.0
            for c2, p_c2 in trans_0_930[c0].items():
                exp_if_not_hit += p_c2*dp[930][c2]

            exp_val = p_hit*payoff_if_hit + (1-p_hit)*exp_if_not_hit
            if exp_val>best_val:
                best_val=exp_val
                best_r=r

        dp[0][c0]=best_val
        best_target[0][c0]=best_r

    return dp,best_target


def get_mean_ratio_of_time_case(past_data, time, case_label):
    """
    time=930 -> ratio_930_from_prevclose, ...
    """
    if time==0:
        return 0.0
    col_ratio= f"ratio_{time}_from_prevclose"
    col_case = f"case_{time}"
    if col_ratio not in past_data.columns or col_case not in past_data.columns:
        return 0.0

    sub = past_data[past_data[col_case]==case_label]
    if len(sub)==0:
        return 0.0
    return sub[col_ratio].mean()

def get_preopen_avg_payoff_if_hit(past_data, r):
    """
    preopen 체결시 payoff
    - 시가>=목표가 => payoff=시가/prev_close -1
    - else => payoff=목표가/prev_close -1
    (체결된 경우만 수집)
    """
    if len(past_data)==0:
        return 0.0

    payoff_list=[]
    for _, row in past_data.iterrows():
        prev_close = row["prevday_close"]
        open_p     = row["sellday_open"]
        high_0930  = row["sellday_900_930_high"]

        target_price = prev_close*(1.0+r)
        # 시가>=목표가 -> payoff= open_p/prev_close -1
        if open_p>=target_price:
            payoff = (open_p/prev_close)-1
        elif high_0930>=target_price:
            # 목표가체결
            payoff = r
        else:
            # 체결안됨 -> skip
            continue

        payoff_list.append(payoff)

    if len(payoff_list)==0:
        return 0.0
    return np.mean(payoff_list)


if __name__=="__main__":
    main()
