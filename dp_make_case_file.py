############################################################
# make_case_file.py
#
# (1) sell_data.csv 로드
# (2) 분봉별 변화율, 변동성 컬럼 생성
# (3) 날짜 오름차순 기준, '이전 날짜'들의 통계(평균/표준편차)만 이용해 case 분류
# (4) 결과: sell_data_with_case.csv 파일 생성
############################################################

import pandas as pd
import numpy as np

def classify_case(ratio, ratio_mean, ratio_std, vol, vol_mean, vol_std):
    """
    사용자 정의 Case 분류 규칙:
      - ratio >= mean + 2*std -> case1
      - ratio <= mean - 2*std -> case2
      - 그 외이고 vol >= vol_mean + 2*vol_std -> case3
      - 나머지 -> case4
    """
    high_ratio = ratio_mean + 2.0 * ratio_std
    low_ratio  = ratio_mean - 2.0 * ratio_std
    high_vol   = vol_mean + 2.0 * vol_std

    if ratio >= high_ratio:
        return 1
    elif ratio <= low_ratio:
        return 2
    elif vol >= high_vol:
        return 3
    else:
        return 4

def main():
    # 1) 원본 CSV 로드
    df = pd.read_csv("sell_data.csv")
    # 예시 스키마:
    # date, code, prevday_close, sellday_open,
    # sellday_900_930_high, sellday_900_930_close, sellday_900_930_volatility,
    # sellday_930_1000_high, sellday_930_1000_close, sellday_930_1000_volatility,
    # sellday_1000_1030_high, sellday_1000_1030_close, sellday_1000_1030_volatility,
    # sellday_1030_1100_high, sellday_1030_1100_close, sellday_1030_1100_volatility,
    # sellday_1100_close

    # 2) 날짜 오름차순 정렬
    df = df.sort_values(by="date").reset_index(drop=True)

    # 3) 분봉별 전날 종가 대비 변화율(ratio) 컬럼 생성
    df["ratio_930_from_prevclose"]  = df["sellday_900_930_close"]  / df["prevday_close"] - 1
    df["ratio_1000_from_prevclose"] = df["sellday_930_1000_close"] / df["prevday_close"] - 1
    df["ratio_1030_from_prevclose"] = df["sellday_1000_1030_close"] / df["prevday_close"] - 1
    df["ratio_1100_from_prevclose"] = df["sellday_1030_1100_close"] / df["prevday_close"] - 1

    # 변동성도 시점별로 구분
    df["vol_930"]  = df["sellday_900_930_volatility"]
    df["vol_1000"] = df["sellday_930_1000_volatility"]
    df["vol_1030"] = df["sellday_1000_1030_volatility"]
    df["vol_1100"] = df["sellday_1030_1100_volatility"]

    # 4) case 분류: 날짜별로, '해당 날짜보다 이전' 데이터만 사용
    case_930_list   = []
    case_1000_list  = []
    case_1030_list  = []
    case_1100_list  = []

    # date가 정수(20250306)라면 정수 비교, 문자열이면 문자열 비교
    # 여기서는 int형으로 가정
    for i in range(len(df)):
        row = df.iloc[i]
        current_date = row["date"]  # 예: 20250306 (int)
        print(current_date)

        # (A) 과거(이전 날짜)만 필터링
        #     same-date는 포함 X
        past_data = df[df["date"] < current_date]
        if len(past_data) == 0:
            # 과거가 없으면 통계 불가 -> 기본값
            r930_mean, r930_std  = 0.0, 1e-9
            r1000_mean, r1000_std= 0.0, 1e-9
            r1030_mean, r1030_std= 0.0, 1e-9
            r1100_mean, r1100_std=0.0, 1e-9

            v930_mean, v930_std  = 0.0, 1e-9
            v1000_mean, v1000_std= 0.0, 1e-9
            v1030_mean, v1030_std=0.0, 1e-9
            v1100_mean, v1100_std=0.0, 1e-9
        else:
            # ratio 평균/표준편차
            r930_mean  = past_data["ratio_930_from_prevclose"].mean()
            r930_std   = past_data["ratio_930_from_prevclose"].std() if past_data["ratio_930_from_prevclose"].std() > 0 else 1e-9

            r1000_mean = past_data["ratio_1000_from_prevclose"].mean()
            r1000_std  = past_data["ratio_1000_from_prevclose"].std() if past_data["ratio_1000_from_prevclose"].std() > 0 else 1e-9

            r1030_mean = past_data["ratio_1030_from_prevclose"].mean()
            r1030_std  = past_data["ratio_1030_from_prevclose"].std() if past_data["ratio_1030_from_prevclose"].std() > 0 else 1e-9

            r1100_mean = past_data["ratio_1100_from_prevclose"].mean()
            r1100_std  = past_data["ratio_1100_from_prevclose"].std() if past_data["ratio_1100_from_prevclose"].std() > 0 else 1e-9

            # vol 평균/표준편차
            v930_mean  = past_data["vol_930"].mean()
            v930_std   = past_data["vol_930"].std() if past_data["vol_930"].std() > 0 else 1e-9

            v1000_mean = past_data["vol_1000"].mean()
            v1000_std  = past_data["vol_1000"].std() if past_data["vol_1000"].std() > 0 else 1e-9

            v1030_mean = past_data["vol_1030"].mean()
            v1030_std  = past_data["vol_1030"].std() if past_data["vol_1030"].std() > 0 else 1e-9

            v1100_mean = past_data["vol_1100"].mean()
            v1100_std  = past_data["vol_1100"].std() if past_data["vol_1100"].std() > 0 else 1e-9

        # (B) 현재 행의 ratio, vol
        ratio_930_from_prevclose = row["ratio_930_from_prevclose"]
        vol_930   = row["vol_930"]

        ratio_1000_from_prevclose= row["ratio_1000_from_prevclose"]
        vol_1000  = row["vol_1000"]

        ratio_1030_from_prevclose= row["ratio_1030_from_prevclose"]
        vol_1030  = row["vol_1030"]

        ratio_1100_from_prevclose= row["ratio_1100_from_prevclose"]
        vol_1100  = row["vol_1100"]

        # (C) Case 분류
        c930  = classify_case(ratio_930_from_prevclose,  r930_mean,  r930_std,
                              vol_930,    v930_mean,  v930_std)
        c1000 = classify_case(ratio_1000_from_prevclose, r1000_mean, r1000_std,
                              vol_1000,   v1000_mean, v1000_std)
        c1030 = classify_case(ratio_1030_from_prevclose, r1030_mean, r1030_std,
                              vol_1030,   v1030_mean, v1030_std)
        c1100 = classify_case(ratio_1100_from_prevclose, r1100_mean, r1100_std,
                              vol_1100,   v1100_mean, v1100_std)

        case_930_list.append(c930)
        case_1000_list.append(c1000)
        case_1030_list.append(c1030)
        case_1100_list.append(c1100)

    # 5) case 컬럼 추가
    df["case_930"]  = case_930_list
    df["case_1000"] = case_1000_list
    df["case_1030"] = case_1030_list
    df["case_1100"] = case_1100_list

    # 6) 결과 저장
    df.to_csv("sell_data_with_case.csv", index=False)
    print("Case 분류 완료 --> sell_data_with_case.csv")

if __name__ == "__main__":
    main()
