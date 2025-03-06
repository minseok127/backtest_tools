def main():
    numbers = set()  # 중복을 허용하지 않는 집합으로 변경

    # 여러 줄로부터 입력을 받음
    print("여러 줄에 걸쳐서 숫자를 입력하세요. 입력이 끝나면 빈 줄을 입력하세요.")
    while True:
        line = input()
        if not line:  # 빈 줄을 입력하면 입력 종료
            break
        # 각 줄에서 숫자만 추출하여 문자열로 변환하여 집합에 추가
        numbers.add("A" + "".join(char for char in line if char.isdigit()))

    # 중복 제거된 리스트 생성
    unique_numbers = list(numbers)

    # 리스트에 저장된 숫자들을 출력
    print("출력된 문자열 배열:")
    print("{" + ", ".join('"' + num + '"' for num in unique_numbers) + "}")

    # 총 개수 출력
    print("총 개수:", len(unique_numbers))


if __name__ == "__main__":
    main()