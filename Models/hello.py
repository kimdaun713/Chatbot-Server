import re
number_dict = {
        '공': 0,
        '영': 0,
        '일': 1,
        '이': 2,
        '삼': 3,
        '사': 4,
        '오': 5,
        '육': 6,
        '칠': 7,
        '팔': 8,
        '구': 9,
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9
    }

unit_dict = {
        '십': 10,
        '백': 100,
        '천': 1000
    }

large_unit_dict = {
        '만': 10000,
        '억': 100000000,
        '조': 1000000000000
    }

def k2n(s):
    s = s.replace('원', '').strip()  # 통화 기호 제거

    total = 0
    current_section = 0
    current_digit = None
    number_buffer = ''

    for char in s:
        if char in number_dict:
            # 숫자나 한글 숫자인 경우 number_buffer에 추가
            number_buffer += char
        elif char in unit_dict:
            if number_buffer:
                current_digit = int(''.join(str(number_dict[c]) for c in number_buffer))
                number_buffer = ''
            elif current_digit is None:
                current_digit = 1  # 앞에 숫자가 없으면 '일'로 간주
            current_section += current_digit * unit_dict[char]
            current_digit = None
        elif char in large_unit_dict:
            if number_buffer:
                current_digit = int(''.join(str(number_dict[c]) for c in number_buffer))
                number_buffer = ''
            if current_digit is not None:
                current_section += current_digit
                current_digit = None
            # **수정된 부분: current_section이 0이면 1로 간주**
            if current_section == 0:
                current_section = 1
            total += current_section * large_unit_dict[char]
            current_section = 0

    # 남은 숫자 처리
    if number_buffer:
        current_section += int(''.join(str(number_dict[c]) for c in number_buffer))
    if current_digit is not None:
        current_section += current_digit

    total += current_section
    return total


def extract_and_convert_prices(prompt):
    valid_chars = set(number_dict.keys()) | set(unit_dict.keys()) | set(large_unit_dict.keys()) | set([' '])
    i = 0
    n = len(prompt)
    result = ''
    while i < n:
        if prompt[i] in valid_chars:
            start = i
            while i < n and prompt[i] in valid_chars:
                i += 1
            substring = prompt[start:i]
            substring_no_space = substring.replace(' ', '')
            if len(substring_no_space) >= 2:
                try:
                    num = k2n(substring_no_space)
                    result += " "+str(num)
                except Exception as e:
                    result += substring
            else:
                result += substring
        else:
            result += prompt[i]
            i +=1
    return result

a = "3조 100만을 천원으로 표현하면 얼마야? "

print(f"전처리 전: {a}")
processed_prompt = extract_and_convert_prices(a)
print("전처리 후: ", end="")
print(processed_prompt)

b="최고 금리 상품 추천해줘. 3억 10만 1천원의 20%는 얼마야? 만오백원은 만오천원은? 오천 구백오십팔억삼천만원은?" 
print(f"전처리 전: {b}")
processed_prompt = extract_and_convert_prices(b)
print("전처리 후: ", end="")
print(processed_prompt)


# Examples:
print(k2n('사천오백6십이원'))
print(k2n('10만 5천원'))     
print(k2n('124만6000원'))    
print(k2n('백이십육만원'))    
print(k2n('이백10만1원'))
print(k2n('156만 126천원'))

