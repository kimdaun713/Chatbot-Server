import sqlite3
import requests
import re
import xml.etree.ElementTree as ET


def create_database():
    conn = sqlite3.connect('docs.db')  # 데이터베이스 파일 이름
    cursor = conn.cursor()

    # 테이블 생성 (id는 자동 증가)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS docs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc TEXT NOT NULL
    )
    ''')
    conn.commit()
    return conn


def get_doc_by_index(index, conn):
    cursor = conn.cursor()
    cursor.execute('SELECT doc FROM docs WHERE id = ?', (index,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        return None


def save_to_db(data_list, conn):
    cursor = conn.cursor()
    for data in data_list:
        cursor.execute("INSERT INTO docs (doc) VALUES (?)", (data,))
    conn.commit()


def process_and_store_data(response_content, conn, unique_prdNm):
    root = ET.fromstring(response_content)

    processed_data = []
    for item in root.findall(".//item"):
        bseDt = item.find("bseDt").text if item.find("bseDt") is not None else ""
        prdNm = item.find("prdNm").text if item.find("prdNm") is not None else ""
        prdOtl = item.find("prdOtl").text if item.find("prdOtl") is not None else ""
        jinTgtCone = item.find("jinTgtCone").text if item.find("jinTgtCone") is not None else ""
        prdJinPpo = item.find("prdJinPpo").text if item.find("prdJinPpo") is not None else ""
        prdJinChnCone = item.find("prdJinChnCone").text if item.find("prdJinChnCone") is not None else ""
        hitIrtCndCone = item.find("hitIrtCndCone").text if item.find("hitIrtCndCone") is not None else ""
        prdJinTrmCone = item.find("prdJinTrmCone").text if item.find("prdJinTrmCone") is not None else ""

        if prdNm not in unique_prdNm:
            sentence = (f"기준 일자 {bseDt} | 예금 상품명 {prdNm} | 상품의 특징 {prdOtl}, "
                        f"가입 대상 {jinTgtCone} | 가입 목적 / 상품 목적 {prdJinPpo} | 가입 채널/가입 경로 {prdJinChnCone} | "
                        f"최고 금리 {hitIrtCndCone} | 가입 기간 {prdJinTrmCone}.")
            sentence = re.sub(r'[\[\]\(\)]', '', sentence)
            processed_data.append(sentence)
            unique_prdNm.add(prdNm)

    save_to_db(processed_data, conn)


# API URL 및 기본 파라미터 설정
url = 'http://apis.data.go.kr/B190030/GetDepositProductInfoService/getDepositProductList'
params = {
    'serviceKey': 'XVQ1VnMFarvOUkVGKfQXUmnbsrWJhWsaHmmsjJW0UU0fosOJUFNjjHMwnTAEugXZdigSflhIr+f5j6KTrh/7pQ==',
    'numOfRows': '1000',
    'sBseDt': '20191101',
    'eBseDt': '20241101'
}

# 데이터베이스 생성 및 중복 검사용 집합 초기화
conn = create_database()
unique_prdNm = set()

# 페이지 번호 1부터 9까지 반복
for page_no in range(1, 47):
    params['pageNo'] = str(page_no)
    response = requests.get(url, params=params)
    if response.status_code == 200:
        process_and_store_data(response.content.decode('utf-8'), conn, unique_prdNm)
    else:
        print(f"Error fetching data for page {page_no}")

# 예시로 첫 번째 데이터 가져오기
index_to_search = 1
sentence = get_doc_by_index(index_to_search, conn)
print(sentence)

conn.close()

