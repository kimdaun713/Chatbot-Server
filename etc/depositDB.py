import sqlite3
import pandas as pd
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


url = 'http://apis.data.go.kr/B190030/GetDepositProductInfoService/getDepositProductList'
params = {
        'serviceKey' : 'XVQ1VnMFarvOUkVGKfQXUmnbsrWJhWsaHmmsjJW0UU0fosOJUFNjjHMwnTAEugXZdigSflhIr+f5j6KTrh/7pQ==',
        'pageNo' : '8',
        'numOfRows' : '1000',
        'sBseDt' : '20231101',
        'eBseDt' : '20241101'}

response = requests.get(url, params=params)
print(response.content.decode('utf-8'))

data = response.content.decode('utf-8')
root = ET.fromstring(data)

def save_to_db(data_list, conn):
    cursor = conn.cursor()
    for data in data_list:
        # 데이터에서 '예금 상품명' 부분 추출
        prdNm = data.split('|')[1].split(' ')[-1]  # "예금 상품명 KDB 국민연금 안심安心통장"에서 "KDB 국민연금 안심安心통장" 추출

        # 중복 체크 쿼리
        cursor.execute("INSERT INTO docs (doc) VALUES (?)", (data,))

    conn.commit()


unique_prdNm = set()
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
                    f"가입 대상 {jinTgtCone} | 가입 목적 / 상품 목적  {prdJinPpo} | 가입 채널/가입 경로  {prdJinChnCone} | "
                    f"최고 금리 {hitIrtCndCone} | 가입 기간 {prdJinTrmCone}.")
        sentence = re.sub(r'[\[\]\(\)]', '', sentence)
        processed_data.append(sentence)
        
        # 처리된 상품명을 집합에 추가
        unique_prdNm.add(prdNm)
    else:
        print(f"Skipped (duplicate): {prdNm}")

conn = create_database()
save_to_db(processed_data, conn)

index_to_search = 1 
sentence = get_doc_by_index(index_to_search, conn)
print(sentence)

conn.close
