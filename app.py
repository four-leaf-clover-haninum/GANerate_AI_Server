import sys
import time
import urllib
from http import HTTPStatus

from flask import Flask, jsonify, request
import pymysql
import boto3

import config
from config import AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY
from config import AWS_S3_BUCKET_NAME, AWS_S3_BUCKET_REGION

app = Flask(__name__)
db = pymysql.connect(host=config.host, user=config.user, password=config.password, db=config.db, charset=config.charset) # 별도 컨피그에 정리
cursor = db.cursor()

def s3_connection():
    '''
    s3 bucket에 연결
    :return: 연결된 s3 객체
    '''
    try:
        s3 = boto3.client(
            service_name='s3',
            region_name=AWS_S3_BUCKET_REGION,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
    except Exception as e:
        print(e)
        exit("ERROR_S3_CONNECTION_FAILED")
    else:
        print("s3 bucket connected!")
        return s3

s3 = s3_connection()
@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route("/test", methods=['POST'])
def test():

    id = request.get_json()


    sql = "SELECT * FROM user"
    cursor.execute(sql)

    data = cursor.fetchall()  # 모든 행 가져오기

    db.commit()
    db.close()

    return jsonify({"data": data, "status": HTTPStatus.OK})

@app.route("/ganerate", methods = ['POST'])
def make_zip(): # 멀티 프로세싱 처리
    try:
        breakpoint()
        # 작업 시간 체크
        start_time = time.time()
        json_data = request.get_json()  # JSON 요청 데이터 가져오기

        if json_data:
            print('apiTest IN')
            # upload_url = json_data.get("uploadUrl")
            original_file_name = json_data.get("originalFileName")
            create_data_size = json_data.get('createDataSize') # 이미지 생성 개수
            upload_file_name = json_data.get("uploadFileName")
            print('apiTest param : ', json_data)

            try:
                # S3로 부터 객체 가져오기
                obj = s3.get_object(Bucket=AWS_S3_BUCKET_NAME, Key=upload_file_name)
                zip_data = obj['Body'].read() # 바이트 형태의 zip 파일
                print(zip_data)


                #딥러닝
                # # 결과 zip을 s3에 저장
                # zip_data = b'PK\x03\x04\x14\x00\x00\x00\x08\x00\x1d[\x0bY\t\x00\x00\x00\t\x00\x00\x00\x08\x00\x00\x00\x65\x78ample.txtThis is a sample text file.\n'

                parts = original_file_name.rsplit('.', 1)
                zipfile_name = parts[0] #확장자 땐 파일이름
                ext = parts[1] #확장자
                millisecond = int(time.time() * 1000)
                upload_zip_file_name = zipfile_name + str(millisecond) + "." + ext # s3 업로드 zip 이름

                s3.put_object(
                    Bucket=AWS_S3_BUCKET_NAME,
                    Key=upload_zip_file_name,
                    Body=zip_data
                )

                # 만약 업로드 이름에 한글이 들어가면 url이 이상해짐. -> 인코딩
                encoded_upload_zip_file_name = urllib.parse.quote(upload_zip_file_name)
                uploaded_url = f"https://{AWS_S3_BUCKET_NAME}.s3.amazonaws.com/{encoded_upload_zip_file_name}"

                # zip파일 사이즈 추출
                zip_data_size_gb = sys.getsizeof(zip_data) / (1024 ** 3)

                # # 객체 다시 저장
                # sql = "INSERT INTO zip_file (is_exam_zip, original_file_name, size_gb, upload_file_name, upload_url) VALUES (%s, %s, %s, %s, %s)"
                # data = (False, zip_file_name, zip_data_size_gb, zip_file_name, uploaded_url)  # 삽입할 값들을 튜플로 정의

                # cursor.execute(sql, data)
                # db.commit()

                # inserted_id = cursor.lastrowid
                # print("Inserted ID:", inserted_id)

            except Exception as e:
                print(e)
                return False

        ## 작업 시간 체크
        print("apiTest elapsed: ", time.time() - start_time)  # seconds
        print('apiTest OUT')

        # 방금 업로드한 객체 조회
        # sql = "SELECT zipfile_id FROM zip_file where zipfile_id = %s"
        # cursor.execute(sql, (inserted_id))
        # row = cursor.fetchone()
        # print(row[0])
        # zipfileId = row[0]

        return jsonify({"originalFileName": original_file_name, "uploadFileName":upload_zip_file_name, "uploadUrl":uploaded_url, "sizeGb":zip_data_size_gb})

    except Exception as e:
        return jsonify("null", 500)


if __name__ == '__main__':
    app.run(debug=True)
