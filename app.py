import os
import random
import sys
import time
import urllib
import zipfile
from concurrent.futures import ThreadPoolExecutor
# from threading import Thread
# import numpy as np
#
# from PIL import Image
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import albumentations as albu
# from albumentations.pytorch import ToTensorV2
# from torchvision.transforms import ToTensor, ToPILImage
# from torchvision.utils import save_image
#
# import torch
# import torch.nn as nn
# from torchvision.transforms import transforms
from flask import Flask, jsonify, request
import pymysql
import boto3

import config
from config import AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY
from config import AWS_S3_BUCKET_NAME, AWS_S3_BUCKET_REGION

# mysql = MySQL()
app = Flask(__name__)

db = pymysql.connect(host=config.host, user=config.user, password=config.password, db=config.db, charset=config.charset) # 별도 컨피그에 정리
executor = ThreadPoolExecutor(max_workers=10)

# class CustomDataset_train(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.allowed_extensions = ['.jpg', '.jpeg', '.png']
#         self.image_files = [f for f in os.listdir(root_dir) if os.path.splitext(f)[1].lower() in self.allowed_extensions]
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.image_files[idx])
#         image = Image.open(img_name)
#
#         if self.transform:
#             image = self.transform(image)
#
#         return image
#
# class CustomDataset_image_extract(Dataset):
#     def __init__(self, root_dir, max_saved=30):
#         self.root_dir = root_dir
#         self.transform = get_training_augmentation()
#         self.allowed_extensions = ['.jpg', '.jpeg', '.png']
#         self.image_files = [f for f in os.listdir(root_dir) if os.path.splitext(f)[1].lower() in self.allowed_extensions]
#         self.max_saved = max_saved
#
#     def __len__(self):
#         return min(self.max_saved, len(self.image_files))
#
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.image_files[idx])
#         image = Image.open(img_name).convert("RGB")
#
#         if self.transform:
#             image = np.array(image)  # NumPy 배열로 변환
#             augmented = self.transform(image=image)
#             image = augmented['image']
#             image = ToTensorV2()(image=image)["image"]  # PyTorch 텐서로 다시 변환
#
#         return image
#
# # CustomDataset 클래스 정의
# class CustomDataset_jong(Dataset):
#     def __init__(self, folder_path, transform=None, max_samples=None):
#         self.folder_path = folder_path
#         self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
#
#         # 최대 샘플 갯수를 지정한다면, 리스트를 해당 갯수만큼 제한합니다.
#         if max_samples:
#             self.image_files = self.image_files[:max_samples]
#
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.folder_path, self.image_files[idx])
#         image = Image.open(img_name)
#
#         if self.transform:
#             image = self.transform(image)
#
#         return image

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

@app.route("/ganerate1", methods = ['POST'])
def gara():
    json_data = request.get_json()

    if json_data:
        print('apiTest IN')
        # upload_url = json_data.get("uploadUrl")
        original_file_name = json_data.get("originalFileName")
        create_data_size = json_data.get('createDataSize')  # 이미지 생성 개수
        upload_file_name = json_data.get("uploadFileName")
        data_product_id = json_data.get("dataProductId")

        time.sleep(100)

        # 이동하려는 디렉터리 경로를 지정합니다.
        directory_path = "/root"

        # 지정한 디렉터리로 이동합니다.
        os.chdir(directory_path)

        parts = original_file_name.rsplit('.', 1)
        zipfile_name = parts[0]  # 확장자 땐 파일이름
        ext = parts[1]  # 확장자
        millisecond = int(time.time() * 1000)
        upload_zip_file_name = zipfile_name + str(millisecond) + "." + ext  # s3 업로드 zip 이름

        gan_zip_path = os.path.join(directory_path, original_file_name);

        with open(gan_zip_path, 'rb') as f:
            gan_zip_data = f.read()

        # s3에 업로드
        s3.put_object(
            Bucket=AWS_S3_BUCKET_NAME,
            Key=upload_zip_file_name,
            Body=gan_zip_data
        )

        # 만약 업로드 이름에 한글이 들어가면 url이 이상해짐. -> 인코딩
        encoded_upload_zip_file_name = urllib.parse.quote(upload_zip_file_name)
        uploaded_url = f"https://{AWS_S3_BUCKET_NAME}.s3.amazonaws.com/{encoded_upload_zip_file_name}"

        # zip파일 사이즈 추출
        zip_data_size_gb = sys.getsizeof(gan_zip_data) / (1024 ** 3)

        db = pymysql.connect(host=config.host, user=config.user, password=config.password, db=config.db,
                             charset=config.charset)  # 별도 컨피그에 정리
        cursor = db.cursor()
        print("connect!!")

        sql = "INSERT INTO zip_file(upload_url, upload_file_name, original_file_name, size_gb) VALUES ('%s', '%s', '%s', '%f')" % (
        uploaded_url, upload_zip_file_name, original_file_name, zip_data_size_gb)

        cursor.execute(sql)
        db.commit()

        data = cursor.fetchall()
        print("zip insert!")

        # INSERT 한 레코드의 ID 값 가져오기
        inserted_zipfile_id = cursor.lastrowid  # 이 값은 zip_file 테이블에 INSERT한 레코드의 ID입니다.
        print(str(inserted_zipfile_id))

        update_sql = "UPDATE data_product SET zipfile_id = '%d' WHERE data_product_id = '%d'" % (
        inserted_zipfile_id, data_product_id)

        cursor.execute(update_sql)
        db.commit()  # UPDATE 후에 commit
        print("update dataProduct!")

        # 압축 해제할 디렉토리 경로
        extracted_dir_path = "/root/gan_zip"

        # 압축 파일을 압축 해제
        with zipfile.ZipFile(gan_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extracted_dir_path)

        num_images_to_upload = 3

        all_image_files = [f for f in os.listdir(extracted_dir_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # 랜덤하게 이미지 선택
        random_images = random.sample(all_image_files, num_images_to_upload)

        # 이미지 업로드
        for i, image_filename in enumerate(random_images):
            # 이미지 파일 이름과 확장자 분리
            image_name, image_extension = os.path.splitext(image_filename)
            # 밀리세컨드 구하기
            millisecond = int(time.time() * 1000)

            # s3 업로드 이미지 파일 이름 생성
            upload_image_file_name = f"{image_name}_{millisecond}{image_extension}"

            # 로컬의 이미지 경로설정
            image_path = os.path.join(extracted_dir_path, image_filename)

            # 이미지 업로드
            # s3에 업로드
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()

                s3.put_object(
                    Bucket=AWS_S3_BUCKET_NAME,
                    Key=upload_image_file_name,
                    Body=image_data
                )

            # 만약 업로드 이름에 한글이 들어가면 url이 이상해짐. -> 인코딩
            encoded_upload_zip_file_name = urllib.parse.quote(upload_image_file_name)
            uploaded_url = f"https://{AWS_S3_BUCKET_NAME}.s3.amazonaws.com/{encoded_upload_zip_file_name}"

            # DB에 이미지 예시 저장
            # db insert
            sql = "INSERT INTO example_image(image_url, original_file_name, upload_file_name, data_product_id) VALUES ('%s', '%s', '%s', '%d')" % (
                uploaded_url, image_filename, upload_image_file_name, data_product_id)
            cursor.execute(sql)

            data = cursor.fetchall()

            if not data:
                db.commit()
                print("insert!")
            else:
                db.rollback()
                print("fail!")

        db.close()
        return jsonify("", 200)

    return jsonify("null", 500)

#
# @app.route("/ganerate", methods = ['POST'])
# def make_zip(): # 멀티 프로세싱 처리
#     try:
#         # 작업 시간 체크
#         start_time = time.time()
#         json_data = request.get_json()  # JSON 요청 데이터 가져오기
#
#         if json_data:
#             print('apiTest IN')
#             # upload_url = json_data.get("uploadUrl")
#             original_file_name = json_data.get("originalFileName")
#             create_data_size = json_data.get('createDataSize') # 이미지 생성 개수
#             upload_file_name = json_data.get("uploadFileName")
#             data_product_id = json_data.get("dataProductId")
#
#             try:
#                 # S3로 부터 객체 가져오기
#                 obj = s3.get_object(Bucket=AWS_S3_BUCKET_NAME, Key=upload_file_name)
#                 zip_data = obj['Body'].read()
#
#                 # 로컬에 압축 해제할 디렉토리 경로 설정 (예: 데스크탑)
#                 desktop_path = os.path.expanduser("~/Desktop") #추후 ec2에 맞게 변경
#                 # desktop_path = os.path.expanduser("~") ec2꺼
#                 # /Desktop/abc.zip
#                 zip_file_path = os.path.join(desktop_path, upload_file_name) #추후 이름도 변경
#
#                 #확장자 제거한 파일 이름 -> 폴더 명으로 사용 (abc)
#                 new_folder_name = os.path.splitext(upload_file_name)[0]
#
#                 # /Desktop/abc
#                 new_folder_path = os.path.join(desktop_path, new_folder_name)
#
#                 # 지원할 이미지 확장자 리스트
#                 supported_extensions = ['.jpg', '.jpeg', '.png']
#
#                 try:
#                     # ZIP 데이터를 로컬 파일로 저장
#                     with open(zip_file_path, 'wb') as zip_file:
#                         zip_file.write(zip_data)
#                     print("ZIP 파일이 로컬에 저장되었습니다.")
#
#                     # /Desktop/abc 생성
#                     os.makedirs(new_folder_path)
#
#                     # 로컬에 저장된 ZIP 파일을 압축 해제
#                     with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#                         zip_ref.extractall(new_folder_path)
#                     print("압축 해제가 완료되었습니다.")
#
#                     # 디렉터리 내부 검사 및 확장자가 맞지 않는 파일 삭제
#                     directory_search_and_delete_file(new_folder_path, supported_extensions)
#
#                     # 로컬에 저장된 ZIP 파일 삭제
#                     os.remove(zip_file_path)
#                     print("로컬에 저장된 ZIP 파일이 삭제되었습니다.")
#
#                 except Exception as e:
#                     print("압축 해제 및 파일 삭제 중 오류 발생:", e)
#
#                 # 압축해제 한 파일 갯수 확인
#                 files_and_folders = os.listdir(new_folder_path)
#                 # 파일만 걸러내어 그 수를 반환합니다.
#                 print(len([name for name in files_and_folders if os.path.isfile(os.path.join(new_folder_path, name))]))
#
#                 # 증강된 이미지 저장 경로 /Desktop/abc/OUTPUT
#                 output_path = os.path.join(new_folder_path, "OUTPUT")
#
#                 if not os.path.exists(output_path):
#                     os.mkdir(output_path)
#
#                 # 데이터셋 및 DataLoader 설정
#                 dataset = CustomDataset_image_extract(new_folder_path)
#                 print("Number of data samples in the dataset:", len(dataset))
#                 data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
#
#                 # 저장할 이미지 수 설정
#                 max_saved = 30 # 총 저장할 이미지 수(마지막에 배포시 10000장으로)
#
#                 # 입력 이미지의 개수 확인
#                 num_images = len(dataset)
#
#                 # 이미지 저장 카운터
#                 saved_count = 0
#
#                 ## 데이터 증강
#                 if num_images >= max_saved:
#                     print("Enough images, skipping augmentation.")
#                     output_path=new_folder_path
#                     pass
#                 else:
#                     # 로더에서 모든 이미지를 로드
#                     # all_images = next(iter(data_loader))
#                     all_images = []
#                     for i in range(len(dataset)):
#                         image = dataset[i]
#                         all_images.append(image)
#
#                     # 증강된 이미지 저장
#                     while saved_count < max_saved:
#                         # 랜덤하게 하나의 이미지를 선택
#                         random_idx = random.randint(0, len(all_images) - 1)
#                         image = all_images[random_idx]
#                         image = image / 255.0 if image.max() > 1 else image  # 이미지를 0-1 범위로 정규화
#
#                         pil_image = ToPILImage()(image.cpu().squeeze(0))  # PyTorch 텐서를 PIL 이미지로 변환
#                         np_image = np.array(pil_image)  # PIL 이미지를 NumPy 배열로 변환
#                         augmented = get_training_augmentation()(image=np_image)  # 증강 적용
#                         tensor_image = ToTensor()(augmented["image"])  # NumPy 배열을 PyTorch 텐서로 변환
#
#                         save_image(tensor_image, os.path.join(output_path, f"augmented_image_{saved_count + 1}.jpg"))
#                         saved_count += 1
#
#
#
#
#
#
#
#
#                 print("딥러닝 시작 전")
#
#                 # 이미지 변환 설정
#                 transform = transforms.Compose([
#                     transforms.Resize((64, 64)),  # 이미지 크기를 조정합니다.
#                     transforms.ToTensor(),  # 이미지를 텐서로 변환합니다.
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                 ])
#
#                 # train_ds = CustomDataset_train(root_dir=output_path, transform=transform)
#                 train_ds = CustomDataset_jong(folder_path=output_path, transform=transform, max_samples= 5000)
#
#                 # train_ds 생성 후 출력
#                 print("데이터셋 샘플 개수:", len(train_ds))
#
#                 sample_image = train_ds[0]
#                 print("Sample image shape:", sample_image.size())
#
#                 # 바꿔준 이미지들을 DataLoader라는 라이브러리로 호출
#                 # train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)# 배치사이즈는 128로
#                 # 바꿔준 이미지들을 DataLoader로 호출
#                 batch_size = 128
#                 train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
#
#                 device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#                 workers = 2
#                 batch_size = 128
#                 image_size = 64
#                 nc = 3
#
#                 # Size of z latent vector (i.e. size of generator input)
#                 nz = 128
#                 # Size of feature maps in generator
#                 ngf = 64
#
#                 # Size of feature maps in discriminator
#                 ndf = 64
#                 lr = 0.0002
#                 beta1 = 0.5
#                 ngpu = 1
#
#                 netG = Generator(ngpu).to(device)
#                 if (device.type == 'cuda') and (ngpu > 1):
#                     netG = nn.DataParallel(netG, list(range(ngpu)))
#
#                 netG.apply(weights_init)
#
#                 netD = Discriminator(ngpu).to(device)
#                 if (device.type == 'cuda') and (ngpu > 1):
#                     netD = nn.DataParallel(netD, list(range(ngpu)))
#
#                 netD.apply(weights_init)
#                 # Initialize the ``MSELoss`` function
#                 criterion = nn.MSELoss()
#
#                 # Establish convention for real and fake labels during training
#                 real_label = 1.
#                 fake_label = 0.
#
#                 # Setup Adam optimizers for both G and D
#                 optimizerD = optim.AdamW(netD.parameters(), lr=lr, betas=(beta1, 0.999))
#                 optimizerG = optim.AdamW(netG.parameters(), lr=lr, betas=(beta1, 0.999))
#                 lr = 0.0002
#                 # Epoch 설정
#                 num_epochs = 10
#
#                 print("fit 시작 전")
#
#                 save_folder = fit(netD=netD, netG=netG, criterion=criterion, optimizerD=optimizerD, optimizerG=optimizerG, train_dl=train_dl, device=device,
#                 num_epochs=num_epochs, nz=nz, real_label=real_label, fake_label=fake_label, output_path=output_path, save_samples=save_samples)
#                 print("fit 후")
#
#                 parts = original_file_name.rsplit('.', 1)
#                 zipfile_name = parts[0] #확장자 땐 파일이름
#                 ext = parts[1] #확장자
#                 millisecond = int(time.time() * 1000)
#                 upload_zip_file_name = zipfile_name + str(millisecond) + "." + ext # s3 업로드 zip 이름
#
#                 gan_zip_path = os.path.join(save_folder, original_file_name);
#
#                 with open(gan_zip_path, 'rb') as f:
#                     gan_zip_data = f.read()
#
#                 # s3에 업로드
#                 s3.put_object(
#                     Bucket=AWS_S3_BUCKET_NAME,
#                     Key=upload_zip_file_name,
#                     Body=gan_zip_data
#                 )
#
#                 # 만약 업로드 이름에 한글이 들어가면 url이 이상해짐. -> 인코딩
#                 encoded_upload_zip_file_name = urllib.parse.quote(upload_zip_file_name)
#                 uploaded_url = f"https://{AWS_S3_BUCKET_NAME}.s3.amazonaws.com/{encoded_upload_zip_file_name}"
#
#                 # zip파일 사이즈 추출
#                 zip_data_size_gb = sys.getsizeof(zip_data) / (1024 ** 3)
#
#                 db = pymysql.connect(host=config.host, user=config.user, password=config.password, db=config.db,
#                                      charset=config.charset)  # 별도 컨피그에 정리
#                 cursor = db.cursor()
#                 print("connect!!")
#
#                 sql = "INSERT INTO zip_file(upload_url, upload_file_name, original_file_name, size_gb) VALUES ('%s', '%s', '%s', '%f')" % (uploaded_url, upload_zip_file_name, original_file_name, zip_data_size_gb)
#
#                 cursor.execute(sql)
#                 db.commit()
#
#                 data = cursor.fetchall()
#                 print("zip insert!")
#
#                 # INSERT 한 레코드의 ID 값 가져오기
#                 inserted_zipfile_id = cursor.lastrowid  # 이 값은 zip_file 테이블에 INSERT한 레코드의 ID입니다.
#                 print(str(inserted_zipfile_id))
#
#                 update_sql = "UPDATE data_product SET zipfile_id = '%d' WHERE data_product_id = '%d'" % (inserted_zipfile_id, data_product_id)
#
#                 cursor.execute(update_sql)
#                 db.commit()  # UPDATE 후에 commit
#                 print("update dataProduct!")
#
#
#                 ganerated_image_path = os.path.join(output_path, "all_images")
#                 num_images_to_upload = 3
#
#                 all_image_files = [f for f in os.listdir(ganerated_image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
#
#                 # 랜덤하게 이미지 선택
#                 random_images = random.sample(all_image_files, num_images_to_upload)
#
#                 # 이미지 업로드
#                 for i, image_filename in enumerate(random_images):
#                     # 이미지 파일 이름과 확장자 분리
#                     image_name, image_extension = os.path.splitext(image_filename)
#                     # 밀리세컨드 구하기
#                     millisecond = int(time.time() * 1000)
#
#                     # s3 업로드 이미지 파일 이름 생성
#                     upload_image_file_name = f"{image_name}_{millisecond}{image_extension}"
#
#                     # 로컬의 이미지 경로설정
#                     image_path = os.path.join(ganerated_image_path, image_filename)
#
#                     # 이미지 업로드
#                     # s3에 업로드
#                     with open(image_path, 'rb') as image_file:
#                         image_data = image_file.read()
#
#                         s3.put_object(
#                             Bucket=AWS_S3_BUCKET_NAME,
#                             Key=upload_image_file_name,
#                             Body=image_data
#                         )
#
#                     # 만약 업로드 이름에 한글이 들어가면 url이 이상해짐. -> 인코딩
#                     encoded_upload_zip_file_name = urllib.parse.quote(upload_image_file_name)
#                     uploaded_url = f"https://{AWS_S3_BUCKET_NAME}.s3.amazonaws.com/{encoded_upload_zip_file_name}"
#
#                     # DB에 이미지 예시 저장
#                     # db insert
#                     sql = "INSERT INTO example_image(image_url, original_file_name, upload_file_name, data_product_id) VALUES ('%s', '%s', '%s', '%d')" % (
#                     uploaded_url, image_filename, upload_image_file_name, data_product_id)
#                     cursor.execute(sql)
#
#                     data = cursor.fetchall()
#
#                     if not data:
#                         db.commit()
#                         print("insert!")
#                     else:
#                         db.rollback()
#                         print("fail!")
#
#             except Exception as e:
#                 print(e)
#                 return False
#
#         ## 작업 시간 체크
#         print("apiTest elapsed: ", time.time() - start_time)  # seconds
#         print('apiTest OUT')
#
#         return jsonify("", 200)
#
#     except Exception as e:
#         return jsonify("null", 500)
#
#     finally:
#         db.close()
#
# # custom weights initialization called on ``netG`` and ``netD``
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)
#
#
# def directory_search_and_delete_file(new_folder_path, supported_extensions):
#     # 디렉토리 내부의 파일 및 폴더 검사 및 삭제
#     for root, dirs, files in os.walk(new_folder_path):
#         for name in files + dirs:
#             file_path = os.path.join(root, name)
#             extension = os.path.splitext(name)[1].lower()
#
#             # 파일 및 폴더 이름을 CP437로 인코딩 후 UTF-8로 디코딩
#             decoded_filename = name.encode('cp437').decode('utf-8')
#
#             # 지원하는 확장자가 아닌 경우 파일 또는 폴더 삭제
#             if extension not in supported_extensions:
#                 if os.path.isfile(file_path):
#                     os.remove(file_path)
#                     print(f"삭제된 파일: {decoded_filename}")
#                 elif os.path.isdir(file_path):
#                     os.rmdir(file_path)
#                     print(f"삭제된 폴더: {decoded_filename}")
#
#
# def to_device(data, device):
#     """Move tensor(s) to chosen device"""
#     if isinstance(data, (list,tuple)):
#         return [to_device(x, device) for x in data]
#     return data.to(device, non_blocking=True)
#
# class DeviceDataLoader():
#     """Wrap a dataloader to move data to a device"""
#     def __init__(self, dl, device):
#         self.dl = dl
#         self.device = device
#
#     def __iter__(self):
#         """Yield a batch of data after moving it to device"""
#         for b in self.dl:
#             yield to_device(b, self.device)
#
#     def __len__(self):
#         """Number of batches"""
#         return len(self.dl)
#
# # 학습 함수에 이미지 저장 로직 추가
# def fit(
#     netD, netG, criterion,
#     optimizerD, optimizerG,
#     train_dl, device,
#     num_epochs, nz,
#     real_label, fake_label,
#     output_path, save_samples):
#
#     # Training Loop
#     iters = 0
#     # For each epoch
#     for epoch in range(num_epochs):
#         # For each batch in the dataloader
#         for i, data in enumerate(train_dl):
#
#             # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
#             ## Train with all-real batch
#             netD.zero_grad()
#
#             # Format batch
#             real_cpu = data[0].to(device)
#             real_cpu = real_cpu.unsqueeze(0)
#             b_size = real_cpu.size(0)
#             label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
#
#             # Forward pass real batch through D
#             #output = netD(real_cpu).view(-1)
#             output = netD(real_cpu).view(-1) # 채널 차원 추가
#
#             # Calculate loss on all-real batch
#             errD_real = criterion(output, label)
#
#             # Calculate gradients for D in backward pass
#             errD_real.backward()
#             D_x = output.mean().item()
#
#             ## Train with all-fake batch
#             # Generate batch of latent vectors
#             noise = torch.randn(b_size, nz, 1, 1, device=device)
#
#             # Generate fake image batch with G
#             fake = netG(noise)
#             label.fill_(fake_label)
#
#             # Classify all fake batch with D
#             output = netD(fake.detach()).view(-1)
#
#             # Calculate D's loss on the all-fake batch
#             errD_fake = criterion(output, label)
#
#             # Calculate the gradients for this batch, accumulated (summed) with previous gradients
#             errD_fake.backward()
#             D_G_z1 = output.mean().item()
#
#             # Compute error of D as sum over the fake and the real batches
#             errD = errD_real + errD_fake
#
#             # Update D
#             optimizerD.step()
#
#             # (2) Update G network: maximize log(D(G(z)))
#             netG.zero_grad()
#             label.fill_(real_label) # fake labels are real for generator cost
#             # Since we just updated D, perform another forward pass of all-fake batch through D
#             output = netD(fake).view(-1)
#
#             # Calculate G's loss based on this output
#             errG = criterion(output, label)
#
#             # Calculate gradients for G
#             errG.backward()
#             D_G_z2 = output.mean().item()
#
#             # Update G
#             optimizerG.step()
#
#             if i == len(train_dl) - 1:
#                 latent_vector = torch.randn(1, nz, 1, 1, device=device)  # 랜덤한 latent_vector 생성
#                 save_folder = save_samples(epoch, latent_vector, netG, output_path, num_epochs, show=False) # 이미지 저장
#
#             iters += 1
#
#
#     zip_filename = os.path.join(output_path, "all_generated_images.zip")
#     with zipfile.ZipFile(zip_filename, "w") as zipf:
#         for root, _, files in os.walk(save_folder):
#             for file in files:
#                 file_path = os.path.join(root, file)
#                 zipf.write(file_path)
#
#
#     return save_folder
#
# def save_samples(epoch, latent_vector, netG, output_path,num_epochs, show=True):
#     # breakpoint()
#     generated_images = generate_images(latent_vector,netG)
#
#     # 전체 이미지를 저장할 디렉토리 생성
#     save_folder = os.path.join(output_path, "all_images")
#     os.makedirs(save_folder, exist_ok=True)
#
#     for i, image in enumerate(generated_images):
#         image_tensor = image.clone().detach().cpu()
#         image_tensor = image_tensor.squeeze(0)
#         image_pil = transforms.ToPILImage()(image_tensor)
#
#         image_filename = f"epoch_{epoch}_generated_image_{i}.png"
#         image_path = os.path.join(save_folder, image_filename)
#
#         # 이미지 저장
#         image_pil.save(image_path)
#
#     if show:
#         pass
#
# def my_collate(batch):
#     return batch
#
# def zip_images(folder_path, output_zip_name):
#     image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
#
#     if not image_files:
#         print("폴더 내에 이미지 파일이 없습니다.")
#         return
#
#     # ZIP 파일 생성
#     with zipfile.ZipFile(os.path.join(folder_path, output_zip_name), 'w') as zipf:
#         for image_file in image_files:
#             file_path = os.path.join(folder_path, image_file)
#             zipf.write(file_path, image_file)
#
#     print("ZIP 파일 생성 완료.")
#
# # 데이터 증강 코드
# def get_training_augmentation():
#     train_transform = [
#
#         albu.HorizontalFlip(p=0.5), # 0.5확률로 수평 뒤집기
#         albu.ShiftScaleRotate(scale_limit=[0.95, 1.05], rotate_limit=45, shift_limit=0.005, p=1, border_mode=0 ), # 이동, 확대, 축소, 회전
#         albu.PadIfNeeded(min_height=400, min_width=400, always_apply=True, border_mode=0),# 이미지가 지정크기보다 작으면, 패딩함
#
#         albu.GaussNoise(p=0.2), # 0.2확률로 가우시안 노이즈 추가
#         albu.Perspective(p=0.3), # 0.3 확률로 투시 변환
#
#         albu.OneOf(    # 주어진 증강 중 하나를 확률로 CLANE, 랜덤 밝기, 랜덤 감마 등을 적용
#             [
#                 albu.CLAHE(p=1),
#                 albu.RandomBrightness(p=1),
#                 albu.RandomGamma(p=1),
#             ],
#             p=0.8,
#         ),
#
#         albu.OneOf( #이미지 날카롭게, 윤곽 강조, 블러 정도 조절
#             [
#                 albu.Sharpen(p=1),
#                 albu.Blur(blur_limit=3, p=1),
#                 albu.MotionBlur(blur_limit=3, p=1),
#             ],
#             p=0.8,
#         ),
#
#         albu.OneOf(
#             [
#                 albu.RandomContrast(p=1), #대비
#                 albu.HueSaturationValue(p=1), #채도
#             ],
#             p=0.8,
#         ),
#     ]
#     return albu.Compose(train_transform)
#
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)
# # Generator Code
# class Generator(nn.Module):
#     def __init__(self, ngpu):
#         super(Generator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d( 128, 64 * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(64 * 8),
#             nn.ReLU(True),
#             # state size. ``(ngf*8) x 4 x 4``
#             nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64 * 4),
#             nn.ReLU(True),
#             # state size. ``(ngf*4) x 8 x 8``
#             nn.ConvTranspose2d( 64 * 4, 64 * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64 * 2),
#             nn.ReLU(True),
#             # state size. ``(ngf*2) x 16 x 16``
#             nn.ConvTranspose2d( 64 * 2, 64, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             # state size. ``(ngf) x 32 x 32``
#             nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. ``(nc) x 64 x 64``
#         )
#     def forward(self, input):
#         return self.main(input)
#
# class Discriminator(nn.Module):
#     def __init__(self, ngpu):
#         super(Discriminator, self).__init__()
#         self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is ``(nc) x 64 x 64``
#             nn.Conv2d(3, 64, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf) x 32 x 32``
#             nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64 * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf*2) x 16 x 16``
#             nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64 * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf*4) x 8 x 8``
#             nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64 * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. ``(ndf*8) x 4 x 4``
#             nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False)
#         )
#     def forward(self, input):
#         return self.main(input)
#
# def generate_images(latent_vector,netG):
#     with torch.no_grad():
#         fake = netG(latent_vector).detach().cpu()
#     return fake

if __name__ == '__main__':
    app.run(debug=True)
