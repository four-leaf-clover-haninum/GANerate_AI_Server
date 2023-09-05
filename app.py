import os
import random
import sys
import time
import urllib
import zipfile
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import save_image

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from flask import Flask, jsonify, request
from flask_mysqldb import MySQL
import pymysql
import boto3

import config
from config import AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY
from config import AWS_S3_BUCKET_NAME, AWS_S3_BUCKET_REGION

# mysql = MySQL()
app = Flask(__name__)

db = pymysql.connect(host=config.host, user=config.user, password=config.password, db=config.db, charset=config.charset) # 별도 컨피그에 정리
executor = ThreadPoolExecutor(max_workers=10)


class CustomDataset_train(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.allowed_extensions = ['.jpg', '.jpeg', '.png']
        self.image_files = [f for f in os.listdir(root_dir) if os.path.splitext(f)[1].lower() in self.allowed_extensions]
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image

class CustomDataset_image_extract(Dataset):
    def __init__(self, root_dir, max_saved=30):
        self.root_dir = root_dir
        self.transform = get_training_augmentation()
        self.allowed_extensions = ['.jpg', '.jpeg', '.png']
        self.image_files = [f for f in os.listdir(root_dir) if os.path.splitext(f)[1].lower() in self.allowed_extensions]
        self.max_saved = max_saved

    def __len__(self):
        return min(self.max_saved, len(self.image_files))

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = np.array(image)  # NumPy 배열로 변환
            augmented = self.transform(image=image)
            image = augmented['image']
            image = ToTensorV2()(image=image)["image"]  # PyTorch 텐서로 다시 변환

        return image

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


# def init():
#     t1 = Thread(target=make_zip)  # make_zip 함수를 실행할 스레드 생성
#     t1.daemon = True  # 데몬 스레드로 설정
#     t1.start()  # 스레드 시작
#
# # @app.route("/generate", methods=['POST'])
# # def handle_request():
# #     init()  # init 함수 호출로 스레드를 시작
# #     return jsonify({"status": "processing"}), 202


@app.route("/ganerate", methods = ['POST'])
def make_zip(): # 멀티 프로세싱 처리
    try:
        # 작업 시간 체크
        start_time = time.time()
        json_data = request.get_json()  # JSON 요청 데이터 가져오기

        if json_data:
            print('apiTest IN')
            # upload_url = json_data.get("uploadUrl")
            original_file_name = json_data.get("originalFileName")
            create_data_size = json_data.get('createDataSize') # 이미지 생성 개수
            upload_file_name = json_data.get("uploadFileName")
            data_product_id = json_data.get("dataProductId")

            try:
                # S3로 부터 객체 가져오기
                obj = s3.get_object(Bucket=AWS_S3_BUCKET_NAME, Key=upload_file_name)
                zip_data = obj['Body'].read()

                # 로컬에 압축 해제할 디렉토리 경로 설정 (예: 데스크탑)
                desktop_path = os.path.expanduser("~/Desktop") #추후 ec2에 맞게 변경
                # /Desktop/abc.zip
                zip_file_path = os.path.join(desktop_path, upload_file_name) #추후 이름도 변경

                #확장자 제거한 파일 이름 -> 폴더 명으로 사용 (abc)
                new_folder_name = os.path.splitext(upload_file_name)[0]

                # /Desktop/abc
                new_folder_path = os.path.join(desktop_path, new_folder_name)

                # 지원할 이미지 확장자 리스트
                supported_extensions = ['.jpg', '.jpeg', '.png']

                try:
                    # ZIP 데이터를 로컬 파일로 저장
                    with open(zip_file_path, 'wb') as zip_file:
                        zip_file.write(zip_data)
                    print("ZIP 파일이 로컬에 저장되었습니다.")

                    # /Desktop/abc 생성
                    os.makedirs(new_folder_path)

                    # 로컬에 저장된 ZIP 파일을 압축 해제
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        zip_ref.extractall(new_folder_path)
                    print("압축 해제가 완료되었습니다.")

                    # 디렉터리 내부 검사 및 확장자가 맞지 않는 파일 삭제
                    directory_search_and_delete_file(new_folder_path, supported_extensions)

                    # 로컬에 저장된 ZIP 파일 삭제
                    os.remove(zip_file_path)
                    print("로컬에 저장된 ZIP 파일이 삭제되었습니다.")

                except Exception as e:
                    print("압축 해제 및 파일 삭제 중 오류 발생:", e)

                # 압축해제 한 파일 갯수 확인
                files_and_folders = os.listdir(new_folder_path)
                # 파일만 걸러내어 그 수를 반환합니다.
                print(len([name for name in files_and_folders if os.path.isfile(os.path.join(new_folder_path, name))]))

                # 데이터 증강
                # data_transforms = transforms.Compose([
                #
                #     transforms.RandomHorizontalFlip(),  # 수평뒤집기
                #     transforms.RandomRotation(10),  # 10도 내에서 회전
                #     transforms.RandomAffine(0, translate=(0.2, 0.2)),  # 20퍼센트 좌우 및 상하 이동
                #     transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # 무작위로 이미지 크롭 및 128x128 사이즈로 이미지 설정
                # ])

                # 증강된 이미지 저장 경로 /Desktop/abc/OUTPUT
                output_path = os.path.join(new_folder_path, "OUTPUT")

                if not os.path.exists(output_path):
                    os.mkdir(output_path)

                # 데이터셋 및 DataLoader 설정
                dataset = CustomDataset_image_extract(new_folder_path)
                print("Number of data samples in the dataset:", len(dataset))
                data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

                # 저장할 이미지 수 설정
                max_saved = 10 # 총 저장할 이미지 수(마지막에 배포시 10000장으로)

                # 입력 이미지의 개수 확인
                num_images = len(dataset)

                # 이미지 저장 카운터
                saved_count = 0

                ## 데이터 증강
                if num_images >= max_saved:
                    print("Enough images, skipping augmentation.")
                    output_path=new_folder_path
                    pass
                else:
                    # 로더에서 모든 이미지를 로드
                    # all_images = next(iter(data_loader))
                    all_images = []
                    for i in range(len(dataset)):
                        image = dataset[i]
                        all_images.append(image)

                    # 증강된 이미지 저장
                    while saved_count < max_saved:
                        # 랜덤하게 하나의 이미지를 선택
                        random_idx = random.randint(0, len(all_images) - 1)
                        image = all_images[random_idx]
                        image = image / 255.0 if image.max() > 1 else image  # 이미지를 0-1 범위로 정규화

                        pil_image = ToPILImage()(image.cpu().squeeze(0))  # PyTorch 텐서를 PIL 이미지로 변환
                        np_image = np.array(pil_image)  # PIL 이미지를 NumPy 배열로 변환
                        augmented = get_training_augmentation()(image=np_image)  # 증강 적용
                        tensor_image = ToTensor()(augmented["image"])  # NumPy 배열을 PyTorch 텐서로 변환

                        save_image(tensor_image, os.path.join(output_path, f"augmented_image_{saved_count + 1}.jpg"))
                        saved_count += 1

                    # all_image_paths = []
                    # for root, _, files in os.walk(new_folder_path):  # dataset_root는 이미지 파일들이 있는 디렉토리 경로입니다.
                    #     for file in files:
                    #         if file.endswith((".jpg", ".jpeg", ".png")):  # 이미지 파일인 경우만 고려하도록 확장자를 지정합니다.
                    #             file_path = os.path.join(root, file)
                    #             all_image_paths.append(file_path)

                print("딥러닝 시작 전")
                # breakpoint()
                # 딥러닝
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),  # 이미지 크기를 조정합니다.
                    transforms.ToTensor(),  # 이미지를 텐서로 변환합니다.
                ])

                train_ds = CustomDataset_train(root_dir=output_path, transform=transform)

                # train_ds 생성 후 출력
                print("데이터셋 샘플 개수:", len(train_ds))

                sample_image = train_ds[0]
                print("Sample image shape:", sample_image.size())

                # 바꿔준 이미지들을 DataLoader라는 라이브러리로 호출
                train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)# 배치사이즈는 128로
                # train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)

                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                # train_dl = DeviceDataLoader(train_dl, device)

                discriminator = nn.Sequential(
                    # in: 3 x 256 x 256
                    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, inplace=True),
                    # out: 64 x 128 x 128로

                    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True),
                    # out: 128 x 16 x 16

                    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True),
                    # out: 256 x 8 x 8

                    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True),
                    # out: 512 x 4 x 4

                    nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.2, inplace=True),
                    # out: 1024 x 8 x 8

                    nn.Conv2d(1024, 1, kernel_size=8, stride=1, padding=0, bias=False),
                    # out: 1 x 1 x 1

                    nn.Flatten(),
                    nn.Sigmoid()
                )
                discriminator = to_device(discriminator, device)
                latent_size = 128

                # Generator
                generator = nn.Sequential(
                    # in: latent_size x 1 x 1

                    nn.ConvTranspose2d(latent_size, 1024, kernel_size=4, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(True),
                    # out: 512 x 4 x 4

                    nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(True),
                    # out: 512 x 8 x 8

                    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(True),
                    # out: 256 x 16 x 16

                    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    # out: 128 x 32 x 32

                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    # out: 64 x 64 x 64

                    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.Tanh(),
                    # out: 3 x 128 x 128

                    nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.Tanh()
                    # out: 3 x 256 x 256
                )

                model = {
                    "discriminator": discriminator.to(device),
                    "generator": generator.to(device)
                }

                criterion = {
                    "discriminator": nn.BCELoss(),
                    "generator": nn.BCELoss()
                }
                lr = 0.0002
                # Epoch 설정
                epochs = 10


                fixed_latent = torch.randn(1, latent_size, 1, 1, device=device)
                sample_interval = 1
                start_idx=1
                batch_size=1
                print("fit 시작 전")

                save_folder = fit(model, criterion, epochs, lr, start_idx, sample_interval, batch_size, latent_size, device, train_dl, fixed_latent, output_path, generator, original_file_name)
                print("fit 후")

                parts = original_file_name.rsplit('.', 1)
                zipfile_name = parts[0] #확장자 땐 파일이름
                ext = parts[1] #확장자
                millisecond = int(time.time() * 1000)
                upload_zip_file_name = zipfile_name + str(millisecond) + "." + ext # s3 업로드 zip 이름

                gan_zip_path = os.path.join(save_folder, original_file_name);

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
                zip_data_size_gb = sys.getsizeof(zip_data) / (1024 ** 3)

                db = pymysql.connect(host=config.host, user=config.user, password=config.password, db=config.db,
                                     charset=config.charset)  # 별도 컨피그에 정리
                cursor = db.cursor()
                print("connect!!")

                sql = "INSERT INTO zip_file(upload_url, upload_file_name, original_file_name, size_gb) VALUES ('%s', '%s', '%s', '%f')" % (uploaded_url, upload_zip_file_name, original_file_name, zip_data_size_gb)

                cursor.execute(sql)
                db.commit()

                data = cursor.fetchall()
                print("zip insert!")

                # INSERT 한 레코드의 ID 값 가져오기
                inserted_zipfile_id = cursor.lastrowid  # 이 값은 zip_file 테이블에 INSERT한 레코드의 ID입니다.
                print(str(inserted_zipfile_id))

                update_sql = "UPDATE data_product SET zipfile_id = '%d' WHERE data_product_id = '%d'" % (inserted_zipfile_id, data_product_id)

                cursor.execute(update_sql)
                db.commit()  # UPDATE 후에 commit
                print("update dataProduct!")


                ganerated_image_path = os.path.join(output_path, "all_images")
                num_images_to_upload = 3

                all_image_files = [f for f in os.listdir(ganerated_image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

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
                    image_path = os.path.join(ganerated_image_path, image_filename)

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

            except Exception as e:
                print(e)
                return False

        ## 작업 시간 체크
        print("apiTest elapsed: ", time.time() - start_time)  # seconds
        print('apiTest OUT')

        return jsonify("", 200)

    except Exception as e:
        return jsonify("null", 500)

    finally:
        db.close()


def directory_search_and_delete_file(new_folder_path, supported_extensions):
    # 디렉토리 내부의 파일 및 폴더 검사 및 삭제
    for root, dirs, files in os.walk(new_folder_path):
        for name in files + dirs:
            file_path = os.path.join(root, name)
            extension = os.path.splitext(name)[1].lower()

            # 파일 및 폴더 이름을 CP437로 인코딩 후 UTF-8로 디코딩
            decoded_filename = name.encode('cp437').decode('utf-8')

            # 지원하는 확장자가 아닌 경우 파일 또는 폴더 삭제
            if extension not in supported_extensions:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"삭제된 파일: {decoded_filename}")
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
                    print(f"삭제된 폴더: {decoded_filename}")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

# 학습 함수에 이미지 저장 로직 추가
def fit(model, criterion, epochs, lr, start_idx, sample_interval, batch_size, latent_size, device, train_dl, fixed_latent, output_path, generator, result_name):
    model["discriminator"].train()
    model["generator"].train()
    torch.cuda.empty_cache()

    # Create optimizers
    optimizer = {
        "discriminator": torch.optim.Adam(model["discriminator"].parameters(),
                                          lr=lr, betas=(0.5, 0.999)),
        "generator": torch.optim.Adam(model["generator"].parameters(),
                                      lr=lr, betas=(0.5, 0.999))
    }

    for epoch in range(epochs):

        for real_images in train_dl:
            print(real_images)  # 리얼 이미지랑, 벨류값이 있어야됨.
            # Train discriminator
            # Clear discriminator gradients
            optimizer["discriminator"].zero_grad()

            # Pass real images through discriminator

            # 일반적으로 dis의 출력은 [batch_size, 1] 형태여야함.
            # 그럼 현재 [1,1] 이 맞음
            real_preds = model["discriminator"](real_images)
            print(real_preds.shape)
            print(real_images.shape)
            real_targets = torch.ones(real_images.size(0), 1, device=device)
            real_loss = criterion["discriminator"](real_preds, real_targets)

            # Generate fake images
            latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
            fake_images = model["generator"](latent)

            # Pass fake images through discriminator
            fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
            fake_preds = model["discriminator"](fake_images)
            fake_loss = criterion["discriminator"](fake_preds, fake_targets)

            # Update discriminator weights
            loss_d = real_loss + fake_loss
            loss_d.backward()
            optimizer["discriminator"].step()

            # Train generator
            # Clear generator gradients
            optimizer["generator"].zero_grad()

            # Generate fake images
            latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
            fake_images = model["generator"](latent)

            # Try to fool the discriminator
            preds = model["discriminator"](fake_images)
            targets = torch.ones(batch_size, 1, device=device)
            loss_g = criterion["generator"](preds, targets)

            # Update generator weights
            loss_g.backward()
            optimizer["generator"].step()

        # Save generated images
        if epoch == epochs - 1 or (epoch + 1) % sample_interval == 0:
            save_folder = save_samples(epoch + start_idx, fixed_latent, output_path, generator, epochs, result_name, show=False)
    return save_folder


def save_samples(epoch, latent_vector, output_path, generator, epochs, result_name, show=True):
    generated_images = generator(latent_vector)

    # 전체 이미지를 저장할 디렉토리 생성
    save_folder = os.path.join(output_path, "all_images")
    os.makedirs(save_folder, exist_ok=True)

    for i, image in enumerate(generated_images):
        image_tensor = image.clone().detach().cpu()
        image_tensor = image_tensor.squeeze(0)
        image_pil = transforms.ToPILImage()(image_tensor)

        image_filename = f"epoch_{epoch}_generated_image_{i}.png"
        image_path = os.path.join(save_folder, image_filename)

        # 이미지 저장
        image_pil.save(image_path)

    if epoch == (epochs):  # 마지막 에포크일 경우 ZIP 파일로 압축
        zip_images(save_folder, result_name)

    return save_folder

    if show:
        # 필요하면 생성된 이미지를 화면에 출력하는 코드를 여기에 추가할 수 있습니다.
        pass

def my_collate(batch):
    return batch

def zip_images(folder_path, output_zip_name):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print("폴더 내에 이미지 파일이 없습니다.")
        return

    # ZIP 파일 생성
    with zipfile.ZipFile(os.path.join(folder_path, output_zip_name), 'w') as zipf:
        for image_file in image_files:
            file_path = os.path.join(folder_path, image_file)
            zipf.write(file_path, image_file)

    print("ZIP 파일 생성 완료.")

# 데이터 증강 코드
def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5), # 0.5확률로 수평 뒤집기
        albu.ShiftScaleRotate(scale_limit=[0.95, 1.05], rotate_limit=45, shift_limit=0.005, p=1, border_mode=0 ), # 이동, 확대, 축소, 회전
        albu.PadIfNeeded(min_height=400, min_width=400, always_apply=True, border_mode=0),# 이미지가 지정크기보다 작으면, 패딩함

        albu.GaussNoise(p=0.2), # 0.2확률로 가우시안 노이즈 추가
        albu.Perspective(p=0.3), # 0.3 확률로 투시 변환

        albu.OneOf(    # 주어진 증강 중 하나를 확률로 CLANE, 랜덤 밝기, 랜덤 감마 등을 적용
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.8,
        ),

        albu.OneOf( #이미지 날카롭게, 윤곽 강조, 블러 정도 조절
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.8,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1), #대비
                albu.HueSaturationValue(p=1), #채도
            ],
            p=0.8,
        ),
    ]
    return albu.Compose(train_transform)

if __name__ == '__main__':
    app.run(debug=False)
