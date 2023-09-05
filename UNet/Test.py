
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# 결과 파일 읽기
result_df = pd.read_csv('Myresult.csv')

# 원하는 인덱스 설정 (예: 10번째 이미지)
desired_index = 434 #   인덱스는 0부터 시작하므로 10번째 이미지의 인덱스는 9입니다.

# 원하는 인덱스의 이미지와 마스크 시각화
mask_rle = result_df.loc[desired_index, 'mask_rle']
if mask_rle != -1:
    mask = rle_decode(mask_rle, (224, 224))
    plt.imshow(mask, cmap='gray')
    plt.title(f'Image {desired_index + 1}: Mask')
else:
    print(f'Image {desired_index}: NULL (No predicted mask)')
plt.show()

# import os
# import cv2
# import pandas as pd
# import numpy as np
# import segmentation_models_pytorch as smp
# import torch
# from torch.utils.data import Dataset, DataLoader
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # RLE 디코딩 함수
# def rle_decode(mask_rle, shape):
#     s = mask_rle.split()
#     starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
#     starts -= 1
#     ends = starts + lengths
#     img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
#     for lo, hi in zip(starts, ends):
#         img[lo:hi] = 1
#     return img.reshape(shape)
#
# # RLE 인코딩 함수
# def rle_encode(mask):
#     pixels = mask.flatten()
#     pixels = np.concatenate([[0], pixels, [0]])
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
#     runs[1::2] -= runs[::2]
#     return ' '.join(str(x) for x in runs)
#
# class SatelliteDataset(Dataset):
#     def __init__(self, csv_file, transform=None, infer=False):
#         self.data = pd.read_csv(csv_file)
#         self.transform = transform
#         self.infer = infer
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         img_path = "D:/open" + self.data.iloc[idx, 1]
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         if self.infer:
#             if self.transform:
#                 image = self.transform(image=image)['image']
#             return image
#
#         mask_rle = self.data.iloc[idx, 2]
#         mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))
#
#         if self.transform:
#             augmented = self.transform(image=image, mask=mask)
#             image = augmented['image']
#             mask = augmented['mask']
#
#         return image, mask
#
# transform = A.Compose(
#     [
#         A.Resize(224, 224),
#         A.Normalize(),
#         ToTensorV2()
#     ]
# )
#
# test_dataset = SatelliteDataset(csv_file='./test.csv', transform=transform, infer=True)
# test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
#
# ENCODER = "resnet101"
# ENCODER_WEIGHTS = "imagenet"
# model = smp.DeepLabV3Plus(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, in_channels=3, classes=1).to(device)
# model.load_state_dict(torch.load('Deeplabv3.pth'))
#
# with torch.no_grad():
#     model.eval()
#     result = []
#     for images in tqdm(test_dataloader):
#         images = images.float().to(device)
#
#         outputs = model(images)
#         masks = torch.sigmoid(outputs).cpu().numpy()
#         masks = np.squeeze(masks, axis=1)
#         masks = (masks > 0.35).astype(np.uint8)  # Threshold = 0.35
#
#         for i in range(len(images)):
#             mask_rle = rle_encode(masks[i])
#             if mask_rle == '':  # 예측된 건물 픽셀이 아예 없는 경우 -1
#                 result.append(-1)
#             else:
#                 result.append(mask_rle)
#
# mini_mask_rle = result[6]
# if mini_mask_rle != -1:
#   mini_mask = rle_decode(mini_mask_rle, (224, 224))
#   plt.imshow(mini_mask, interpolation='nearest')
# else:
#   print('NULL')
