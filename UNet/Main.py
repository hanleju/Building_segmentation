import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader , random_split

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from UNet_3Plus import UNet_3Plus

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = "./open"+self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.data.iloc[idx, 2]
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


transform = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1)),
        A.RandomScale(p=0.5, scale_limit=(-0.2, 0.2)),
        A.CropNonEmptyMaskIfExists(p=1,height=224,width=224),
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ]
)
test_transform = A.Compose(
    [
        A.Normalize(),
        ToTensorV2()
    ]
)
train_dataset = SatelliteDataset(csv_file='./train.csv', transform=transform)
dataset_size = len(train_dataset)
train_ratio = 0.9
train_size = int(train_ratio * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=0)

def calculate_accuracy(pred_masks, true_masks, threshold=0.5):
    # 0.5를 기준으로 이진화하여 예측된 마스크와 실제 마스크가 일치하는 픽셀 개수 계산
    pred_masks = (pred_masks > threshold).astype(np.uint8)
    true_masks = true_masks.astype(np.uint8)

    # 예측된 마스크와 실제 마스크가 일치하는 픽셀 개수 계산
    intersection = np.logical_and(pred_masks, true_masks).sum()
    union = np.logical_or(pred_masks, true_masks).sum()

    # 정확도 계산
    accuracy = intersection / union
    return accuracy


# model 초기화
model = UNet_3Plus().to(device)
model.load_state_dict(torch.load('BestUNet3++5.pth'))
# loss function과 optimizer 정의
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)


best_val_loss = float('inf')
best_model_state = None

# training loop
for epoch in range(15):  # 10 에폭 동안 학습합니다.
    model.train()
    epoch_loss = 0
    for images, masks in tqdm(train_loader):
        images = images.float().to(device)
        masks = masks.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.unsqueeze(1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    model.eval()
    val_loss = 0
    test_accuracy = 0
    total_samples = 0
    with torch.no_grad():
        for val_images, val_masks in tqdm(val_loader):
            val_images = val_images.float().to(device)
            val_masks = val_masks.float().to(device)
            val_outputs = model(val_images)
            pred_masks = torch.sigmoid(val_outputs).cpu().numpy()
            true_masks = val_masks.cpu().numpy()

            # 정확도 계산하여 누적
            batch_accuracy = calculate_accuracy(pred_masks, true_masks)
            test_accuracy += batch_accuracy * val_images.shape[0]
            total_samples += val_images.shape[0]
            loss = criterion(val_outputs, val_masks.unsqueeze(1))
            val_loss += loss.item()
        test_accuracy /= total_samples
    print(f'Epoch {epoch + 1}, Train Loss: {epoch_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}')
    print(f"Validation Accuracy: {test_accuracy:.4f}")
    if epoch ==19:
        torch.save(model.state_dict(),'LastUNet3++.pth')
    # 현재 에폭에서의 검증 손실이 가장 작은 경우 모델 파라미터를 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()


torch.save(best_model_state, 'BestUNet3++.pth')


test_dataset = SatelliteDataset(csv_file='test.csv', transform=test_transform, infer=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

with torch.no_grad():
    model.eval()
    result = []
    for images in tqdm(test_dataloader):
        images = images.float().to(device)

        outputs = model(images)
        masks = torch.sigmoid(outputs).cpu().numpy()
        masks = np.squeeze(masks, axis=1)
        masks = (masks > 0.5).astype(np.uint8)  # Threshold = 0.5

        for i in range(len(images)):
            mask_rle = rle_encode(masks[i])
            if mask_rle == '':  # 예측된 건물 픽셀이 아예 없는 경우 -1
                result.append(-1)
            else:
                result.append(mask_rle)
submit = pd.read_csv('./open/submission.csv')
submit['mask_rle'] = result

submit.to_csv('result.csv', index=False)