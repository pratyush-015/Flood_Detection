import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import os
import albumentations as A


class FloodDataset(Dataset):

    def __init__(self, image_dir, label_dir, file_list, augment=False):

        with open(file_list) as f:
            self.ids = f.read().splitlines()

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.augment = augment

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        img_id = self.ids[idx]

        img_path = os.path.join(self.image_dir, img_id + "_image.tif")
        label_path = os.path.join(self.label_dir, img_id + "_label.tif")

        with rasterio.open(img_path) as src:
            image = src.read(out_dtype="float32")

        image = np.clip(image, 0, 10000) / 10000.0

        SAR_HH = image[0]
        SAR_HV = image[1]
        GREEN = image[2]
        RED = image[3]
        NIR = image[4]
        SWIR = image[5]

        ndwi = (GREEN - NIR) / (GREEN + NIR + 1e-6)
        sar_ratio = SAR_HH / (SAR_HV + 1e-6)

        image = np.stack([
            SAR_HH,
            SAR_HV,
            GREEN,
            RED,
            NIR,
            SWIR,
            ndwi,
            sar_ratio
        ])

        with rasterio.open(label_path) as src:
            mask = src.read(1)

        mask = (mask > 0).astype(np.float32)

        if self.augment:

            aug = self.transform(
                image=image.transpose(1,2,0),
                mask=mask
            )

            image = aug["image"].transpose(2,0,1)
            mask = aug["mask"]

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask