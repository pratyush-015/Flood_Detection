import torch
import rasterio
import numpy as np
import os
import pandas as pd
from model import get_model
from rle import mask_to_rle
import csv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model()
model.load_state_dict(torch.load("flood_model.pth"))
model = model.to(device)
model.eval()


test_ids = open("split/test.txt").read().splitlines()

results = []


for img_id in test_ids:

    img_path = os.path.join("data/prediction/image", img_id + "_image.tif")

    with rasterio.open(img_path) as src:
        image = src.read().astype(np.float32)

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

    image = torch.tensor(image).unsqueeze(0).to(device)

    with torch.no_grad():

        pred1 = model(image)

        pred2 = torch.flip(model(torch.flip(image,[3])),[3])
        pred3 = torch.flip(model(torch.flip(image,[2])),[2])

        pred = (pred1 + pred2 + pred3) / 3

        pred = torch.sigmoid(pred)

        mask = (pred > 0.5).cpu().numpy()[0,0]

    rle = mask_to_rle(mask)

    results.append((img_id, rle))


df = pd.DataFrame(results, columns=["id", "rle_mask"])
df.to_csv("submission.csv", index=False, encoding="utf-8", quoting=csv.QUOTE_NONE,
    escapechar=" ")

print("submission.csv created")