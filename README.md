# Flood Detection using Satellite Imagery

This project detects flooded regions using multi-sensor satellite data
(SAR + multispectral imagery).

Developed for **ANRF AISEHack - Flood Detection Challenge (IBM)**.

## Features
- Multi-modal satellite data fusion
- NDWI and SAR feature engineering
- EfficientNet-based UNet segmentation
- Test-time augmentation
- Run-length encoding for submission

## Dataset
- 79 training patches
- 19 test patches
- 512x512 resolution
- Channels:
  - SAR HH
  - SAR HV
  - Green
  - Red
  - NIR
  - SWIR

## Model
UNet with EfficientNet-B3 encoder.

Additional engineered features:
- NDWI
- SAR Ratio

Total input channels: 8

## Training

```bash
python train.py
