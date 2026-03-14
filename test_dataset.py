from dataset import FloodDataset

dataset = FloodDataset("data/image","data/label","split/train.txt")

print("Dataset size:", len(dataset))

img, mask = dataset[0]

print("Image shape:", img.shape)
print("Mask shape:", mask.shape)