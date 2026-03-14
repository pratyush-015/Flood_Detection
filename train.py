import torch
from torch.utils.data import DataLoader
from dataset import FloodDataset
from model import get_model
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dice_loss(pred, target):

    pred = torch.sigmoid(pred)

    smooth = 1e-6

    intersection = (pred * target).sum()

    return 1 - ((2 * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))


def focal_loss(inputs, targets, alpha=0.8, gamma=2):

    BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-BCE)

    focal = alpha * (1 - pt) ** gamma * BCE

    return focal.mean()


def main():

    train_dataset = FloodDataset(
        "data/image",
        "data/label",
        "split/train.txt",
        augment=True
    )

    val_dataset = FloodDataset(
        "data/image",
        "data/label",
        "split/val.txt"
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    model = get_model().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 40

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        print(f"\nEpoch {epoch+1}/{epochs}")

        for images, masks in tqdm(train_loader):

            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)

            loss = focal_loss(preds, masks) + dice_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print("Average Loss:", avg_loss)

    torch.save(model.state_dict(), "flood_model.pth")

    print("Model saved")


if __name__ == "__main__":
    main()