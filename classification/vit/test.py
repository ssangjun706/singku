import os
import torch
from tqdm import tqdm
from data import HAM10000
from torch.utils.data import DataLoader

from model.vit import ViT
import torch.nn as nn

import constants as args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def test(model, data_loader, device):
    loss_fn = nn.CrossEntropyLoss().to(device)
    avg_loss, accr = 0, 0

    model.eval()
    with tqdm(data_loader, desc="Test") as pbar:
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            avg_loss += loss_fn(pred, y).item()
            accr += (pred.argmax(1) == y.argmax(1)).type(torch.float32).sum().item()

    avg_loss /= len(data_loader)
    accr /= len(data_loader.dataset)
    return avg_loss, accr


def main():
    dataset = HAM10000(
        image_dir=args.image_dir,
        label_path=args.label_path,
        resize=args.resize,
        train=False,
        ratio=0.81,
    )

    test_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = ViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        h_dim=args.h_dim,
        mlp_dim=args.mlp_dim,
        num_classes=args.num_classes,
    )

    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    model = nn.DataParallel(model)
    model.to(device)

    _, accr = test(model, test_loader, device)
    print(f"Accuracy: {accr * 100 :.4f}%")


if __name__ == "__main__":
    main()
