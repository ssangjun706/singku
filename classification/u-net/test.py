import os
import argparse
import torch
import torch.nn.functional
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from model import UNet
from data import HAM10000
from utils import normalize, iou_metric

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--num_workers", default=32, type=int)

parser.add_argument("--gpu_num", default="0,1,2,3,4,5,6,7", type=str)
parser.add_argument("--interpolate", default="bilinear", type=str)

parser.add_argument("--image_dir", default="dataset/imgs", type=str)
parser.add_argument("--mask_dir", default="dataset/masks", type=str)


def main(args):
    ham = HAM10000(image_dir=args.image_dir, mask_dir=args.mask_dir, train=False)
    test_loader = DataLoader(ham, batch_size=args.batch_size, drop_last=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    model.load_state_dict(torch.load("model/best_model.pt", map_location=device))
    model = nn.DataParallel(model).to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    iou, loss = 0, 0
    with tqdm(test_loader) as pbar:
        for X, y in pbar:
            X, y = normalize(X.to(device)), y.to(device)
            pred = model(X)
            pred = F.interpolate(pred, size=(450, 600), mode=args.interpolate)
            loss += loss_fn(pred, y).item()
            iou += iou_metric(pred, y)

    iou /= len(test_loader)
    loss /= len(test_loader)

    print(f"Average Loss: {loss :.4f}")
    print(f"Average IoU: {iou :.4f}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
