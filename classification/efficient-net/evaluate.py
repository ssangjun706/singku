import torch
from torch.utils.data import DataLoader

from model import EfficientNet
from data import HAM10000
import constants as args

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, test_loader, device):
    accuracy = 0
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for X, y in tqdm(test_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            accuracy += (pred.argmax(1) == y.argmax(1)).float().sum().item()

    accuracy /= len(test_loader.dataset)
    return accuracy


def main():
    dataset = HAM10000(
        image_dir=args.test_dir,
        label_path=args.label_path,
        resize=args.resize,
    )

    test_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = EfficientNet(args.num_classes)
    accuracy = evaluate(model, test_loader, device)
    print(f"Accuracy: {accuracy :.8f}")


if __name__ == "__main__":
    main()
