import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import CustomDataset
from model import TransformerEncoder


num_sequences = 64
h_dim = 768
checkpoint_path = "./checkpoint_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

test_path = "data/amazon_test.csv"
test_data = CustomDataset(test_path, num_sequences)
test_loader = DataLoader(test_data, batch_size=1024, drop_last=True, num_workers=32)

model = TransformerEncoder(h_dim=h_dim, num_sequences=num_sequences).to(device)
model.load_state_dict(checkpoint)
model.eval()

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)


def main():
    test_loss, accuracy = 0, 0
    with torch.no_grad():
        for tokens, masks, label in tqdm(test_loader, desc="Test"):
            tokens, masks, label = (
                tokens.to(device),
                masks.to(device),
                label.to(device),
            )
            pred = model(tokens, masks)
            test_loss += loss_fn(pred, label).item()
            accuracy += (
                (pred.argmax(1) == label.argmax(1)).type(torch.float).sum().item()
            )

        test_loss /= len(test_loader)
        accuracy /= len(test_loader.dataset)

        print(f"Loss: {test_loss:.4f}")
        print(f"accuracy: {accuracy * 100:.2f}")


if __name__ == "__main__":
    main()
