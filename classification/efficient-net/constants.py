train_dir = "../../data/ham10000/imgs/train"
val_dir = "../../data/ham10000/imgs/val"
test_dir = "../../data/ham10000/imgs/test"
label_path = "../../data/ham10000/label.csv"

checkpoint_path = "checkpoint/checkpoint_model.pt"
model_path = "checkpoint/best_model.pt"

batch_size = 128
epochs = 20
lr = 5e-4

resize = 400
num_classes = 7

num_workers = 32
use_wandb = True
use_checkpoint = False
