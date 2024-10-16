image_dir = "../../data/ham10000/imgs"
label_path = "../../data/ham10000/label.csv"

batch_size = 128
epochs = 25
lr = 5e-4

patch_size = 16
h_dim = 768
mlp_dim = 3072
resize = 450
image_size = 224
num_classes = 7

num_workers = 32
use_wandb = True
use_checkpoint = False

model = "vit"

if model == "vit":
    checkpoint = "checkpoint/checkpoint_model_vit.pt"
else:
    checkpoint = "checkpoint/checkpoint_model_resnet.pt"
