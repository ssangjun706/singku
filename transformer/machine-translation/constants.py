import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--lr', type=float, default=5e-6)
parser.add_argument('--h_dim', type=int, default=512)
parser.add_argument('--seq_len', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--vocab_size', type=int, default=128257)
parser.add_argument('--ignore_index', type=int, default=128004)
parser.add_argument('--warmup', type=int, default=10)

parser.add_argument('--checkpoint_path', type=str, default="model/checkpoint_model.pt")
parser.add_argument('--checkpoint_per_epoch', type=int, default=5)
parser.add_argument('--tokenizer_path', type=str, default="/home/sangjun/model/llama-3.2-11b-vision/")

args = parser.parse_args()
