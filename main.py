import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

from train import train
from train import evaluate
from model import get_model
from data_loader import get_dataloaders

# class Config:
#     data_dir = "/home/student/nn_workshop/transfer-learning-workshop/data/hymenoptera_data"  # Update path if needed
#     model_name = "resnet18"                       # resnet18 or resnet34
#     batch_size = 32
#     epochs = 5
#     lr = 0.001
#     feature_extract = True
#     checkpoint_dir = "checkpoints"

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--feature_extract", type=bool, default=True)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    return parser.parse_args()

args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size)
model = get_model(args.model_name, num_classes=2, feature_extract=args.feature_extract)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

print("Starting Training...\n")
train(model, train_loader, val_loader, criterion, optimizer,
      device, args.epochs, args.checkpoint_dir, args.model_name)

print("\nEvaluating on Test Set...")
test_acc = evaluate(model, test_loader, device)
print(f"Test Accuracy: {test_acc:.4f}")