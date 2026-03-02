import torch
import os
from tqdm import tqdm


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs , labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs,1)
            total += labels.size(0)
            correct += (preds==labels).sum().item()
    return correct/total

def train(model, train_loader, val_loader, criterion, optimizer,
          device, epochs, checkpoint_dir, model_name):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        val_acc = evaluate(model, val_loader, device)
        print(f"\nEpoch {epoch+1} | Loss: {running_loss:.4f} | Val Acc: {val_acc:.4f}")
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch+1,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_acc
        }, checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_path}")
