import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset_split import create_stratified_split

def main():
    # Loading pre-trained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freezing the first 3 blocks (early layers) - Strategy 4
    ct = 0
    for child in model.children():
        ct += 1
        if ct < 7:  # First 6 children modules are frozen
            for param in child.parameters():
                param.requires_grad = False

    # Replacing the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)  # 10 classes for iNaturalist

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader, val_loader, test_loader, class_names = create_stratified_split(
        dataset_dir="inaturalist_12k",
        img_size=224,
        batch_size=64,
        val_size=0.2,
        subset_fraction=0.25,  # Using 25% of the data for training
    )

    criterion = nn.CrossEntropyLoss()
    # Using different learning rates for frozen and trainable layers
    optimizer = optim.Adam([
        {'params': model.fc.parameters(), 'lr': 0.001},
        {'params': model.layer4.parameters(), 'lr': 0.0001}
    ])

    num_epochs = 5
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

if __name__ == '__main__':
    # This is necessary for Windows to properly handle multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    torch.multiprocessing.freeze_support()
    main()
