import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from model import PersonDetectorCNN

class PersonDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')

        image = Image.open(img_path).convert('L')
        has_person = 1.0 if os.path.exists(label_path) and os.path.getsize(label_path) > 0 else 0.0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([has_person], dtype=torch.float32)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"matériel : {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = PersonDataset("./data/processed/images/train", "./data/processed/labels/train", transform=transform)
    val_dataset = PersonDataset("./data/processed/images/val", "./data/processed/labels/val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = PersonDetectorCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    print(f'debut entrainement en {epochs} epoques')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoque {epoch+1} : perte = {running_loss/len(train_loader):.4f}")

    os.makedirs("./models", exist_ok=True)
    torch.save(model.state_dict(), "./models/person_detector_fp32.pth")
    print("FP32 save dans ./models/person_detector_fp32.pth")

if __name__ == "__main__":
    train()