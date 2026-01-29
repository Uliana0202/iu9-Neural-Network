import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

DATA_ROOT = 'data/data'
CSV_FILE = 'labels.csv'
IMG_SIZE = 300 # Теория: нужно "разглядеть" узоры
NUM_CLASSES = 50

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


train_transforms = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    # Теория: бабочки симметричные, поэтому подойдут повороты и отражения
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),

    # Теория: так как цвет критичен, играемся только с яркостью и контрастностью
    v2.ColorJitter(brightness=0.2, contrast=0.2),

    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ButterflyDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, mode='train'):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row['filenames']
        img_path = os.path.join(self.root_dir, file_name)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.mode == 'test':
            return image, file_name
        else:
            label = int(row['label'])
            return image, torch.tensor(label, dtype=torch.long)


def get_model(num_classes, freeze_backbone=True):
    model = models.efficientnet_b3(weights='DEFAULT')

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )

    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Training")

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples

        pbar.set_postfix({'loss': f'{epoch_loss:.4f}', 'acc': f'{epoch_acc:.4f}'})

    return epoch_loss, epoch_acc.item()


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples

    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='macro')

    return epoch_loss, val_acc, val_f1


if __name__ == '__main__':
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    full_df = pd.read_csv(CSV_FILE)

    train_df, val_df = train_test_split(full_df, test_size=0.1, stratify=full_df['label'], random_state=42)
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    # Сколько картинок в каждом классе
    counts = train_df['label'].value_counts().sort_index()
    weights_per_class = 1.0 / counts

    # Собираем веса для каждой картинки
    samples_weights = []
    for label in train_df['label']:
        samples_weights.append(weights_per_class[label])

    samples_weights = torch.DoubleTensor(samples_weights)
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(train_df), replacement=True)

    train_ds = ButterflyDataset(train_df, DATA_ROOT, transform=train_transforms, mode='train')
    val_ds = ButterflyDataset(val_df, DATA_ROOT, transform=val_transforms, mode='val')

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    # ОБУЧЕНИЕ ГОЛОВЫ
    model = get_model(NUM_CLASSES, freeze_backbone=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    best_f1 = 0.0
    history = []

    for epoch in range(15):
        print(f"\nPart 1: Epoch {epoch + 1}/15")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_model_part1.pth')

    # РАЗМОРОЗКА И ДООБУЧЕНИЕ
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    model.load_state_dict(torch.load('best_model_part1.pth'))

    for epoch in range(25):
        print(f"\nPart 2: Epoch {epoch + 1}/25")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'model.pth')

    print(f"\nЛучший F1: {best_f1:.4f}")

'''
Train size: 4459, Val size: 496

Part 1: Epoch 1/15
Training: 100%|██████████| 140/140 [00:26<00:00,  5.19it/s, loss=1.6070, acc=0.7416]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.41it/s]
Val Loss: 1.2547 | Acc: 0.8548 | F1: 0.8532

Part 1: Epoch 2/15
Training: 100%|██████████| 140/140 [00:27<00:00,  5.12it/s, loss=1.2164, acc=0.8719]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.61it/s]
Val Loss: 1.2335 | Acc: 0.8629 | F1: 0.8572

Part 1: Epoch 3/15
Training: 100%|██████████| 140/140 [00:26<00:00,  5.20it/s, loss=1.1705, acc=0.8928]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.60it/s]
Val Loss: 1.2268 | Acc: 0.8690 | F1: 0.8635

Part 1: Epoch 4/15
Training: 100%|██████████| 140/140 [00:27<00:00,  5.18it/s, loss=1.1345, acc=0.9011]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.53it/s]
Val Loss: 1.1868 | Acc: 0.8891 | F1: 0.8880

Part 1: Epoch 5/15
Training: 100%|██████████| 140/140 [00:26<00:00,  5.19it/s, loss=1.1011, acc=0.9112]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.46it/s]
Val Loss: 1.1694 | Acc: 0.8972 | F1: 0.8947

Part 1: Epoch 6/15
Training: 100%|██████████| 140/140 [00:26<00:00,  5.19it/s, loss=1.1069, acc=0.9121]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.48it/s]
Val Loss: 1.1240 | Acc: 0.9133 | F1: 0.9119

Part 1: Epoch 7/15
Training: 100%|██████████| 140/140 [00:26<00:00,  5.20it/s, loss=1.0769, acc=0.9211]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.56it/s]
Val Loss: 1.1465 | Acc: 0.8911 | F1: 0.8884

Part 1: Epoch 8/15
Training: 100%|██████████| 140/140 [00:26<00:00,  5.19it/s, loss=1.0542, acc=0.9368]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.55it/s]
Val Loss: 1.1378 | Acc: 0.8972 | F1: 0.8956

Part 1: Epoch 9/15
Training: 100%|██████████| 140/140 [00:27<00:00,  5.18it/s, loss=1.0587, acc=0.9305]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.56it/s]
Val Loss: 1.1390 | Acc: 0.8891 | F1: 0.8861

Part 1: Epoch 10/15
Training: 100%|██████████| 140/140 [00:26<00:00,  5.19it/s, loss=1.0552, acc=0.9269]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.55it/s]
Val Loss: 1.1373 | Acc: 0.8911 | F1: 0.8876

Part 1: Epoch 11/15
Training: 100%|██████████| 140/140 [00:26<00:00,  5.19it/s, loss=1.0313, acc=0.9433]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.61it/s]
Val Loss: 1.1273 | Acc: 0.9012 | F1: 0.8975

Part 1: Epoch 12/15
Training: 100%|██████████| 140/140 [00:26<00:00,  5.20it/s, loss=1.0179, acc=0.9468]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.53it/s]
Val Loss: 1.1078 | Acc: 0.9052 | F1: 0.9026

Part 1: Epoch 13/15
Training: 100%|██████████| 140/140 [00:26<00:00,  5.20it/s, loss=1.0081, acc=0.9493]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.50it/s]
Val Loss: 1.1196 | Acc: 0.9133 | F1: 0.9081

Part 1: Epoch 14/15
Training: 100%|██████████| 140/140 [00:26<00:00,  5.20it/s, loss=1.0196, acc=0.9412]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.57it/s]
Val Loss: 1.1059 | Acc: 0.9153 | F1: 0.9115

Part 1: Epoch 15/15
Training: 100%|██████████| 140/140 [00:26<00:00,  5.19it/s, loss=1.0129, acc=0.9491]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.56it/s]
Val Loss: 1.1209 | Acc: 0.9052 | F1: 0.9014

Part 2: Epoch 1/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.9987, acc=0.9500]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.56it/s]
Val Loss: 1.0108 | Acc: 0.9456 | F1: 0.9442

Part 2: Epoch 2/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.9400, acc=0.9670]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.53it/s]
Val Loss: 0.9825 | Acc: 0.9435 | F1: 0.9431

Part 2: Epoch 3/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.8949, acc=0.9789]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.58it/s]
Val Loss: 0.9665 | Acc: 0.9496 | F1: 0.9488

Part 2: Epoch 4/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.8675, acc=0.9856]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.49it/s]
Val Loss: 0.9511 | Acc: 0.9536 | F1: 0.9542

Part 2: Epoch 5/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.61it/s, loss=0.8426, acc=0.9915]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.57it/s]
Val Loss: 0.9436 | Acc: 0.9496 | F1: 0.9492

Part 2: Epoch 6/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.8315, acc=0.9910]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.43it/s]
Val Loss: 0.9340 | Acc: 0.9516 | F1: 0.9508

Part 2: Epoch 7/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.8267, acc=0.9915]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.46it/s]
Val Loss: 0.9199 | Acc: 0.9577 | F1: 0.9580

Part 2: Epoch 8/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.8118, acc=0.9948]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.53it/s]
Val Loss: 0.9153 | Acc: 0.9597 | F1: 0.9599

Part 2: Epoch 9/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.7983, acc=0.9971]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.60it/s]
Val Loss: 0.9115 | Acc: 0.9577 | F1: 0.9575

Part 2: Epoch 10/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.61it/s, loss=0.7980, acc=0.9973]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.54it/s]
Val Loss: 0.9057 | Acc: 0.9556 | F1: 0.9551

Part 2: Epoch 11/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.7906, acc=0.9964]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.57it/s]
Val Loss: 0.9145 | Acc: 0.9496 | F1: 0.9492

Part 2: Epoch 12/25
Training: 100%|██████████| 140/140 [01:28<00:00,  1.59it/s, loss=0.7907, acc=0.9955]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.48it/s]
Val Loss: 0.8992 | Acc: 0.9577 | F1: 0.9569

Part 2: Epoch 13/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.7808, acc=0.9971]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.44it/s]
Val Loss: 0.8969 | Acc: 0.9597 | F1: 0.9591

Part 2: Epoch 14/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.7742, acc=0.9987]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.43it/s]
Val Loss: 0.8960 | Acc: 0.9577 | F1: 0.9575

Part 2: Epoch 15/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.61it/s, loss=0.7732, acc=0.9982]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.55it/s]
Val Loss: 0.8860 | Acc: 0.9617 | F1: 0.9617

Part 2: Epoch 16/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.7678, acc=0.9991]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.56it/s]
Val Loss: 0.8799 | Acc: 0.9637 | F1: 0.9639

Part 2: Epoch 17/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.7679, acc=0.9975]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.49it/s]
Val Loss: 0.8885 | Acc: 0.9597 | F1: 0.9597

Part 2: Epoch 18/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.7658, acc=0.9993]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.61it/s]
Val Loss: 0.8840 | Acc: 0.9617 | F1: 0.9616

Part 2: Epoch 19/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.7636, acc=0.9982]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.57it/s]
Val Loss: 0.8927 | Acc: 0.9577 | F1: 0.9565

Part 2: Epoch 20/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.7612, acc=0.9984]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.49it/s]
Val Loss: 0.8884 | Acc: 0.9536 | F1: 0.9524

Part 2: Epoch 21/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.7604, acc=0.9984]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.58it/s]
Val Loss: 0.8790 | Acc: 0.9597 | F1: 0.9602

Part 2: Epoch 22/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.7591, acc=0.9993]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.56it/s]
Val Loss: 0.8866 | Acc: 0.9577 | F1: 0.9580

Part 2: Epoch 23/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.7573, acc=0.9987]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.58it/s]
Val Loss: 0.8815 | Acc: 0.9597 | F1: 0.9603

Part 2: Epoch 24/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.7580, acc=0.9982]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.52it/s]
Val Loss: 0.8821 | Acc: 0.9597 | F1: 0.9598

Part 2: Epoch 25/25
Training: 100%|██████████| 140/140 [01:27<00:00,  1.60it/s, loss=0.7556, acc=0.9991]
Validating: 100%|██████████| 16/16 [00:02<00:00,  5.57it/s]
Val Loss: 0.8803 | Acc: 0.9597 | F1: 0.9598

Лучший F1: 0.9639
'''