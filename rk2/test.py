import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import v2

IMG_SIZE = 300
NUM_CLASSES = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

test_transforms = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_model(num_classes):
    model = models.efficientnet_b3(weights=None)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    test_dir = os.path.join(args.path, 'test')

    model_path = 'model.pth'
    model = get_model(NUM_CLASSES)

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    # Собираем файлы
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    image_files.sort()

    filenames = []
    predictions = []

    with torch.no_grad():
        for img_name in tqdm(image_files):
            img_path = os.path.join(test_dir, img_name)

            try:
                image = Image.open(img_path).convert("RGB")
                img_tensor = test_transforms(image).unsqueeze(0).to(DEVICE)

                image_flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
                img_flip_tensor = test_transforms(image_flipped).unsqueeze(0).to(DEVICE)

                output_orig = model(img_tensor)
                output_flip = model(img_flip_tensor)

                avg_output = (output_orig + output_flip) / 2.0

                _, pred = torch.max(avg_output, 1)

                filenames.append(img_name)
                predictions.append(pred.item())

            except Exception as e:
                print(f"Сбой на файле {img_name}: {e}")

    save_path = 'label_test.csv'
    df = pd.DataFrame({
        'filenames': filenames,
        'label': predictions
    })
    df.to_csv(save_path, index=False)