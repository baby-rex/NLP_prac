# 🔧 Full CNN Transfer Learning Project with ResNet50 + Optuna + Grad-CAM

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import optuna
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.transforms.functional import to_pil_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# STEP 1: SET PATHS
base_dir = "/Users//Users/tanmayjaipuriar/Downloads/Mac learn/Prac/archive"
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# STEP 2: DEFINE TRANSFORMS (Resize, Normalize)
mean = [0.485, 0.456, 0.406] 
std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# STEP 3: LOAD DATASETS & DATALOADERS
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)

# STEP 4: MODEL CREATION FUNCTION FOR OPTUNA

def define_model(trial):
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze base

    # Get trial-suggested hyperparameters
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    hidden_units = trial.suggest_int("hidden_units", 256, 1024, step=128)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, num_classes)
    )

    return model

# STEP 5: TRAINING FUNCTION WITH VAL ACCURACY

def train_model(model, optimizer, criterion, epochs=5):
    model = model.cuda()
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        for images, labels in tqdm(train_loader):
            images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        best_val_acc = max(best_val_acc, val_acc)
        print(f"Epoch {epoch+1}: Val Accuracy = {val_acc:.4f}")

    return best_val_acc

# STEP 6: OPTUNA OBJECTIVE FUNCTION

def objective(trial):
    model = define_model(trial)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    return train_model(model, optimizer, criterion)

# 🔍 STEP 7: RUN OPTUNA
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
print("\nBest hyperparameters:", study.best_params)

#  STEP 8: FINAL MODEL WITH BEST PARAMS
best_params = study.best_params

final_model = models.resnet50(pretrained=True)
for param in final_model.parameters():
    param.requires_grad = False

num_ftrs = final_model.fc.in_features
final_model.fc = nn.Sequential(
    nn.Linear(num_ftrs, best_params['hidden_units']),
    nn.ReLU(),
    nn.Dropout(best_params['dropout']),
    nn.Linear(best_params['hidden_units'], num_classes)
)

final_model = final_model.cuda()
optimizer = optim.Adam(final_model.fc.parameters(), lr=best_params['lr'])
criterion = nn.CrossEntropyLoss()

train_model(final_model, optimizer, criterion, epochs=5)

#  STEP 9: TEST ACCURACY
final_model.eval()
total, correct = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = final_model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"\nTest Accuracy: {correct / total:.4f}")

#  STEP 10: INFERENCE + GRAD-CAM

def predict_and_visualize(img_path):
    image = Image.open(img_path).convert('RGB')
    orig = transforms.Resize((224, 224))(image)
    input_tensor = val_test_transform(orig).unsqueeze(0).cuda()

    target_layers = [final_model.layer4[-1]]
    cam = GradCAM(model=final_model, target_layers=target_layers, use_cuda=True)
    targets = [ClassifierOutputTarget(0)]  # Class index (optional)

    grayscale_cam = cam(input_tensor=input_tensor)[0]
    rgb_img = np.array(orig).astype(np.float32) / 255
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    plt.imshow(visualization)
    plt.title("Grad-CAM Visualization")
    plt.axis('off')
    plt.show()

    # Prediction
    output = final_model(input_tensor)
    _, predicted_class = torch.max(output, 1)
    print(f"Predicted Bird Species: {class_names[predicted_class.item()]}")

# Example usage:
# predict_and_visualize("/Users/yourname/Downloads/sample.jpg")
