import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from timm import create_model
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix, classification_report
from matplotlib import pyplot as plt
from PIL import Image

DATASET_PATH = "./kvasir-dataset-v2"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = ImageFolder(DATASET_PATH, transform=transform)
num_classes = len(dataset.classes)

print("Classes:", dataset.classes)
print("Number of classes:", num_classes)
print("Total images:", len(dataset))

indices = list(range(len(dataset)))
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.3,
    random_state=42,
    stratify=dataset.targets
)

val_idx, test_idx = train_test_split(
    test_idx,
    test_size=0.66,
    random_state=42,
    stratify=[dataset.targets[i] for i in test_idx]
)

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

class HybridEfficientNetViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientnet = create_model(
            "efficientnet_b3",
            pretrained=True,
            num_classes=num_classes
        )
        self.vit = create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=num_classes
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * num_classes, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        efficientnet_features = self.efficientnet(x)
        vit_features = self.vit(x)
        combined_features = torch.cat(
            (efficientnet_features, vit_features),
            dim=1
        )
        return self.fc(combined_features)

model = HybridEfficientNetViT(num_classes).to(DEVICE)

class ClassBalancedLoss(nn.Module):
    def __init__(self, beta, num_classes):
        super().__init__()
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits, labels):
        class_counts = np.bincount(
            labels.detach().cpu().numpy(),
            minlength=self.num_classes
        )
        effective_num = 1.0 - np.power(self.beta, class_counts)
        weights = (1.0 - self.beta) / (effective_num + 1e-8)
        weights = weights / np.sum(weights)
        weights = torch.tensor(
            weights,
            dtype=torch.float32,
            device=logits.device
        )
        return nn.CrossEntropyLoss(weight=weights)(logits, labels)

loss_fn = ClassBalancedLoss(
    beta=0.999,
    num_classes=num_classes
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR
)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total

def validate_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    preds_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    mcc = matthews_corrcoef(
        labels_list,
        preds_list
    )

    return (
        total_loss / len(loader),
        100.0 * correct / total,
        mcc
    )

best_mcc = -1.0

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "val_mcc": []
}

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    train_loss, train_acc = train_epoch(
        model,
        train_loader,
        optimizer,
        loss_fn,
        DEVICE
    )

    val_loss, val_acc, val_mcc = validate_epoch(
        model,
        val_loader,
        loss_fn,
        DEVICE
    )

    print(
        f"Train Loss: {train_loss:.4f}, "
        f"Train Acc: {train_acc:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f}, "
        f"Val Acc: {val_acc:.2f}%, "
        f"Val MCC: {val_mcc:.4f}"
    )

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_mcc"].append(val_mcc)

    if val_mcc > best_mcc:
        best_mcc = val_mcc
        torch.save(
            model.state_dict(),
            "best_efficientnet_vit_model.pth"
        )
        print(f"Best model saved with MCC: {val_mcc:.4f}")

model.load_state_dict(
    torch.load(
        "best_efficientnet_vit_model.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

test_loss, test_acc, test_mcc = validate_epoch(
    model,
    test_loader,
    loss_fn,
    DEVICE
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")
print(f"Test MCC: {test_mcc:.4f}")

def compute_metrics(model, dataloader, device, class_names):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(
            dataloader,
            desc="Testing"
        ):
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    mcc = matthews_corrcoef(
        all_labels,
        all_preds
    )

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names
    )

    return mcc, cm, report

class_names = dataset.classes

test_mcc, confusion_mat, classification_rep = compute_metrics(
    model,
    test_loader,
    DEVICE,
    class_names
)

print(f"\nMatthews Correlation Coefficient: {test_mcc:.4f}")
print("\nConfusion Matrix:")
print(confusion_mat)
print("\nClassification Report:")
print(classification_rep)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(
    history["train_loss"],
    label="Train Loss"
)
plt.plot(
    history["val_loss"],
    label="Validation Loss"
)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(
    history["train_acc"],
    label="Train Accuracy"
)
plt.plot(
    history["val_acc"],
    label="Validation Accuracy"
)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Epochs")
plt.legend()

plt.tight_layout()
plt.show()

def preprocess_image(image_path, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict_image(image_path, model, class_names, device):
    model.eval()

    image_tensor = preprocess_image(
        image_path
    ).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(
            outputs,
            dim=1
        )

        pred_idx = torch.argmax(
            probs,
            dim=1
        ).item()

        pred_class = class_names[pred_idx]
        pred_prob = probs[0][pred_idx].item()

    return pred_class, pred_prob

image_path = "/content/kvasir-dataset-v2/dyed-resection-margins/00adb051-3e76-4482-b0e5-5207a028470b.jpg"

predicted_class, confidence = predict_image(
    image_path,
    model,
    class_names,
    DEVICE
)

print(
    f"Prediction: {predicted_class} "
    f"(Confidence: {confidence:.2f})"
)

image = Image.open(image_path)

plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.title(
    f"Predicted: {predicted_class} "
    f"({confidence:.2f})"
)
plt.axis("off")
plt.show()