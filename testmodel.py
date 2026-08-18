import os
import glob
import random
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from timm import create_model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

MODEL_PATH = "./best_efficientnet_vit_model.pth"
DATASET_PATH = "./kvasir-dataset-v2"
NUM_IMAGES = 10
IMAGE_SIZE = 224
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("Device: CUDA")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Device: CPU")
    print("GPU: Not available")

class_names = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis"
]

NUM_CLASSES = len(class_names)

class HybridEfficientNetViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.efficientnet = create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=num_classes
        )

        self.vit = create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=num_classes
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * num_classes, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        efficientnet_output = self.efficientnet(x)
        vit_output = self.vit(x)

        combined = torch.cat(
            (efficientnet_output, vit_output),
            dim=1
        )

        return self.fc(combined)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{os.path.abspath(MODEL_PATH)}"
    )

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{os.path.abspath(DATASET_PATH)}"
    )

print("\nLoading model...")

model = HybridEfficientNetViT(NUM_CLASSES).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.eval()

print("Model loaded successfully")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

image_files = []

for class_name in class_names:
    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    files = glob.glob(
        os.path.join(class_path, "*.jpg")
    )

    files += glob.glob(
        os.path.join(class_path, "*.jpeg")
    )

    files += glob.glob(
        os.path.join(class_path, "*.png")
    )

    image_files.extend(files)

if len(image_files) == 0:
    raise FileNotFoundError(
        "No images were found in the dataset."
    )

print("Total images found:", len(image_files))

NUM_IMAGES = min(NUM_IMAGES, len(image_files))

selected_images = random.sample(
    image_files,
    NUM_IMAGES
)

os.makedirs(
    "gradcam_results",
    exist_ok=True
)

class EfficientNetBranch(nn.Module):
    def __init__(self, hybrid_model):
        super().__init__()

        self.efficientnet = hybrid_model.efficientnet
        self.vit = hybrid_model.vit
        self.fc = hybrid_model.fc

    def forward(self, x):
        efficientnet_output = self.efficientnet(x)
        vit_output = self.vit(x)

        combined = torch.cat(
            (efficientnet_output, vit_output),
            dim=1
        )

        return self.fc(combined)

cam_model = EfficientNetBranch(model).to(DEVICE)
cam_model.eval()

target_layer = cam_model.efficientnet.conv_head

cam = GradCAM(
    model=cam_model,
    target_layers=[target_layer]
)

correct = 0
total = 0

print("\nStarting predictions...\n")

for index, image_path in enumerate(selected_images):

    actual_class = os.path.basename(
        os.path.dirname(image_path)
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    input_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    start_time = time.time()

    with torch.no_grad():
        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=3,
            dim=1
        )

    predicted_idx = top_indices[0, 0].item()
    predicted_class = class_names[predicted_idx]
    confidence = top_probabilities[0, 0].item()

    elapsed = time.time() - start_time

    if predicted_class == actual_class:
        correct += 1

    total += 1

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=None
    )[0]

    original = np.array(
        image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )
    ).astype(
        np.float32
    ) / 255.0

    visualization = show_cam_on_image(
        original,
        grayscale_cam,
        use_rgb=True
    )

    print("=" * 70)
    print(f"Image {index + 1}/{NUM_IMAGES}")
    print("=" * 70)
    print("Actual:     ", actual_class)
    print("Predicted:  ", predicted_class)
    print("Confidence: ", f"{confidence:.2%}")
    print("Inference:  ", f"{elapsed:.2f} seconds")
    print("\nTop 3 predictions:")

    for rank in range(3):
        cls = class_names[
            top_indices[0, rank].item()
        ]

        prob = top_probabilities[
            0, rank
        ].item()

        print(
            f"{rank + 1}. "
            f"{cls}: "
            f"{prob:.2%}"
        )

    print()

    result_path = os.path.join(
        "gradcam_results",
        f"prediction_{index + 1}.png"
    )

    plt.figure(
        figsize=(18, 6)
    )

    plt.subplot(1, 3, 1)

    plt.imshow(original)

    plt.title(
        f"Original\n"
        f"Actual: {actual_class}"
    )

    plt.axis("off")

    plt.subplot(1, 3, 2)

    plt.imshow(
        grayscale_cam,
        cmap="jet"
    )

    plt.title(
        "Grad-CAM Heatmap"
    )

    plt.axis("off")

    plt.subplot(1, 3, 3)

    plt.imshow(
        visualization
    )

    plt.title(
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence:.2%}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        result_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

accuracy = 100.0 * correct / total

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Images tested: {total}"
)

print(
    f"Correct predictions: {correct}"
)

print(
    f"Accuracy on sampled images: {accuracy:.2f}%"
)

print(
    "Grad-CAM results saved in:"
)

print(
    os.path.abspath(
        "gradcam_results"
    )
)