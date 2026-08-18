import os
import torch
from torchvision import transforms
from PIL import Image
from model import HybridEfficientNetViT
from gradcam import GradCAMGenerator

CLASS_NAMES = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis"
]

class InferenceService:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        
        self.model = HybridEfficientNetViT(len(CLASS_NAMES)).to(self.device)
        self._load_model(model_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])
        
        self.gradcam_generator = GradCAMGenerator(self.model, self.device)

    def _load_model(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model not found at {path}")
            
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True
        )
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def predict(self, image_file):
        image = Image.open(image_file).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            # Get all 8 probabilities so we can filter out dyed classes
            top_probabilities, top_indices = torch.topk(probabilities, k=8, dim=1)
            
        # Filter out dyed classes
        valid_predictions = []
        for i in range(8):
            cls = CLASS_NAMES[top_indices[0, i].item()]
            prob = top_probabilities[0, i].item()
            if not cls.startswith("dyed"):
                valid_predictions.append({"class": cls, "probability": float(prob)})
                
        predicted_class = valid_predictions[0]["class"]
        confidence = valid_predictions[0]["probability"]
        
        top_predictions = valid_predictions[:3]
            
        gradcam_results = self.gradcam_generator.generate(input_tensor, image)
        
        return {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "top_predictions": top_predictions,
            "device": "CUDA" if torch.cuda.is_available() else "CPU",
            "device_name": self.device_name,
            "gradcam": gradcam_results
        }
