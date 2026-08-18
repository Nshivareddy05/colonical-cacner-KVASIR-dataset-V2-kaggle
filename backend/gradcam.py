import numpy as np
import base64
from io import BytesIO
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import EfficientNetBranch

class GradCAMGenerator:
    def __init__(self, hybrid_model, device):
        self.device = device
        self.cam_model = EfficientNetBranch(hybrid_model).to(self.device)
        self.cam_model.eval()
        
        target_layer = self.cam_model.efficientnet.conv_head
        self.cam = GradCAM(
            model=self.cam_model,
            target_layers=[target_layer]
        )

    def generate(self, input_tensor, original_image):
        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=None
        )[0]
        
        # Ensure original image is float32 in [0, 1] range for overlay
        if isinstance(original_image, Image.Image):
            original = np.array(original_image.resize((224, 224))).astype(np.float32) / 255.0
        else:
            original = original_image
            
        visualization = show_cam_on_image(
            original,
            grayscale_cam,
            use_rgb=True
        )
        
        # Convert grayscale heatmap to RGB colored heatmap for standalone display
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg') # non-interactive backend
        colormap = plt.get_cmap('jet')
        heatmap_colored = colormap(grayscale_cam)[:, :, :3]
        heatmap_colored = np.uint8(255 * heatmap_colored)
        
        return {
            "heatmap": self._numpy_to_base64(heatmap_colored),
            "overlay": self._numpy_to_base64(visualization)
        }
        
    def _numpy_to_base64(self, image_array):
        # image_array is assumed to be in [0, 255] uint8 if it's an overlay or colored heatmap
        if image_array.dtype != np.uint8:
            image_array = np.uint8(255 * image_array)
        img = Image.fromarray(image_array)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
