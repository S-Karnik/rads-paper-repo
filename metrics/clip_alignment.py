import torch
import torchvision.transforms.functional as TF
from transformers import CLIPModel
import numpy as np
from typing import List

import test_utils

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import CLIPModel, CLIPTokenizer
import numpy as np

class CLIPAlignment:
    def __init__(self, memorized_images: torch.Tensor = None, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        model_id = "openai/clip-vit-large-patch14"
        self.clip_model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.clip_model.eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id)

        if memorized_images is not None:
            self.set_targets(memorized_images)
        else:
            self.memorized_embeddings = None

    def set_targets(self, memorized_images: torch.Tensor):
        memorized_images = self._ensure_bchw(memorized_images).to(self.device)
        self.memorized_embeddings = self._compute_embeddings(memorized_images)  # [M, D]

    @staticmethod
    def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:      # [C, H, W]
            return x.unsqueeze(0)
        if x.dim() == 4:      # [B, C, H, W]
            return x
        raise ValueError(f"Expected image tensor of shape [C,H,W] or [B,C,H,W], got {tuple(x.shape)}")

    def _compute_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        images = self._ensure_bchw(images)

        if images.dtype != torch.float32 and images.dtype != torch.float16 and images.dtype != torch.bfloat16:
            images = images.float()

        if images.max() > 1.0:
            images = images.float() / 255.0

        images = images.to(self.device)

        images = TF.resize(images, 224, interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        images = TF.center_crop(images, (224, 224))

        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        images = TF.normalize(images, mean=mean, std=std)

        with torch.inference_mode():
            feats = self.clip_model.get_image_features(pixel_values=images)  # [B, D]

        feats = F.normalize(feats, p=2, dim=-1)  # unit length
        return feats

    def compute_text_embeddings(self, prompts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(prompts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_alignment(self, images: torch.Tensor) -> np.ndarray:
        q = self._compute_embeddings(images)                  # [B, D]
        r = self.memorized_embeddings                         # [1, D]
        sim = (q * r).sum(dim=-1)                             # [B] cosine sim
        return sim.detach().cpu().numpy()

    def compute_text_alignment(self, image: torch.Tensor, prompt: List[str]) -> torch.Tensor:
        # 1. Encode Image
        # image: [B, C, H, W] -> [B, D]
        image_features = self._compute_embeddings(image)
        
        # 2. Encode Text correctly using the internal model
        # prompt: List[str] of length B
        text_features = self.compute_text_embeddings(prompt)
        
        # 3. Compute Similarity (Diagonal)
        # image_features: [B, D]
        # text_features: [B, D]
        # We want the similarity between image[i] and prompt[i]
        similarity = (image_features * text_features).sum(dim=-1)
        return similarity

def test_image_should_align() -> None:
    # Precondition.
    img_tensor = test_utils.load_test_image() # Astronaut
    
    # Use the same image as memorized -> should have perfect alignment (approx 1.0)
    memorized = img_tensor.clone()
    
    # Under test.
    aligner = CLIPAlignment(memorized)
    score = aligner.compute_alignment(img_tensor)
    
    # Postcondition.
    assert score[0] > 0.9, f"Expected high alignment score for same image, got {score[0]}"
    print("test_image_should_align passed!")

def test_image_should_not_align() -> None:
    # Precondition.
    img_tensor = test_utils.load_test_image() # Astronaut
    
    # Create a dummy 'memorized' image that is just black/noise
    # Ideally we'd use a real different image, but noise is sufficient for 'not align' check
    memorized = torch.rand_like(img_tensor)
    
    # Under test.
    aligner = CLIPAlignment(memorized)
    score = aligner.compute_alignment(img_tensor)
    
    # Postcondition.
    assert score[0] < 0.5, f"Expected low alignment score for random noise, got {score[0]}"
    print("test_image_should_not_align passed!")

if __name__ == "__main__":
    test_image_should_align()
    test_image_should_not_align()
