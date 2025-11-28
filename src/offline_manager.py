import os
import torch
import subprocess
from torchvision import models, transforms
from transformers import CamembertTokenizer, CamembertModel
from PIL import Image


class OfflineManager:
    def __init__(self, base_path="models"):
        self.base_path = base_path

    # =============================
    #         COMPUTER VISION
    # =============================
    def load_resnet50(self):
        path = os.path.join(self.base_path, "cv/resnet50/model.pt")
        model = models.resnet50()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model

    def load_efficientnet(self):
        path = os.path.join(self.base_path, "cv/efficientnet/model.pt")
        model = models.efficientnet_b0()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model

    # =============================
    #             NLP
    # =============================
    def load_camembert(self):
        model_path = os.path.join(self.base_path, "nlp/camembert")
        tokenizer = CamembertTokenizer.from_pretrained(model_path)
        model = CamembertModel.from_pretrained(model_path)
        model.eval()
        return tokenizer, model

    # =============================
    #             OCR
    # =============================
    def check_tesseract(self):
        try:
            subprocess.run(["tesseract", "--version"], stdout=subprocess.DEVNULL)
            return True
        except FileNotFoundError:
            return False

    # =============================
    #     IMAGE INFERENCE DEMO
    # =============================
    def predict_image(self, image_path, model):
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        img = Image.open(image_path).convert("RGB")
        x = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            output = model(x)
        return output


# --------- DEMO ----------
if __name__ == "__main__":
    m = OfflineManager()

    print("Chargement ResNet50…")
    resnet = m.load_resnet50()
    print("OK ")

    print("Chargement CamemBERT…")
    tok, camembert = m.load_camembert()
    print("OK ")

    print("Tesseract présent :", m.check_tesseract())
