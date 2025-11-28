# import os
# import torch
# import subprocess

# import torchvision.models as models
# from transformers import CamembertModel, CamembertTokenizer


# def create_dirs():
#     paths = [
#         "models/cv/resnet50",
#         "models/cv/efficientnet",
#         "models/nlp/camembert",
#         "models/ocr"
#     ]
#     for p in paths:
#         os.makedirs(p, exist_ok=True)


# def download_cv_models():
#     print("Téléchargement ResNet50...")
#     resnet = models.resnet50(weights="IMAGENET1K_V2")
#     torch.save(resnet.state_dict(), "models/cv/resnet50/model.pt")

#     print("Téléchargement EfficientNet...")
#     eff = models.efficientnet_b0(weights="IMAGENET1K_V1")
#     torch.save(eff.state_dict(), "models/cv/efficientnet/model.pt")


# def download_nlp_models():
#     print("Téléchargement CamemBERT (modèle + tokenizer)...")
#     model = CamembertModel.from_pretrained("camembert-base")
#     tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

#     model.save_pretrained("models/nlp/camembert")
#     tokenizer.save_pretrained("models/nlp/camembert")


# def check_tesseract():
#     print("Vérification de Tesseract OCR...")
#     try:
#         subprocess.run(["tesseract", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         print(" Tesseract détecté.")
#     except FileNotFoundError:
#         print(" Tesseract n'est pas installé.")
#         print("Intallez-le ici : https://github.com/tesseract-ocr/tesseract")
#         print("Puis installez la langue FR : tesseract-ocr-fra")


# if __name__ == "__main__":
#     print("=== Initialisation Offline des Modèles ===")
#     create_dirs()
#     download_cv_models()
#     download_nlp_models()
#     check_tesseract()
#     print("=== Terminé ! Tous les modèles sont téléchargés offline. ===")



import os
from xml.parsers.expat import model
import torch
import subprocess

import torchvision.models as models
from transformers import CamembertModel, CamembertTokenizer

def create_dirs():
    paths = [
    "models/cv/resnet50",
    "models/cv/efficientnet",
    "models/nlp/camembert",
    "models/ocr"
    ]
    for p in paths:
        os.makedirs(p, exist_ok=True)
        print(" Dossiers models/ créés")

def download_cv_models():
    print(" Téléchargement ResNet50...")
    resnet = models.resnet50(weights="IMAGENET1K_V2")
    torch.save(resnet.state_dict(), "models/cv/resnet50/model.pt")

    print(" Téléchargement EfficientNet B0...")
    eff = models.efficientnet_b0(weights="IMAGENET1K_V1")
    torch.save(eff.state_dict(), "models/cv/efficientnet/model.pt")

    print(" Modèles CV téléchargés")

def download_nlp_models():
    print("⬇ Téléchargement CamemBERT (modèle + tokenizer)...")
    model = CamembertModel.from_pretrained("camembert-base")
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    model.save_pretrained("models/nlp/camembert")
    tokenizer.save_pretrained("models/nlp/camembert")

    print(" Modèles NLP téléchargés")

def check_tesseract():
    print(" Vérification de Tesseract OCR...")
    try:
        subprocess.run(["tesseract", "--version"], stdout=subprocess.DEVNULL)
        print(" Tesseract détecté")
    except FileNotFoundError:
        print("  Tesseract non installé")
        print("Installez-le ici : [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)")
        print("Et ajoutez la langue FR : tesseract-ocr-fra")

    if __name__ == "__main__":
        print("\n=== Initialisation OFFLINE ===")
        create_dirs()
        download_cv_models()
        download_nlp_models()
        check_tesseract()
        print("\n===  Tous les modèles sont prêts en mode OFFLINE ===")