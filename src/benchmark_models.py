import time
import torch
from offline_manager import OfflineManager


def benchmark_model(model, input_data, n=10):
    start = time.time()
    with torch.no_grad():
        for _ in range(n):
            model(input_data)
    end = time.time()
    return (end - start) / n


if __name__ == "__main__":
    manager = OfflineManager()

    # ==== CV ====
    dummy_image = torch.randn(1, 3, 224, 224)

    print("\n=== Benchmark ResNet50 ===")
    resnet = manager.load_resnet50()
    print("Time:", benchmark_model(resnet, dummy_image))

    print("\n=== Benchmark EfficientNet ===")
    eff = manager.load_efficientnet()
    print("Time:", benchmark_model(eff, dummy_image))

    # ==== NLP ====
    print("\n=== Benchmark Camembert ===")
    tok, cam = manager.load_camembert()

    dummy_text = tok("Bonjour, ceci est un test.", return_tensors="pt")
    start = time.time()
    cam(**dummy_text)
    end = time.time()
    print("Time:", end - start)
