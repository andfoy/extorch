"""Generate AOTI-compiled .pt2 packages for popular models.

Usage:  .venv/bin/python test/fixtures/generate_aoti_popular_models.py [--device cuda]

Compiles each model via torch._inductor.aoti_compile_and_package() for
benchmarking ExTorch.AOTI vs ExTorch.Export vs PyTorch.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch._inductor import aoti_compile_and_package

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))


def compile_model(name, model, example_input, device="cpu"):
    """Export and AOTI-compile a model, saving as {name}_aoti.pt2."""
    path = os.path.join(FIXTURES_DIR, f"{name}_aoti.pt2")
    try:
        model.eval()
        if device != "cpu":
            model = model.to(device)
            example_input = example_input.to(device)

        with torch.no_grad():
            exported = torch.export.export(model, (example_input,))

        aoti_compile_and_package(exported, package_path=path)

        # Verify
        loaded = torch._inductor.aoti_load_package(path)
        with torch.no_grad():
            output = loaded(example_input)
        print(f"  {name}: compiled (output shape: {output.shape})")
        return True
    except Exception as e:
        print(f"  {name}: FAILED - {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()
    device = args.device

    torch.manual_seed(42)

    print(f"=== AOTI compilation (device={device}) ===\n")

    # Production vision models
    from torchvision.models import (
        alexnet, vgg11, squeezenet1_0, mobilenet_v2,
        resnet18, resnet50, vit_b_16, ViT_B_16_Weights,
    )

    models = [
        ("alexnet",    alexnet(weights=None),             torch.randn(1, 3, 224, 224)),
        ("vgg11",      vgg11(weights=None),               torch.randn(1, 3, 224, 224)),
        ("squeezenet", squeezenet1_0(weights=None),        torch.randn(1, 3, 224, 224)),
        ("mobilenetv2", mobilenet_v2(weights=None),        torch.randn(1, 3, 224, 224)),
        ("resnet18",   resnet18(weights=None),             torch.randn(1, 3, 224, 224)),
        ("resnet50",   resnet50(weights=None),             torch.randn(1, 3, 224, 224)),
        ("vit_b_16",   vit_b_16(weights=ViT_B_16_Weights.DEFAULT), torch.randn(1, 3, 224, 224)),
    ]

    # Synthetic models
    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(32, 64)
            self.norm = nn.LayerNorm(64)
            self.attn = nn.MultiheadAttention(64, 4, batch_first=True)
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            x = self.embed(x)
            x = self.norm(x)
            attn_out, _ = self.attn(x, x, x)
            x = x + attn_out
            x = x.mean(dim=1)
            return self.fc(x)

    class MiniBERT(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(32, 64)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=128, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.head = nn.Linear(64, 10)

        def forward(self, x):
            x = self.embed(x)
            x = self.encoder(x)
            x = x.mean(dim=1)
            return self.head(x)

    class SimpleAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 32))
            self.decoder = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())

        def forward(self, x):
            return self.decoder(self.encoder(x))

    class ConvAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
            self.enc2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
            self.dec1 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
            self.dec2 = nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1)

        def forward(self, x):
            x = torch.relu(self.enc1(x))
            x = torch.relu(self.enc2(x))
            x = torch.relu(self.dec1(x))
            return torch.sigmoid(self.dec2(x))

    models += [
        ("simple_transformer", SimpleTransformer(), torch.randn(1, 16, 32)),
        ("mini_bert",          MiniBERT(),          torch.randn(1, 16, 32)),
        ("autoencoder",        SimpleAutoencoder(),  torch.randn(1, 784)),
        ("conv_autoencoder",   ConvAutoencoder(),    torch.randn(1, 3, 32, 32)),
    ]

    for name, model, example in models:
        compile_model(name, model, example, device)

    print("\nDone.")
