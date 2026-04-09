"""Export popular models via torch.export.save for ExTorch testing."""

import torch
import torch.nn as nn
import os
import traceback

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))


def export_model(name, model, example_input, strict=True):
    """Export a model and save as .pt2."""
    path = os.path.join(FIXTURES_DIR, f"{name}.pt2")
    try:
        model.eval()
        with torch.no_grad():
            exported = torch.export.export(model, (example_input,), strict=strict)
        torch.export.save(exported, path)

        # Verify
        loaded = torch.export.load(path)
        output = loaded.module()(example_input)
        if isinstance(output, torch.Tensor):
            shape = output.shape
        else:
            shape = type(output).__name__
        print(f"  {name}: saved ({shape})")

        # Show ops used
        ops = set()
        for node in exported.graph.nodes:
            if node.op == 'call_function':
                ops.add(str(node.target).split('.')[-1])
        print(f"    ops: {sorted(ops)}")
        return True
    except Exception as e:
        print(f"  {name}: FAILED - {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    torch.manual_seed(42)

    print("=== Classical CNNs ===")

    # AlexNet
    from torchvision.models import alexnet
    export_model("alexnet", alexnet(weights=None), torch.randn(1, 3, 224, 224))

    # SqueezeNet
    from torchvision.models import squeezenet1_0
    export_model("squeezenet", squeezenet1_0(weights=None), torch.randn(1, 3, 224, 224))

    # MobileNetV2
    from torchvision.models import mobilenet_v2
    export_model("mobilenetv2", mobilenet_v2(weights=None), torch.randn(1, 3, 224, 224))

    # ResNet18
    from torchvision.models import resnet18
    export_model("resnet18", resnet18(weights=None), torch.randn(1, 3, 224, 224))

    # VGG11 (smaller than VGG16)
    from torchvision.models import vgg11
    export_model("vgg11", vgg11(weights=None), torch.randn(1, 3, 224, 224))

    print("\n=== Simple custom models ===")

    # Simple transformer encoder
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

    export_model("simple_transformer", SimpleTransformer(), torch.randn(1, 16, 32))

    # Simple LSTM model
    class SimpleLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(32, 64, batch_first=True)
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(h.squeeze(0))

    export_model("simple_lstm", SimpleLSTM(), torch.randn(1, 16, 32), strict=False)

    # Simple autoencoder
    class SimpleAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 32))
            self.decoder = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    export_model("autoencoder", SimpleAutoencoder(), torch.randn(1, 784))

    print("\nDone.")
