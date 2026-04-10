"""Export popular models via torch.export.save for ExTorch testing."""

import torch
import torch.nn as nn
import os
import traceback

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))


def save_tensor_bin(name, tensor):
    """Save a tensor as raw contiguous float32 bytes (native endian)."""
    path = os.path.join(FIXTURES_DIR, f"{name}.bin")
    arr = tensor.detach().cpu().contiguous().float().numpy()
    with open(path, "wb") as f:
        f.write(arr.tobytes())
    print(f"    ref: {name}.bin (shape={tuple(arr.shape)})")


def export_model(name, model, example_input, strict=True):
    """Export a model, save as .pt2, and dump reference (input, output) bytes."""
    path = os.path.join(FIXTURES_DIR, f"{name}.pt2")
    try:
        model.eval()
        with torch.no_grad():
            exported = torch.export.export(model, (example_input,), strict=strict)
        torch.export.save(exported, path)

        # Verify by reloading and running. The reloaded module computes the
        # same output as the in-memory model since the weights are baked into
        # the exported program. We save the reloaded output as the reference
        # so the test verifies what the .pt2 archive itself produces.
        loaded = torch.export.load(path)
        with torch.no_grad():
            output = loaded.module()(example_input)
        if isinstance(output, torch.Tensor):
            shape = output.shape
            save_tensor_bin(f"{name}_input", example_input)
            save_tensor_bin(f"{name}_output", output)
        else:
            shape = type(output).__name__
            print(f"    skip ref: non-tensor output ({shape})")
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

    # ResNet50 (bottleneck blocks instead of basic blocks)
    from torchvision.models import resnet50
    export_model("resnet50", resnet50(weights=None), torch.randn(1, 3, 224, 224))

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

    # Mini-BERT: real nn.TransformerEncoder with 2 layers, exercises
    # scaled_dot_product_attention, layer norm, feed-forward, residuals.
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

    export_model("mini_bert", MiniBERT(), torch.randn(1, 16, 32))

    # Convolutional autoencoder with ConvTranspose2d (segmentation/GAN-style decoder)
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

    export_model("conv_autoencoder", ConvAutoencoder(), torch.randn(1, 3, 32, 32))

    print("\nDone.")
