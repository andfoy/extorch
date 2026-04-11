"""Export real-world models for ExTorch deployment testing.

Usage:
    .venv/bin/python examples/models/export_models.py [--device cuda] [--aoti]

Exports each model as:
  - {name}.pt2          (torch.export.save — for ExTorch.Export)
  - {name}_aoti.pt2     (AOTI compiled — for ExTorch.AOTI, if --aoti)
  - {name}_input.bin    (reference input as raw float32 bytes)
  - {name}_config.json  (preprocessing config: shapes, normalization, tokenizer)
"""

import argparse
import json
import os
import traceback

import torch
import torch.nn as nn

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)


def save_input(name, tensor):
    path = os.path.join(FIXTURES_DIR, f"{name}_input.bin")
    with open(path, "wb") as f:
        f.write(tensor.detach().cpu().contiguous().float().numpy().tobytes())


def save_config(name, config):
    path = os.path.join(FIXTURES_DIR, f"{name}_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def export_model(name, model, example_input, config, device="cpu", aoti=False):
    path = os.path.join(FIXTURES_DIR, f"{name}.pt2")
    try:
        model.eval()
        if device != "cpu":
            model = model.to(device)
            example_input = tuple(
                x.to(device) if isinstance(x, torch.Tensor) else x for x in example_input
            )

        with torch.no_grad():
            exported = torch.export.export(model, example_input)
        torch.export.save(exported, path)

        # Save reference input and config
        for i, inp in enumerate(example_input):
            if isinstance(inp, torch.Tensor):
                suffix = f"_input{'_' + str(i) if len(example_input) > 1 else ''}"
                save_input(f"{name}{suffix}", inp.cpu())
        save_config(name, config)

        # Verify
        loaded = torch.export.load(path)
        with torch.no_grad():
            output = loaded.module()(*example_input)
        if isinstance(output, torch.Tensor):
            print(f"  {name}: exported (output shape: {output.shape})")
        else:
            print(f"  {name}: exported (output type: {type(output).__name__})")

        # AOTI compilation
        if aoti:
            aoti_path = os.path.join(FIXTURES_DIR, f"{name}_aoti.pt2")
            try:
                from torch._inductor import aoti_compile_and_package
                aoti_compile_and_package(exported, package_path=aoti_path)
                print(f"  {name}: AOTI compiled")
            except Exception as e:
                print(f"  {name}: AOTI failed - {e}")

        return True
    except Exception as e:
        print(f"  {name}: FAILED - {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--aoti", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)

    # =====================================================================
    # 1. CLIP — Vision+Text multimodal (image search, zero-shot classify)
    # =====================================================================
    print("\n=== CLIP ViT-B/32 ===")
    try:
        import clip
        clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")

        # Export just the visual encoder (text encoder has dynamic shapes)
        class CLIPVisualEncoder(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.visual = model.visual.float()

            def forward(self, image):
                return self.visual(image)

        clip_visual = CLIPVisualEncoder(clip_model)
        clip_input = torch.randn(1, 3, 224, 224)
        export_model("clip_visual", clip_visual, (clip_input,), {
            "type": "vision_encoder",
            "input_shape": [1, 3, 224, 224],
            "preprocessing": {
                "resize": 224,
                "center_crop": 224,
                "normalize_mean": [0.48145466, 0.4578275, 0.40821073],
                "normalize_std": [0.26862954, 0.26130258, 0.27577711]
            },
            "output": "image_embedding_512d"
        }, args.device, args.aoti)
    except ImportError:
        print("  CLIP: skipped (pip install git+https://github.com/openai/CLIP)")

    # =====================================================================
    # 2. DistilBERT — Text classification / NLU
    # =====================================================================
    print("\n=== DistilBERT ===")
    try:
        from transformers import DistilBertModel, DistilBertConfig

        config = DistilBertConfig(
            vocab_size=30522, hidden_size=768, num_attention_heads=12,
            num_hidden_layers=6, intermediate_size=3072
        )
        bert = DistilBertModel(config).eval()

        # Fixed sequence length for export
        class DistilBertFixed(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.classifier = nn.Linear(768, 2)  # binary classification

            def forward(self, input_ids, attention_mask):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                cls = outputs.last_hidden_state[:, 0]
                return self.classifier(cls)

        bert_cls = DistilBertFixed(bert)
        seq_len = 128
        input_ids = torch.randint(0, 30522, (1, seq_len))
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)

        export_model("distilbert", bert_cls, (input_ids, attention_mask), {
            "type": "text_classification",
            "tokenizer": "distilbert-base-uncased",
            "max_seq_length": seq_len,
            "input_shapes": {"input_ids": [1, seq_len], "attention_mask": [1, seq_len]},
            "output": "logits_2d"
        }, args.device, args.aoti)
    except ImportError:
        print("  DistilBERT: skipped (pip install transformers)")

    # =====================================================================
    # 3. MobileNetV3 — Lightweight classification (edge/mobile)
    # =====================================================================
    print("\n=== MobileNetV3-Small ===")
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

    mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    mobilenet_input = torch.randn(1, 3, 224, 224)

    export_model("mobilenet_v3_small", mobilenet, (mobilenet_input,), {
        "type": "image_classification",
        "input_shape": [1, 3, 224, 224],
        "num_classes": 1000,
        "preprocessing": {
            "resize": 256,
            "center_crop": 224,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225]
        },
        "output": "logits_1000d",
        "labels": "imagenet1k"
    }, args.device, args.aoti)

    # =====================================================================
    # 4. EfficientNet-B0 — Balanced classification
    # =====================================================================
    print("\n=== EfficientNet-B0 ===")
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    effnet_input = torch.randn(1, 3, 224, 224)

    export_model("efficientnet_b0", effnet, (effnet_input,), {
        "type": "image_classification",
        "input_shape": [1, 3, 224, 224],
        "num_classes": 1000,
        "preprocessing": {
            "resize": 256,
            "center_crop": 224,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225]
        },
        "output": "logits_1000d"
    }, args.device, args.aoti)

    # =====================================================================
    # 5. ResNet50 — Production classification baseline
    # =====================================================================
    print("\n=== ResNet50 (with weights) ===")
    from torchvision.models import resnet50, ResNet50_Weights

    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet_input = torch.randn(1, 3, 224, 224)

    export_model("resnet50_prod", resnet, (resnet_input,), {
        "type": "image_classification",
        "input_shape": [1, 3, 224, 224],
        "num_classes": 1000,
        "preprocessing": {
            "resize": 256,
            "center_crop": 224,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225]
        },
        "output": "logits_1000d"
    }, args.device, args.aoti)

    # =====================================================================
    # 6. ViT-B/16 — Vision Transformer (production)
    # =====================================================================
    print("\n=== ViT-B/16 (with weights) ===")
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    vit_input = torch.randn(1, 3, 224, 224)

    export_model("vit_b_16_prod", vit, (vit_input,), {
        "type": "image_classification",
        "input_shape": [1, 3, 224, 224],
        "num_classes": 1000,
        "preprocessing": {
            "resize": 256,
            "center_crop": 224,
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225]
        },
        "output": "logits_1000d"
    }, args.device, args.aoti)

    # =====================================================================
    # 7. DeepLabV3 — Semantic Segmentation
    # =====================================================================
    print("\n=== DeepLabV3-MobileNetV3 ===")
    try:
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

        deeplab = deeplabv3_mobilenet_v3_large(weights=None, num_classes=21)

        class DeepLabWrapper(nn.Module):
            """Unwrap the OrderedDict output to a single tensor."""
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model(x)["out"]

        deeplab_wrapped = DeepLabWrapper(deeplab)
        deeplab_input = torch.randn(1, 3, 320, 320)

        export_model("deeplabv3_mobilenet", deeplab_wrapped, (deeplab_input,), {
            "type": "semantic_segmentation",
            "input_shape": [1, 3, 320, 320],
            "num_classes": 21,
            "preprocessing": {
                "resize": 320,
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225]
            },
            "output": "pixel_logits_21_classes"
        }, args.device, args.aoti)
    except Exception as e:
        print(f"  DeepLabV3: FAILED - {e}")

    # =====================================================================
    # 8. Whisper-tiny encoder — Speech feature extraction
    # =====================================================================
    print("\n=== Whisper-tiny (encoder only) ===")
    try:
        from transformers import WhisperModel, WhisperConfig

        whisper_config = WhisperConfig(
            d_model=384, encoder_layers=4, decoder_layers=4,
            encoder_attention_heads=6, decoder_attention_heads=6,
            encoder_ffn_dim=1536, decoder_ffn_dim=1536
        )
        whisper = WhisperModel(whisper_config).eval()

        class WhisperEncoderOnly(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.encoder = model.encoder

            def forward(self, input_features):
                return self.encoder(input_features).last_hidden_state

        whisper_enc = WhisperEncoderOnly(whisper)
        # Whisper expects 80-channel mel spectrogram, 3000 frames (30 seconds)
        mel_input = torch.randn(1, 80, 3000)

        export_model("whisper_tiny_encoder", whisper_enc, (mel_input,), {
            "type": "speech_encoder",
            "input_shape": [1, 80, 3000],
            "preprocessing": {
                "sample_rate": 16000,
                "n_mels": 80,
                "chunk_length_s": 30
            },
            "output": "encoder_hidden_states"
        }, args.device, args.aoti)
    except ImportError:
        print("  Whisper: skipped (pip install transformers)")

    print("\nDone. Fixtures saved to:", FIXTURES_DIR)
