# Real-World Model Deployment Tests

End-to-end deployment examples with production models.

## Models

| Model | Type | Params | Task |
|---|---|---:|---|
| CLIP ViT-B/32 | Vision encoder | 87M | Image search, zero-shot classification |
| DistilBERT | NLP | 66M | Text classification, sentiment analysis |
| MobileNetV3-Small | Classification | 2.5M | Edge/mobile inference |
| EfficientNet-B0 | Classification | 5.3M | Balanced accuracy/speed |
| ResNet50 | Classification | 25M | Production baseline |
| ViT-B/16 | Classification | 86M | Vision transformer |
| DeepLabV3-MobileNet | Segmentation | 11M | Semantic segmentation |
| Whisper-tiny encoder | Speech | 9M | Audio feature extraction |

## Setup

```bash
# Install Python dependencies
.venv/bin/pip install torch torchvision transformers clip

# Export all models
.venv/bin/python examples/models/export_models.py --device cuda --aoti

# Run benchmark (all models, all inference paths)
mix run examples/models/benchmark.exs

# Run full image classification pipeline
mix run examples/models/image_classification_pipeline.exs [optional_image.jpg]
```

## Benchmark Output

The benchmark reports median latency (ms) for each model across three paths:

```
model                    type                interp     native       aoti
--------------------------------------------------------------------------------
clip_visual              multimodal            X.XX       X.XX       X.XX
distilbert               nlp                   X.XX       X.XX       X.XX
mobilenet_v3_small       classification        X.XX       X.XX       X.XX
efficientnet_b0          classification        X.XX       X.XX       X.XX
resnet50_prod            classification        X.XX       X.XX       X.XX
vit_b_16_prod            transformer           X.XX       X.XX       X.XX
deeplabv3_mobilenet      segmentation          X.XX       X.XX       X.XX
whisper_tiny_encoder     speech                X.XX       X.XX       X.XX
```

## What This Proves

- **Train in Python, serve from the BEAM** — every model is exported from standard
  PyTorch/torchvision/transformers and runs on ExTorch without modification
- **Three inference paths** — interpreter (debug), native (production), AOTI (fastest)
- **Full pipeline** — preprocessing, inference, postprocessing in Elixir
- **GPU acceleration** — all models run on CUDA with proper device management
