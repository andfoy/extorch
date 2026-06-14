"""Export Hugging Face YOLOS (transformer-based object detector) and
write reference IO for ExTorch verification.

YOLOS is a ViT-style detection model that emits a fixed-size tensor of
N candidate predictions (no anchor boxes, no NMS inside the model).
This means it exports cleanly via torch.export — unlike anchor-based
detectors (Faster/Mask R-CNN, SSD, RetinaNet, FCOS) which all fail
with GuardOnDataDependentSymNode inside torchvision's batched_nms.

The fixtures here let the Elixir test prove ExTorch can run a real
modern detection model end-to-end with output matching PyTorch.
"""
import os

import torch
from transformers import YolosConfig, YolosForObjectDetection

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.model = m

    def forward(self, pixel_values):
        out = self.model(pixel_values=pixel_values)
        return out.logits, out.pred_boxes


def save_tensor_bin(name, tensor):
    path = os.path.join(FIXTURES_DIR, f"{name}.bin")
    arr = tensor.detach().cpu().contiguous().float().numpy()
    with open(path, "wb") as f:
        f.write(arr.tobytes())
    print(f"  Saved {path} (shape={tuple(arr.shape)})")


def main():
    torch.manual_seed(0)

    # Use the small default config (random weights — fine, we only
    # need numerical equivalence between PyTorch and ExTorch on the
    # same weights, not pretrained accuracy).
    config = YolosConfig()
    model = YolosForObjectDetection(config).eval()
    wrapper = Wrap(model).eval()

    pixel_values = torch.randn(1, 3, 416, 416)

    with torch.no_grad():
        logits_ref, boxes_ref = wrapper(pixel_values)
        exported = torch.export.export(wrapper, (pixel_values,), strict=False)

    pt2_path = os.path.join(FIXTURES_DIR, "yolos.pt2")
    torch.export.save(exported, pt2_path)
    print(f"Saved {pt2_path}")

    save_tensor_bin("yolos_input", pixel_values)
    save_tensor_bin("yolos_logits", logits_ref)
    save_tensor_bin("yolos_boxes", boxes_ref)
    print(f"\nLogits shape: {tuple(logits_ref.shape)}")
    print(f"Boxes shape: {tuple(boxes_ref.shape)}")


if __name__ == "__main__":
    main()
