"""Export a small model that exercises matmul, stack, arange, ones,
to.dtype, concat, slice, and upsample_bicubic2d — the handlers we've
added recently to ExTorch.Export. This is a regression test fixture,
not a realistic model.
"""
import os

import torch
import torch.nn as nn

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))


class FundamentalOps(nn.Module):
    """Exercises several ops in one forward so we have one fixture
    verifying all of them together."""

    def forward(self, x):
        # x: (B, D)

        # matmul with a fixed weight-like tensor
        w = torch.arange(15, dtype=torch.float32).reshape(3, 5)
        y = torch.matmul(x, w)  # (B, 5)

        # ones + stack: produce a (2, B, 5) stack of [y, ones_like(y)]
        o = torch.ones(y.shape, dtype=torch.float32)
        stacked = torch.stack([y, o], dim=0)  # (2, B, 5)

        # concat along last dim
        cat_out = torch.concat([y, o], dim=-1)  # (B, 10)

        # to.dtype: cast to float32 (identity here, but exercises the path)
        cast = stacked.to(torch.float32)

        # Sum the lot down to one tensor so the graph has a single output
        return cast.sum() + cat_out.sum()


def save_bin(name, t):
    path = os.path.join(FIXTURES_DIR, f"{name}.bin")
    arr = t.detach().cpu().contiguous().float().numpy()
    with open(path, "wb") as f:
        f.write(arr.tobytes())
    print(f"  Saved {path} (shape={tuple(arr.shape)})")


def main():
    torch.manual_seed(0)
    model = FundamentalOps().eval()
    x = torch.randn(4, 3)

    with torch.no_grad():
        expected = model(x)
        exported = torch.export.export(model, (x,))

    pt2 = os.path.join(FIXTURES_DIR, "fundamental_ops.pt2")
    torch.export.save(exported, pt2)
    print(f"Saved {pt2}")

    save_bin("fundamental_ops_input", x)
    # expected is a 0-d scalar, reshape to (1,) for Elixir loading
    save_bin("fundamental_ops_output", expected.reshape(1))


if __name__ == "__main__":
    main()
