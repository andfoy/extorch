#pragma once
#include <memory>
#include <iostream>
#include <torch/torch.h>
#include "rust/cxx.h"

struct Device;
struct Scalar;
struct ScalarList;
struct TorchSlice;
struct TorchIndex;
using CrossTensor = torch::Tensor;
