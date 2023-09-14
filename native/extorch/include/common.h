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
struct PrintOptions;
struct SortResult;
struct OptionalInt;
struct TensorOut;
using CrossTensor = torch::Tensor;
