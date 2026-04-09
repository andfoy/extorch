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
struct TensorTuple;
struct TensorList;
struct TensorOrInt;
struct IntListOrInt;
struct IValueNode;
struct IValueFlat;
struct NamedTensor;
using CrossTensor = torch::Tensor;
struct CrossModuleImpl;
using CrossModule = CrossModuleImpl;
struct CrossNNModuleImpl;
using CrossNNModule = CrossNNModuleImpl;
struct CrossAOTILoaderImpl;
using CrossAOTILoader = CrossAOTILoaderImpl;
