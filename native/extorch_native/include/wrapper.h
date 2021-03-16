#pragma once
#include "common.h"
#include "creation.h"

rust::Slice<const int64_t> size(const std::shared_ptr<CrossTensor> &tensor);
rust::String dtype(const std::shared_ptr<CrossTensor> &tensor);
Device device(const std::shared_ptr<CrossTensor> &tensor);
rust::String repr(const std::shared_ptr<CrossTensor> &tensor);
std::shared_ptr<CrossTensor> unsqueeze(
    const std::shared_ptr<CrossTensor> &tensor,
    int64_t dim);
