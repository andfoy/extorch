#pragma once
#include "common.h"
#include "creation.h"
#include "info.h"

std::shared_ptr<CrossTensor> unsqueeze(
    const std::shared_ptr<CrossTensor> &tensor,
    int64_t dim);
