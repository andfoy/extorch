
#include "common.h"
#include "utils.h"

bool allclose(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        double rtol, double atol, bool equal_nan);
