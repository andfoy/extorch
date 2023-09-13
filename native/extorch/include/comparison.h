
#include "common.h"
#include "utils.h"

bool allclose(
        const std::shared_ptr<CrossTensor> &input,
        const std::shared_ptr<CrossTensor> &other,
        double rtol, double atol, bool equal_nan);

std::shared_ptr<CrossTensor> argsort(
        const std::shared_ptr<CrossTensor> &input,
        int64_t dim,
        bool descending,
        bool stable);

SortResult sort(
        const std::shared_ptr<CrossTensor> &input,
        int64_t dim,
        bool descending,
        bool stable,
        SortResult out_r);
