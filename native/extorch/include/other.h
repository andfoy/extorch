
#include "common.h"
#include "utils.h"

std::shared_ptr<CrossTensor> view_as_complex(
    const std::shared_ptr<CrossTensor> &input);

std::shared_ptr<CrossTensor> resolve_conj(
    const std::shared_ptr<CrossTensor> &input);
