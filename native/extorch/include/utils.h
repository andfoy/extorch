#include "common.h"

extern std::unordered_map<std::string, torch::ScalarType> type_mapping;
extern std::unordered_map<torch::ScalarType, std::string> inv_type_mapping;
extern std::unordered_map<std::string, torch::DeviceType> device_mapping;
extern std::unordered_map<std::string, torch::Layout> layout_mapping;
extern std::unordered_map<torch::Layout, std::string> inv_layout_mapping;
extern std::unordered_map<std::string, torch::MemoryFormat> memory_fmt_mapping;
extern std::unordered_map<torch::MemoryFormat, std::string> inv_memory_fmt_mapping;

torch::TensorOptions get_tensor_options(rust::String s_dtype,
                                        rust::String s_layout, Device ddevice,
                                        bool requires_grad, bool pin_memory,
                                        rust::String s_mem_fmt);
torch::Scalar get_scalar_type(Scalar scalar);
torch::detail::TensorDataContainer get_scalar_list(rust::Vec<Scalar> list);
torch::detail::TensorDataContainer get_complex_tensor_parts(
        rust::Vec<Scalar> list,
        c10::ScalarType scalar_type,
        torch::TensorOptions opts,
        const int64_t *ptr);

std::vector<CrossTensor> unpack_tensor_tuple(TensorTuple tuple, int64_t sz_constraint);
TensorTuple pack_tensor_tuple(std::vector<std::shared_ptr<CrossTensor>> vec);
