#include "common.h"

extern std::unordered_map<std::string, torch::ScalarType> type_mapping;
extern std::unordered_map<std::string, torch::DeviceType> device_mapping;
extern std::unordered_map<std::string, torch::Layout> layout_mapping;
extern std::unordered_map<std::string, torch::MemoryFormat> memory_fmt_mapping;

torch::TensorOptions get_tensor_options(rust::String s_dtype,
                                        rust::String s_layout, Device ddevice,
                                        bool requires_grad, bool pin_memory,
                                        rust::String s_mem_fmt);
torch::Scalar get_scalar_type(Scalar scalar);
torch::detail::TensorDataContainer get_scalar_list(rust::Vec<Scalar> list);
