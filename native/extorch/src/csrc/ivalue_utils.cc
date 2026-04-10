#include "extorch/src/native.rs.h"
#include "extorch/include/ivalue_utils.h"

#include <sstream>

// Tags: 0=tensor, 1=int, 2=float, 3=bool, 4=string, 5=none,
//       6=tuple, 7=list, 8=dict, 9=device

static void flatten_ivalue_recursive(
    const c10::IValue &ivalue,
    rust::Vec<IValueNode> &nodes,
    int64_t parent_idx)
{
    IValueNode node;
    node.parent_idx = parent_idx;
    node.child_count = 0;
    node.int_val = 0;
    node.float_val = 0.0;
    node.bool_val = false;
    node.string_val = rust::String("");
    node.tensor = std::shared_ptr<CrossTensor>(nullptr);

    if (ivalue.isTensor()) {
        node.tag = 0;
        node.tensor = std::make_shared<CrossTensor>(ivalue.toTensor());
        nodes.push_back(std::move(node));
    } else if (ivalue.isInt()) {
        node.tag = 1;
        node.int_val = ivalue.toInt();
        nodes.push_back(std::move(node));
    } else if (ivalue.isDouble()) {
        node.tag = 2;
        node.float_val = ivalue.toDouble();
        nodes.push_back(std::move(node));
    } else if (ivalue.isBool()) {
        node.tag = 3;
        node.bool_val = ivalue.toBool();
        nodes.push_back(std::move(node));
    } else if (ivalue.isString()) {
        node.tag = 4;
        node.string_val = rust::String(ivalue.toStringRef());
        nodes.push_back(std::move(node));
    } else if (ivalue.isNone()) {
        node.tag = 5;
        nodes.push_back(std::move(node));
    } else if (ivalue.isDevice()) {
        node.tag = 9;
        auto dev = ivalue.toDevice();
        node.string_val = rust::String(c10::DeviceTypeName(dev.type()));
        node.int_val = dev.has_index() ? dev.index() : -1;
        nodes.push_back(std::move(node));
    } else if (ivalue.isTuple()) {
        node.tag = 6;
        auto tuple = ivalue.toTuple();
        node.child_count = static_cast<int64_t>(tuple->elements().size());
        int64_t my_idx = static_cast<int64_t>(nodes.size());
        nodes.push_back(std::move(node));
        for (const auto &elem : tuple->elements()) {
            flatten_ivalue_recursive(elem, nodes, my_idx);
        }
    } else if (ivalue.isList()) {
        node.tag = 7;
        auto list = ivalue.toList();
        node.child_count = static_cast<int64_t>(list.size());
        int64_t my_idx = static_cast<int64_t>(nodes.size());
        nodes.push_back(std::move(node));
        for (const auto &elem : list) {
            flatten_ivalue_recursive(elem, nodes, my_idx);
        }
    } else if (ivalue.isGenericDict()) {
        node.tag = 8;
        auto dict = ivalue.toGenericDict();
        node.child_count = static_cast<int64_t>(dict.size()) * 2;
        int64_t my_idx = static_cast<int64_t>(nodes.size());
        nodes.push_back(std::move(node));
        for (const auto &entry : dict) {
            flatten_ivalue_recursive(entry.key(), nodes, my_idx);
            flatten_ivalue_recursive(entry.value(), nodes, my_idx);
        }
    } else {
        // Fallback: represent as string
        node.tag = 4;
        std::ostringstream oss;
        oss << ivalue;
        node.string_val = rust::String(oss.str());
        nodes.push_back(std::move(node));
    }
}

IValueFlat flatten_ivalue(const c10::IValue &ivalue) {
    IValueFlat result;
    flatten_ivalue_recursive(ivalue, result.nodes, -1);
    return result;
}
