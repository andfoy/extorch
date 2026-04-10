#pragma once
#include "common.h"

struct IValueNode;
struct IValueFlat;

/// Flatten a c10::IValue into a pre-order IValueFlat node list.
///
/// Tags: 0=tensor, 1=int, 2=float, 3=bool, 4=string, 5=none,
///       6=tuple, 7=list, 8=dict, 9=device
///
/// This is the canonical serialization used for crossing the
/// CXX bridge between Rust NIFs and C++ libtorch code.
IValueFlat flatten_ivalue(const c10::IValue &ivalue);
