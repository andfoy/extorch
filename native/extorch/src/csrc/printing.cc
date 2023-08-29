#include "extorch/src/native.rs.h"
#include "extorch/include/printing.h"


inline torch::Tensor tensor_totype(torch::Tensor tensor) {
    torch::ScalarType dtype = tensor.is_mps() ? torch::kFloat : torch::kDouble;
    return tensor.to(dtype);
}

struct Formatter {
    PrintOptions opts;
    bool floating_dtype;
    bool int_mode;
    bool sci_mode;
    int64_t max_width;

    Formatter(torch::Tensor tensor, const PrintOptions opts) {
        this->opts = opts;
        floating_dtype = tensor.is_floating_point();
        int_mode = true;
        sci_mode = false;
        max_width = 1;

        torch::NoGradGuard guard;
        tensor = tensor.to(torch::kCPU);
        torch::Tensor tensor_view = tensor.reshape(-1);

        if(!floating_dtype) {
            AT_DISPATCH_INTEGRAL_TYPES(tensor_view.scalar_type(), "max_repr_len", [&] {
                auto data_ptr = tensor_view.data_ptr<scalar_t>();
                for(int i = 0; i < tensor_view.numel(); i++) {
                    // char placeholder[64] = {0};
                    std::stringstream ss;
                    ss << data_ptr[i];
                    auto sz = ss.str().size();
                    // snprintf(placeholder, 64, "%lld", data_ptr[i]);
                    // auto sz = strlen(placeholder);
                    max_width = sz > max_width ? sz : max_width;
                }
            });
        } else {
            torch::Tensor nonzero_finite_vals = torch::masked_select(
                tensor_view, torch::isfinite(tensor_view) & tensor_view.ne(0)
            );

            if(nonzero_finite_vals.numel() == 0) {
                return;
            }

            auto nonzero_finite_abs = tensor_totype(nonzero_finite_vals.abs());
            auto nonzero_finite_min = tensor_totype(nonzero_finite_abs.min());
            auto nonzero_finite_max = tensor_totype(nonzero_finite_abs.max());

            for(int i = 0; i < nonzero_finite_vals.numel(); i++) {
                auto value = nonzero_finite_vals[i];
                auto diff = value != torch::ceil(value);
                bool is_diff = *(diff.data_ptr<bool>());
                if(is_diff) {
                    int_mode = false;
                    break;
                }
            }

            if(int_mode) {
                if(*(nonzero_finite_max / nonzero_finite_min > 1000.0).data_ptr<bool>()
                        || *(nonzero_finite_max > 1.0e8).data_ptr<bool>()) {
                    sci_mode = true;
                    std::stringstream fmt_ss;
                    fmt_ss << "%." << opts.precision << "e";
                    auto format = fmt_ss.str();
                    AT_DISPATCH_FLOATING_TYPES(nonzero_finite_vals.scalar_type(), "max_repr_len", [&] {
                        auto data_ptr = nonzero_finite_vals.data_ptr<scalar_t>();
                        for(int i = 0; i < nonzero_finite_vals.numel(); i++) {
                            char placeholder[64] = {0};
                            snprintf(placeholder, 64, format.c_str(), data_ptr[i]);
                            auto sz = strlen(placeholder);
                            max_width = sz > max_width ? sz : max_width;
                        }
                    });
                } else {
                    AT_DISPATCH_FLOATING_TYPES(nonzero_finite_vals.scalar_type(), "max_repr_len", [&] {
                        auto data_ptr = nonzero_finite_vals.data_ptr<scalar_t>();
                        for(int i = 0; i < nonzero_finite_vals.numel(); i++) {
                            char placeholder[64] = {0};
                            snprintf(placeholder, 64, "%.0f", data_ptr[i]);
                            auto sz = strlen(placeholder);
                            max_width = sz > max_width ? sz : max_width;
                        }
                    });
                }
            } else {
                if (*(nonzero_finite_max / nonzero_finite_min > 1000.0).data_ptr<bool>()
                        || *(nonzero_finite_max > 1.0e8).data_ptr<bool>()
                            || *(nonzero_finite_min < 1.0e-4).data_ptr<bool>()) {

                    sci_mode = true;
                    std::stringstream fmt_ss;
                    fmt_ss << "%." << opts.precision << "e";
                    auto format = fmt_ss.str();
                    AT_DISPATCH_FLOATING_TYPES(nonzero_finite_vals.scalar_type(), "max_repr_len", [&] {
                        auto data_ptr = nonzero_finite_vals.data_ptr<scalar_t>();
                        for(int i = 0; i < nonzero_finite_vals.numel(); i++) {
                            char placeholder[64] = {0};
                            snprintf(placeholder, 64, format.c_str(), data_ptr[i]);
                            auto sz = strlen(placeholder);
                            max_width = sz > max_width ? sz : max_width;
                        }
                    });
                } else {
                    std::stringstream fmt_ss;
                    fmt_ss << "%." << opts.precision << "f";
                    auto format = fmt_ss.str();
                    AT_DISPATCH_FLOATING_TYPES(nonzero_finite_vals.scalar_type(), "max_repr_len", [&] {
                        auto data_ptr = nonzero_finite_vals.data_ptr<scalar_t>();
                        for(int i = 0; i < nonzero_finite_vals.numel(); i++) {
                            char placeholder[64] = {0};
                            snprintf(placeholder, 64, format.c_str(), data_ptr[i]);
                            auto sz = strlen(placeholder);
                            max_width = sz > max_width ? sz : max_width;
                        }
                    });
                }
            }
        }

        if(opts.sci_mode != 0) {
            sci_mode = opts.sci_mode - 1;
        }
    }

    int64_t width() {
        return max_width;
    }

    template<typename T>
    std::string format(T value) {
        std::string format;
        std::string ret(max_width, ' ');
        bool add_dot = false;
        size_t sz;

        if(floating_dtype) {
            if(sci_mode) {
                std::stringstream fmt_ss;
                fmt_ss << "%" << max_width << "." << opts.precision << "e";
                format = fmt_ss.str();
            } else if(int_mode) {
                format = "%.0f";
                if(!(std::isinf(value) || std::isnan(value))) {
                    add_dot = true;
                }
            } else {
                std::stringstream fmt_ss;
                fmt_ss << "%" << "." << opts.precision << "f";
                format = fmt_ss.str();
            }

            snprintf(ret.data(), max_width + 1, format.c_str(), value);
            sz = strlen(ret.c_str());
            ret = ret.substr(0, sz);

        } else {
            std::stringstream ss;
            ss << value;
            ret = ss.str();
            sz = ret.size();
        }

        if(add_dot) {
            ret += ".";
        }

        std::string padding(max_width - sz, ' ');
        return padding + ret;
    }

};

template<>
std::string Formatter::format(uint8_t value) {
    std::string ret(max_width, ' ');
    bool add_dot = false;
    size_t sz;

    std::stringstream ss;
    ss << ((int64_t) value);
    ret = ss.str();
    sz = ret.size();

    if(add_dot) {
        ret += ".";
    }

    std::string padding(max_width - sz, ' ');
    return padding + ret;
}

template <std::ctype_base::mask mask>
class IsNot
{
    std::locale myLocale;       // To ensure lifetime of facet...
    std::ctype<char> const* myCType;
public:
    IsNot( std::locale const& l = std::locale() )
        : myLocale( l )
        , myCType( &std::use_facet<std::ctype<char> >( l ) )
    {
    }
    bool operator()( char ch ) const
    {
        return ! myCType->is( mask, ch );
    }
};

typedef IsNot<std::ctype_base::space> IsNotSpace;

torch::Tensor get_summarized_data(torch::Tensor tensor, const PrintOptions opts) {
    auto dim = tensor.dim();
    if(dim == 0) {
        return tensor;
    } else if(dim == 1) {
        if(tensor.size(0) > 2 * opts.edgeitems) {
            return torch::cat({tensor.index({at::indexing::Slice(c10::nullopt, opts.edgeitems)}),
                               tensor.index({at::indexing::Slice(-opts.edgeitems)}) });
        }
        return tensor;
    }

    if(!opts.edgeitems) {
        std::vector<int64_t> size(dim, 0);
        return tensor.new_empty(size);
    } else if(tensor.size(0) > 2 * opts.edgeitems) {
        auto dim_size = tensor.size(0);
        std::vector<torch::Tensor> values;
        for(int i = 0; i < opts.edgeitems; i++) {
            values.push_back(get_summarized_data(tensor[i], opts));
        }
        for(int i = dim_size - opts.edgeitems; i < dim_size; i++) {
            values.push_back(get_summarized_data(tensor[i], opts));
        }
        return torch::stack(values);
    } else {
        auto dim_size = tensor.size(0);
        std::vector<torch::Tensor> values;
        for(int i = 0; i < dim_size; i++) {
            values.push_back(get_summarized_data(tensor[i], opts));
        }
        return torch::stack(values);
    }

}

std::string lstrip(std::string const& original) {
    std::string::const_iterator left = std::find_if(original.begin(), original.end(), IsNotSpace() );
    return std::string(left, original.end());
}

std::string _scalar_str(torch::Tensor tensor, Formatter* formatter1, Formatter* formatter2 = nullptr) {
    if(formatter2 != nullptr) {
        auto real_str = _scalar_str(torch::real(tensor), formatter1);
        auto imag_str = lstrip(_scalar_str(torch::imag(tensor), formatter2)) + "j";
        if(imag_str[0] == '+' || imag_str[0] == '-') {
            return real_str + imag_str;
        } else {
            return real_str + "+" + imag_str;
        }
    } else {
        std::string result;
        AT_DISPATCH_ALL_TYPES(tensor.scalar_type(), "repr_value", [&] {
            scalar_t value = *tensor.data_ptr<scalar_t>();
            result = formatter1->format<scalar_t>(value);
        });
        return result;
    }
}

std::string _vector_str(
        torch::Tensor tensor, size_t indent, bool summarize,
        Formatter* formatter1, Formatter* formatter2 = nullptr) {

    auto element_length = formatter1->width() + 2;
    if(formatter2 != nullptr) {
      element_length += formatter2->width() + 1;
    }

    auto opts = formatter1->opts;
    auto elements_per_line = (int) floor((opts.linewidth - indent) / ((double) element_length));
    elements_per_line = 1 > elements_per_line ? 1 : elements_per_line;

    std::vector<std::string> data;
    if(summarize && !opts.edgeitems) {
        data.push_back("...");
    } else if(summarize && tensor.size(0) > 2 * opts.edgeitems) {
        auto left_slice = tensor.index({torch::indexing::Slice(c10::nullopt, opts.edgeitems)});
        auto right_slice = tensor.index({torch::indexing::Slice(-opts.edgeitems)});

        for(int i = 0; i < left_slice.numel(); i++) {
            data.push_back(_scalar_str(left_slice[i], formatter1, formatter2));
        }

        data.push_back("...");
        for(int i = 0; i < right_slice.numel(); i++) {
            data.push_back(_scalar_str(right_slice[i], formatter1, formatter2));
        }
    } else {
        for(int i = 0; i < tensor.numel(); i++) {
            data.push_back(_scalar_str(tensor[i], formatter1, formatter2));
        }
    }

    std::ostringstream ss;
    ss << "[";

    int last_chunk = std::max((int)(data.size() - elements_per_line), 0);
    for(int i = 0; i < data.size(); i += elements_per_line) {
        int left_ending = std::min((int) data.size(), i + elements_per_line);
        for(int j = i; j < left_ending; j++) {
            ss << data[j];
            if(j < left_ending - 1) {
                ss << ", ";
            }
        }

        if(i < last_chunk) {
            ss << ",\n" << std::string(indent + 1, ' ');
        }
    }
    ss << "]";
    auto result = ss.str();
    return result;
}

std::string _tensor_str_with_formatter(
        torch::Tensor tensor, size_t indent, bool summarize,
        Formatter* formatter1, Formatter* formatter2 = nullptr) {

    auto dim = tensor.dim();
    auto opts = formatter1->opts;
    if(dim == 0) {
        return _scalar_str(tensor.to(torch::kCPU), formatter1, formatter2);
    } else if(dim == 1) {
        return _vector_str(tensor.to(torch::kCPU), indent, summarize, formatter1, formatter2);
    } else {
        std::vector<std::string> slices;
        if(summarize && tensor.size(0) > 2 * opts.edgeitems) {
            for(int i = 0; i < opts.edgeitems; i++) {
                slices.push_back(_tensor_str_with_formatter(
                    tensor[i].to(torch::kCPU), indent + 1, summarize, formatter1, formatter2));
            }
            slices.push_back("...");

            for(int i = tensor.size(0) - opts.edgeitems; i < tensor.size(0); i++) {
                slices.push_back(_tensor_str_with_formatter(
                    tensor[i].to(torch::kCPU), indent + 1, summarize, formatter1, formatter2));
            }
        } else {
            for(int i = 0; i < tensor.size(0); i++) {
                slices.push_back(_tensor_str_with_formatter(
                    tensor[i].to(torch::kCPU), indent + 1, summarize, formatter1, formatter2));
            }
        }

        std::ostringstream ss;
        ss << "[";
        for(int i = 0; i < slices.size(); i++) {
            ss << slices[i];
            if(i < slices.size() - 1) {
                ss << ",";
                ss << std::string(dim - 1, '\n');
                ss << std::string(indent + 1, ' ');
            }
        }

        ss << "]";
        return ss.str();
    }
}

std::string _tensor_str(torch::Tensor& tensor, const PrintOptions opts, size_t indent) {
    if(tensor.numel() == 0) {
        return "[]";
    } else {
        if(tensor.has_names()) {
            tensor = tensor.rename(c10::nullopt);
        }

        bool summarize = tensor.numel() > opts.threshold;

        if(tensor._is_zerotensor()) {
            tensor = tensor.clone();
        }

        if(!tensor.is_contiguous()) {
            tensor = tensor.contiguous();
        }

        if(tensor.is_neg()) {
            tensor = tensor.resolve_neg();
        }
        if(tensor.scalar_type() == torch::kHalf || tensor.scalar_type() == torch::kBFloat16) {
           tensor = tensor.to(torch::kFloat);
        }

        if(tensor.scalar_type() == torch::kComplexFloat) {
            tensor = tensor.to(torch::kComplexDouble);
        }

        if(tensor.is_complex()) {
            tensor = tensor.resolve_conj();
            auto real_part = torch::real(tensor);
            auto real_formatter = new Formatter(
                summarize ? get_summarized_data(real_part, opts) : real_part,
                opts);

            auto imag_part = torch::imag(tensor);
            auto imag_formatter = new Formatter(
                summarize ? get_summarized_data(imag_part, opts) : imag_part,
                opts);

            return _tensor_str_with_formatter(
                tensor, indent, summarize, real_formatter, imag_formatter);
        } else {
            auto in_tensor = summarize ? get_summarized_data(tensor, opts) : tensor;
            auto formatter = new Formatter(in_tensor, opts);
            return _tensor_str_with_formatter(
                tensor, indent, summarize, formatter);
        }
    }
}
