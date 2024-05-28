/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_SORT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_SORT_H_

#include <functional>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

class sortOp {
 public:
  struct Attributes {
    int64_t dimension;
    bool is_stable;
  };
  Attributes attributes;
  Tensor operand_dequantized;
  Tensor output_dequantized;
  std::vector<std::byte> operand_dequantized_data;
  std::vector<std::byte> output_dequantized_data;
};

sortOp Create(sortOp::Attributes attributes);

absl::Status Prepare(sortOp& op, const std::vector<Tensor>& inputs,
                     std::vector<Tensor>& results);

absl::Status Evaluate(sortOp& op, const std::vector<Tensor>& inputs,
                      std::vector<Tensor>& results);

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_SORT_H_
