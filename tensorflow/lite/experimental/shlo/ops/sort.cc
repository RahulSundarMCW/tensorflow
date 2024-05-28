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

#include "tensorflow/lite/experimental/shlo/ops/sort.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

template <typename DataType>
bool Comparator(DataType lhs, DataType rhs) {
  return lhs > rhs;
}

absl::Status CheckParameters(const std::vector<Tensor>& inputs,
                             int64_t dimension, bool is_stable,
                             std::vector<Tensor>& results) {
  // Constraint 1
  if (inputs.empty()) {
    return absl::InvalidArgumentError(
        "There must be at least one input tensor.");
  }

  // Constraint 2
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].tensor_element_type() != results[i].tensor_element_type()) {
      return absl::FailedPreconditionError(
          "Input and result tensor types must be the same.");
    }
  }

  // Constraint 3
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].shape() != results[i].shape()) {
      return absl::FailedPreconditionError(
          "Input and result tensor shapes must be the same.");
    }
  }

  // Constraint 4
  int64_t rank = inputs[0].Rank();
  if (dimension < -rank || dimension >= rank) {
    return absl::InvalidArgumentError("Dimension is out of range.");
  }

  return absl::OkStatus();
}

template <DataType storage_type>

absl::Status SortImpl(const std::vector<Tensor>& inputs, int64_t dimension,
                      bool is_stable, std::vector<Tensor>& outputs) {
  using StorageT = StorageType<storage_type>;
  auto adjusted_dimension =
      dimension >= 0 ? dimension : dimension + inputs[0].Rank();
  auto rank = inputs[0].Rank();

  const DimensionSize operand_size = outputs[0].NumElements();
  const Axis operand_rank = outputs[0].Rank();

  absl::InlinedVector<DimensionSize, kMaxNumDimensions> operand_index(
      operand_rank);
  absl::InlinedVector<DimensionSize, kMaxNumDimensions> output_index(
      operand_rank);

  for (DimensionSize k = 0; k < operand_size; ++k) {
    outputs[0].GetNdIndex(k, operand_index);
    auto result_it = operand_index;
    if ((result_it)[adjusted_dimension] != 0) continue;

    std::vector<int64_t> indices(inputs[0].shape().Dim(adjusted_dimension));
    std::iota(indices.begin(), indices.end(), 0);

    auto comparator_wrapper = [&](int64_t lhs_handle, int64_t rhs_handle) {
      std::vector<float> args;
      auto lhs_index = result_it;
      auto rhs_index = result_it;
      lhs_index[adjusted_dimension] = lhs_handle;
      rhs_index[adjusted_dimension] = rhs_handle;
      for (const auto& input : inputs) {
        args.emplace_back(input.Get<storage_type>(lhs_index));
        args.emplace_back(input.Get<storage_type>(rhs_index));
      }
      return Comparator(args[0], args[1]);
    };

    if (is_stable) {
      std::stable_sort(indices.begin(), indices.end(), comparator_wrapper);
    } else {
      std::sort(indices.begin(), indices.end(), comparator_wrapper);
    }

    for (size_t input_handle = 0; input_handle < indices.size();
         ++input_handle) {
      int64_t result_handle = indices[input_handle];
      for (size_t i = 0; i < inputs.size(); ++i) {
        auto input_index = result_it;
        auto result_index = result_it;
        input_index[adjusted_dimension] = input_handle;
        result_index[adjusted_dimension] = result_handle;
        StorageT element(inputs[i].Get<storage_type>(input_index));
        outputs[i].Set<storage_type>(result_index, element);
      }
    }
  }

  return absl::OkStatus();
}

sortOp Create(sortOp::Attributes attributes) {
  return {.attributes = attributes};
}

absl::Status Prepare(sortOp& op, const std::vector<Tensor>& inputs,
                     std::vector<Tensor>& results) {
  if (absl::Status status = CheckParameters(inputs, op.attributes.dimension,
                                            op.attributes.is_stable, results);
      !status.ok()) {
    return status;
  }
  return absl::OkStatus();
}

absl::Status Evaluate(sortOp& op, const std::vector<Tensor>& inputs,
                      std::vector<Tensor>& results) {
  DISPATCH_INT_FLOAT(SortImpl, inputs[0].StorageType(), inputs,
                     op.attributes.dimension, op.attributes.is_stable, results);

  return absl::FailedPreconditionError("Unsupported tensor type.");
}

}  // namespace shlo_ref
