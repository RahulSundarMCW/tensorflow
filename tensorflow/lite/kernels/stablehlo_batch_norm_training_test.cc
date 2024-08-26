/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class StablehloBatchNormTrainingOpModel : public SingleOpModel {
 public:
  StablehloBatchNormTrainingOpModel(
      const TensorData& input, const TensorData& scale,
      const TensorData& offset,
      const TfLiteStablehloBatchNormTrainingParams& params) {
    input_ = AddInput(input);
    scale_ = AddInput(scale);
    offset_ = AddInput(offset);
    output_ = AddOutput(input.type);
    batch_mean_ = AddOutput(input.type);
    batch_var_ = AddOutput(input.type);
    SetBuiltinOp(BuiltinOperator_STABLEHLO_BATCH_NORM_TRAINING,
                 BuiltinOptions2_StablehloBatchNormTrainingOptions,
                 CreateStablehloBatchNormTrainingOptions(
                     builder_, params.epsilon, params.feature_index)
                     .Union());
    BuildInterpreter({GetShape(input_), GetShape(scale_), GetShape(offset_)},
                     /*num_threads=*/-1, /*allow_fp32_relax_to_fp16=*/false,
                     /*apply_delegate=*/false, /*allocate_and_delegate=*/false,
                     /*use_simple_allocator=*/false);

    AllocateAndDelegate(true);
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }

  template <typename T>
  void SetScale(std::initializer_list<T> data) {
    PopulateTensor<T>(scale_, data);
  }

  template <typename T>
  void SetOffset(std::initializer_list<T> data) {
    PopulateTensor<T>(offset_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  template <typename T>
  std::vector<T> GetBatchMean() {
    return ExtractVector<T>(batch_mean_);
  }

  template <typename T>
  std::vector<T> GetBatchVar() {
    return ExtractVector<T>(batch_var_);
  }

 protected:
  int input_;
  int scale_;
  int offset_;
  int output_;
  int batch_mean_;
  int batch_var_;
};

TEST(StablehloBatchNormTrainingOpTest, Ex1) {
  TfLiteStablehloBatchNormTrainingParams params = {0.0, 2};
  StablehloBatchNormTrainingOpModel model(
      {TensorType_FLOAT32, {2, 2, 2}}, {TensorType_INT64, {2}},
      {TensorType_FLOAT32, {2}}, params);
  model.SetInput<float>({1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 1.0, 2.0});
  model.SetScale<float>({1.0, 1.0});
  model.SetOffset<float>({1.0, 1.0});

  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<float> expected_values = {0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0};
  std::vector<float> expected_mean = {2.0, 3.0};
  std::vector<float> expected_var = {1.0, 1.0};
  EXPECT_THAT(model.GetOutput<float>(), ElementsAreArray(expected_values));
  EXPECT_THAT(model.GetBatchMean<float>(), ElementsAreArray(expected_mean));
  EXPECT_THAT(model.GetBatchVar<float>(), ElementsAreArray(expected_var));
}

}  // namespace
}  // namespace tflite