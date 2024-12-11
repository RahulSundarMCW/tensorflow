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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using testing::ElementsAreArray;

class ComplexOpModel : public SingleOpModel {
 public:
  ComplexOpModel(const TensorData& input1, const TensorData& input2,
                 const TensorData& output) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_STABLEHLO_COMPLEX, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  template <typename T>
  void SetInputs(std::initializer_list<T> data_input1,
                 std::initializer_list<T> data_input2) {
    PopulateTensor<T>(input1_, data_input1);
    PopulateTensor<T>(input2_, data_input2);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

 protected:
  int input1_;
  int input2_;
  int output_;
};

TEST(ComplexOpTest, Float32Test) {
  ComplexOpModel model({TensorType_FLOAT32, {3, 4}},
                       {TensorType_FLOAT32, {3, 4}},
                       {TensorType_COMPLEX64, {}});
  model.SetInputs<float>({-5.07035875, -3.72310281, -4.24285221, 4.083100e+00,
                          -3.61923885, -2.82356691, 7.91711425, -0.592124462,
                          -5.61838675, 5.38161421, -2.99476314, 1.32623148},
                         {-1.51469374, 0.660175204, -2.72695398, -1.17417383,
                          3.38128543, 0.810583353, -1.625512, 2.66601849,
                          4.05134487, 4.48275757, -2.34912467, -0.595442355});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<std::complex<float>> expected_values = {
      {-5.07035875, -1.51469374}, {-3.72310281, 0.660175204},
      {-4.24285221, -2.72695398}, {4.083100e+00, -1.17417383},
      {-3.61923885, 3.38128543},  {-2.82356691, 0.810583353},
      {7.91711425, -1.625512},    {-0.592124462, 2.66601849},
      {-5.61838675, 4.05134487},  {5.38161421, 4.48275757},
      {-2.99476314, -2.34912467}, {1.32623148, -0.595442355}};
  std::vector<std::complex<float>> result_values =
      model.GetOutput<std::complex<float>>();
  EXPECT_THAT(result_values, ElementsAreArray(expected_values));
}

TEST(ComplexOpTest, Float64Test) {
  ComplexOpModel model({TensorType_FLOAT64, {2}}, {TensorType_FLOAT64, {2}},
                       {TensorType_COMPLEX128, {}});
  model.SetInputs<_Float64>({1.0, 3.0}, {2.0, 4.0});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  std::vector<std::complex<_Float64>> expected_values = {
      {1.0f, 2.0f},
      {3.0f, 4.0f},
  };
  std::vector<std::complex<_Float64>> result_values =
      model.GetOutput<std::complex<_Float64>>();
  EXPECT_THAT(result_values, ElementsAreArray(expected_values));
}

}  // namespace
}  // namespace tflite