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

#include "tensorflow/lite/kernels/stablehlo_elementwise.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace stablehlo_count_leading_zeros {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

template <typename DataType>
TfLiteStatus EvalImpl(const TfLiteTensor* operand, TfLiteTensor* result) {
  const int num_elements = NumElements(result);
  const DataType* input = GetTensorData<DataType>(operand);
  DataType* output = GetTensorData<DataType>(result);
  using UIntType = typename std::make_unsigned<DataType>::type;
  int num_bits = sizeof(UIntType) * 8;
  int leading_zeros;

  for (int i = 0; i < num_elements; ++i) {
    UIntType arg = static_cast<UIntType>(input[i]);
    if (arg == 0) {
      leading_zeros = num_bits;
    } else if (num_bits == 64) {
      leading_zeros = __builtin_clzll(arg);
    } else {
      leading_zeros =
          __builtin_clz(static_cast<uint32_t>(arg)) - (32 - num_bits);
    }
    output[i] = static_cast<DataType>(leading_zeros);
  }

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE(context,
                 input->type == kTfLiteInt32 || input->type == kTfLiteInt64 ||
                     input->type == kTfLiteInt16 || input->type == kTfLiteInt8);
  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {int leading_zeros;
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TfLiteType data_type = input->type;

  if (data_type == kTfLiteInt8) {
    return EvalImpl<int8_t>(input, output);
  } else if (data_type == kTfLiteInt16) {
    return EvalImpl<int16_t>(input, output);
  } else if (data_type == kTfLiteInt32) {
    return EvalImpl<int32_t>(input, output);
  } else if (data_type == kTfLiteInt64) {
    return EvalImpl<int64_t>(input, output);
  } else {
    TF_LITE_KERNEL_LOG(context, "(Index Type: %s) currently not supported.\n",
                       TfLiteTypeGetName(data_type));
    return kTfLiteError;
  }
}

}  // namespace
}  // namespace stablehlo_count_leading_zeros

TfLiteRegistration* Register_STABLEHLO_COUNT_LEADING_ZEROS() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 stablehlo_count_leading_zeros::Prepare,
                                 stablehlo_count_leading_zeros::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
