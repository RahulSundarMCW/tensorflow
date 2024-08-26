#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/add.h"
#include "tensorflow/lite/kernels/internal/reference/broadcast_to.h"
#include "tensorflow/lite/kernels/internal/reference/div.h"
#include "tensorflow/lite/kernels/internal/reference/mul.h"
#include "tensorflow/lite/kernels/internal/reference/reduce.h"
#include "tensorflow/lite/kernels/internal/reference/sub.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace batch_norm_training {
namespace {

constexpr int kInputTensor = 0;
constexpr int kScaleTensor = 1;
constexpr int kOffsetTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kBatchMeanTensor = 1;
constexpr int kBatchVarTensor = 2;

template <typename DataType>
TfLiteStatus EvalImpl(const TfLiteTensor* operand, const TfLiteTensor* scale,
                      const TfLiteTensor* offset, TfLiteTensor* output,
                      TfLiteTensor* batch_mean, TfLiteTensor* batch_var,
                      int feature_index, float epsilon) {
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 3);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* scale;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kScaleTensor, &scale));
  const TfLiteTensor* offset;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kOffsetTensor, &offset));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TfLiteTensor* batch_mean;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kBatchMeanTensor, &batch_mean));
  TfLiteTensor* batch_var;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kBatchVarTensor, &batch_var));

  const TfLiteStablehloBatchNormTrainingParams* data =
      reinterpret_cast<TfLiteStablehloBatchNormTrainingParams*>(
          node->builtin_data);

  const int feature_index = data->feature_index;
  TF_LITE_ENSURE(context,
                 feature_index >= 0 && feature_index < input->dims->size);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, scale->type);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, offset->type);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, batch_mean->type);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, batch_var->type);

  // TF_LITE_ENSURE_EQ(context, scale->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, scale->dims->data[0],
                    input->dims->data[feature_index]);

  // TF_LITE_ENSURE_EQ(context, offset->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, offset->dims->data[0],
                    input->dims->data[feature_index]);

  // TF_LITE_ENSURE_EQ(context, batch_mean->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, batch_mean->dims->data[0],
                    input->dims->data[feature_index]);

  // TF_LITE_ENSURE_EQ(context, batch_var->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, batch_var->dims->data[0],
                    input->dims->data[feature_index]);

  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input->type);

  TF_LITE_ENSURE_OK(
      context,
      context->ResizeTensor(context, output, TfLiteIntArrayCopy(input->dims)));

  TfLiteIntArray* mean_var_shape = TfLiteIntArrayCreate(1);
  mean_var_shape->data[0] = input->dims->data[feature_index];
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, batch_mean, mean_var_shape));
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, batch_var, mean_var_shape));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* operand;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor, &operand));
  const TfLiteTensor* scale;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kScaleTensor, &scale));
  const TfLiteTensor* offset;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kOffsetTensor, &offset));

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TfLiteTensor* batch_mean;
  TF_LITE_ENSURE_OK(
      context, GetOutputSafe(context, node, kBatchMeanTensor, &batch_mean));
  TfLiteTensor* batch_var;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kBatchVarTensor, &batch_var));

  const TfLiteStablehloBatchNormTrainingParams* data =
      reinterpret_cast<TfLiteStablehloBatchNormTrainingParams*>(
          node->builtin_data);

  int feature_index = data->feature_index;
  float epsilon = data->epsilon;

  if (operand->type == kTfLiteFloat32) {
    return EvalImpl<float>(operand, scale, offset, output, batch_mean,
                           batch_var, feature_index, epsilon);
  } else if (operand->type == kTfLiteFloat16) {
    return EvalImpl<Eigen::half>(operand, scale, offset, output, batch_mean,
                                 batch_var, feature_index, epsilon);
  } else if (operand->type == kTfLiteBFloat16) {
    return EvalImpl<Eigen::bfloat16>(operand, scale, offset, output, batch_mean,
                                     batch_var, feature_index, epsilon);
  } else if (operand->type == kTfLiteInt32) {
    return EvalImpl<int32_t>(operand, scale, offset, output, batch_mean,
                             batch_var, feature_index, epsilon);
  } else {
    TF_LITE_KERNEL_LOG(context, "Data type %s not supported.",
                       TfLiteTypeGetName(operand->type));
    return kTfLiteError;
  }
}

}  // namespace
}  // namespace batch_norm_training

TfLiteRegistration* Register_BATCH_NORM_TRAINING() {
  static TfLiteRegistration r = {nullptr, nullptr, batch_norm_training::Prepare,
                                 batch_norm_training::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
