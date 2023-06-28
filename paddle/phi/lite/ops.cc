// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/lite/op_base.h"
#include "paddle/phi/lite/op_registry.h"

#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/lite/kernels.h"

namespace phi {
namespace lite {

class ScaleOp : public OpBase {
 public:
  explicit ScaleOp(const std::string& op_type) : OpBase(op_type) {}

  void InferMeta() override {
    phi::UnchangedInferMeta(phi::MetaTensor(*param_.x),
                            &phi::MetaTensor(param_.output));
  }

  void AttachParam(OpContext* op_context) override {
    param_.x = op_context->GetTensor("X");
    param_.output = op_context->GetMutableTensor("Out");
    param_.scale = op_context->GetAttr<float>("scale");
    param_.bias = op_context->GetAttr<float>("bias");
    param_.bias_after_scale = op_context->GetAttr<bool>("bias_after_scale");
  }

  std::vector<std::unique_ptr<LiteKernel>> CreateKernels(
      const std::vector<phi::KernelKey>& kernel_keys) const override {
    std::vector<std::unique_ptr<LiteKernel>> kernels;
    for (const auto& kernel_key : kernel_keys) {
      const auto& kernel =
          phi::KernelFactory::Instance().SelectKernel(Type(), kernel_key);
      auto kernel_ptr = std::make_unique<ScaleKernel>(kernel_key, kernel);
      kernel_ptr->SetParam(param_);
      kernels.emplace_back(std::make_unique<ScaleKernel>(kernel_key, kernel));
    }
  }

 private:
  ScaleParam param_;
};

}  // namespace lite
}  // namespace phi

PD_REGISTER_LITE_OP(scale, phi::lite::ScaleOp);
