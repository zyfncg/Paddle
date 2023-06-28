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

#pragma once

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/kernel_factory.h"

#include "paddle/phi/lite/kernel_base.h"
#include "paddle/phi/lite/op_params.h"

namespace phi {
namespace lite {

class ScaleKernel : public LiteKernel {
 public:
  void SetParam(const ScaleParam& param) { param_ = param; }

  void Run() override {
    using kernel_signature = void (*)(const phi::DeviceContext&,
                                      const phi::DenseTensor&,
                                      const phi::Scalar&,
                                      float,
                                      bool,
                                      phi::DenseTensor*);
    auto* kernel_fn = kernel_.GetVariadicKernelFn<kernel_signature>();
    kernel_fn(*dev_ctx_,
              *param_.x,
              param_.scale,
              param_.bias,
              param_.bias_after_scale,
              param_.output);
  }

 private:
  ScaleParam param_;
};

}  // namespace lite
}  // namespace phi
