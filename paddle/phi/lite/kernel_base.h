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

#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/kernel_factory.h"

namespace phi {
namespace lite {

class LiteKernel {
 public:
  LiteKernel(KernelKey kernel_key, const Kernel& kernel)
      : kernel_key_(kernel_key), kernel_{kernel} {}

  virtual void Run() = 0;

  void SetContext(DeviceContext* context) { dev_ctx_ = context; }

  Backend GetBackend() const { return kernel_key_.backend(); }

 protected:
  KernelKey kernel_key_;
  const Kernel& kernel_;
  phi::DeviceContext* dev_ctx_{nullptr};
};

}  // namespace lite
}  // namespace phi
