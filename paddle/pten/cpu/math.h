/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/pten/core/base_tensor.h"
#include "paddle/pten/module/sign.h"

// fluid headers [may be replaced by new impl]
#include "paddle/fluid/platform/device_context.h"

namespace pt {

using CPUDeviceContext = paddle::platform::CPUDeviceContext;

template <typename T>
void Sign(const CPUDeviceContext& dev_ctx,
          const BaseTensor& x,
          BaseTensor* out) {
  module::Sign<CPUDeviceContext, T>(dev_ctx, x, out);
}

}  // namespace pt
