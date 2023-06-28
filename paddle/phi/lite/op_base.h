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

#include <string>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/lite/kernel_base.h"

namespace phi {
namespace lite {

class OpContext {
 public:
  OpContext();

  virtual const phi::DenseTensor* GetTensor(const std::string& name) const = 0;

  virtual phi::DenseTensor* GetMutableTensor(const std::string& name) = 0;

  template <typename T>
  T GetAttr(const std::string& name) const {
    T attr;
    GetAttr(name, &attr);
    return attr;
  }

 private:
  virtual void GetAttr(const std::string& name, float* attr) const {}
  virtual void GetAttr(const std::string& name, double* attr) const {}
  virtual void GetAttr(const std::string& name, int* attr) const {}
  virtual void GetAttr(const std::string& name, bool* attr) const {}
};

/**
 * The base class of operator in lite.
 */
class OpBase {
 public:
  OpBase() = default;

  explicit OpBase(const std::string& op_type) : op_type_(op_type) {}

  virtual ~OpBase() = default;

  const std::string& Type() const { return op_type_; }

  virtual void InferMeta();

  virtual void AttachParam(OpContext* op_context) = 0;

  virtual std::vector<std::unique_ptr<LiteKernel>> CreateKernels(
      const std::vector<phi::KernelKey>& kernel_keys) const = 0;

 private:
  const std::string op_type_;
};

}  // namespace lite
}  // namespace phi
