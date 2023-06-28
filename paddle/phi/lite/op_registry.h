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

#include <functional>
#include <memory>
#include <unordered_map>

#include "paddle/phi/lite/op_base.h"

namespace phi {
namespace lite {

class LiteOpFactory {
 public:
  // Register a function to create an op
  void RegisterCreator(const std::string& op_type,
                       std::function<std::shared_ptr<OpBase>()> fun) {
    op_factory_[op_type] = fun;
  }

  static LiteOpFactory& Instance() {
    static LiteOpFactory* x = new LiteOpFactory();
    return *x;
  }

  std::shared_ptr<OpBase> Create(const std::string& op_type) const {
    auto it = op_factory_.find(op_type);
    if (it == op_factory_.end()) return nullptr;
    return it->second();
  }

 private:
  std::unordered_map<std::string, std::function<std::shared_ptr<OpBase>()>>
      op_factory_;
};

// Register OpBase by initializing a static LiteOpRegistrar instance
class LiteOpRegistrar {
 public:
  LiteOpRegistrar(const std::string& op_type,
                  std::function<std::shared_ptr<OpBase>()> fun) {
    LiteOpFactory::Instance().RegisterCreator(op_type, fun);
  }
};

}  // namespace lite
}  // namespace phi

// Register an op.
#define PD_REGISTER_LITE_OP(op_type__, OpClass)                              \
  static phi::lite::LiteOpRegistrar op_type__##__registry(#op_type__, []() { \
    return std::unique_ptr<phi::lite::OpBase>(new OpClass(#op_type__));      \
  });                                                                        \
  int touch_op_##op_type__() { return 0; }
