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

#include <vector>

#include "paddle/ir/pattern_rewrite/drr/api/drr_pass_context.h"
#include "paddle/ir/pattern_rewrite/pattern_match.h"

namespace ir {
namespace drr {

template <typename SourceOp, typename DrrFunctor>
struct DrrRewritePattern : public ir::OpRewritePattern<SourceOp> {
  DrrRewritePattern(ir::IrContext* context, ir::PatternBenefit benefit)
      : ir::OpRewritePattern<SourceOp>(context, benefit) {
    DrrPassContext drr_context;
    DrrFunctor functor;
    functor(&drr_context);

    source_pattern_graph_ = drr_context.source_pattern_graph();
    constraints_ = drr_context.constraints();
    result_pattern_graph_ = drr_context.result_pattern_graph();
  }

  bool Match(SourceOp op) const override {
    // Match

    return true;
  }

  void Rewrite(SourceOp op,
               ir::PatternRewriter& rewriter) const override {  // NOLINT
    // Rewrite
  }

  std::shared_ptr<SourcePatternGraph> source_pattern_graph_;
  std::vector<Constrain> constraints_;
  std::shared_ptr<ResultPatternGraph> result_pattern_graph_;
};

}  // namespace drr
}  // namespace ir
