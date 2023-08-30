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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_dialect.h"
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_op.h"
#include "paddle/fluid/ir/drr/api/drr_pattern_base.h"
#include "paddle/ir/pass/pass.h"
#include "paddle/ir/pass/pass_manager.h"
#include "paddle/ir/pattern_rewrite/pattern_rewrite_driver.h"

class FusedLinearPattern : public ir::drr::DrrPatternBase<FusedLinearPattern> {
 public:
  void operator()(ir::drr::DrrPatternContext *ctx) const override {
    ir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op("pd.matmul",
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    const auto &add = pat.Op("pd.add");

    pat.Tensor("tmp") = matmul(pat.Tensor("x"), pat.Tensor("w"));
    pat.Tensor("out") = add(pat.Tensor("tmp"), pat.Tensor("bias"));

    // Result patterns：要替换为的子图
    ir::drr::ResultPattern res = pat.ResultPattern();
    const auto &act_attr =
        res.Attr([](const ir::drr::MatchContext &match_ctx) -> std::any {
          return "none";
        });
    const auto &fused_gemm_epilogue = res.Op("pd.fused_gemm_epilogue",
                                             {{{"trans_x", pat.Attr("trans_x")},
                                               {"trans_y", pat.Attr("trans_y")},
                                               {"activation", act_attr}}});
    fused_gemm_epilogue(
        {&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("bias")},
        {&res.Tensor("out")});
  }
};

class FusedLinearGradPattern
    : public ir::drr::DrrPatternBase<FusedLinearGradPattern> {
 public:
  void operator()(ir::drr::DrrPatternContext *ctx) const override {
    ir::drr::SourcePattern pat = ctx->SourcePattern();
    const auto &matmul = pat.Op("pd.matmul",
                                {{"transpose_x", pat.Attr("trans_x")},
                                 {"transpose_y", pat.Attr("trans_y")}});
    const auto &matmul_grad = pat.Op("pd.matmul_grad",
                                     {{"transpose_x", pat.Attr("trans_x")},
                                      {"transpose_y", pat.Attr("trans_y")}});
    const auto &add = pat.Op("pd.add");
    const auto &add_grad = pat.Op("pd.add_grad");

    pat.Tensor("tmp") = matmul(pat.Tensor("x"), pat.Tensor("w"));
    pat.Tensor("out") = add(pat.Tensor("tmp"), pat.Tensor("bias"));
    add_grad({&pat.Tensor("tmp"), &pat.Tensor("bias"), &pat.Tensor("out_grad")},
             {&pat.Tensor("tmp_grad"), &pat.Tensor("bias_grad")});
    matmul_grad({&pat.Tensor("x"), &pat.Tensor("w"), &pat.Tensor("tmp_grad")},
                {&pat.Tensor("x_grad"), &pat.Tensor("w_grad")});

    // Result patterns：要替换为的子图
    ir::drr::ResultPattern res = pat.ResultPattern();
    const auto &act_attr =
        res.Attr([](const ir::drr::MatchContext &match_ctx) -> std::any {
          return "none";
        });
    const auto &fused_gemm_epilogue = res.Op("pd.fused_gemm_epilogue",
                                             {{{"trans_x", pat.Attr("trans_x")},
                                               {"trans_y", pat.Attr("trans_y")},
                                               {"activation", act_attr}}});
    const auto &fused_gemm_epilogue_grad =
        res.Op("pd.fused_gemm_epilogue_grad",
               {{{"trans_x", pat.Attr("trans_x")},
                 {"trans_y", pat.Attr("trans_y")},
                 {"activation_grad", act_attr}}});
    fused_gemm_epilogue(
        {&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("bias")},
        {&res.Tensor("out")});
    fused_gemm_epilogue_grad(
        {&res.Tensor("x"), &res.Tensor("w"), &res.Tensor("out_grad")},
        {&res.Tensor("x_grad"),
         &res.Tensor("w_grad"),
         &res.Tensor("bias_grad")});
  }
};

class FusedLinearPass : public ir::Pass {
 public:
  FusedLinearPass() : ir::Pass("FusedLinearPass", 1) {}

  bool Initialize(ir::IrContext *context) override {
    ir::RewritePatternSet ps(context);
    ps.Add(FusedLinearPattern().Build(context));
    ps.Add(FusedLinearGradPattern().Build(context));

    patterns_ = ir::FrozenRewritePatternSet(std::move(ps));
    return true;
  }

  void Run(ir::Operation *op) override {
    ir::GreedyRewriteConfig cfg;
    cfg.use_top_down_traversal = true;
    cfg.max_iterations = 10;
    ir::ApplyPatternsGreedily(op->region(0), patterns_, cfg);
  }

  bool CanApplyOn(ir::Operation *op) const override {
    return op->name() == "builtin.module" && op->num_regions() > 0;
  }

 private:
  ir::FrozenRewritePatternSet patterns_;
};

void BuildProgram(ir::Builder &builder) {  // NOLINT
  VLOG(1) << "####### BuildProgram Start";
  paddle::dialect::FullOp full_input_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{1, 512, 64},
                                             1.5);
  // linear 1
  paddle::dialect::FullOp full_weight_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 64}, 1.5);
  paddle::dialect::FullOp full_bias_op1 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64}, 1.0);
  paddle::dialect::MatmulOp matmul_op1 =
      builder.Build<paddle::dialect::MatmulOp>(full_input_op1.out(),
                                               full_weight_op1.out());
  paddle::dialect::AddOp add_op1 = builder.Build<paddle::dialect::AddOp>(
      matmul_op1.out(), full_bias_op1.out());
  // linear 2
  paddle::dialect::FullOp full_weight_op2 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 64}, 1.5);
  paddle::dialect::FullOp full_bias_op2 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64}, 1.0);
  paddle::dialect::MatmulOp matmul_op2 =
      builder.Build<paddle::dialect::MatmulOp>(add_op1.out(),
                                               full_weight_op2.out());
  paddle::dialect::AddOp add_op2 = builder.Build<paddle::dialect::AddOp>(
      matmul_op2.out(), full_bias_op2.out());
  paddle::dialect::ReluOp relu_op =
      builder.Build<paddle::dialect::ReluOp>(add_op2.out());
  // linear 3
  paddle::dialect::FullOp full_weight_op3 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 64}, 1.5);
  paddle::dialect::FullOp full_bias_op3 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64}, 1.0);
  paddle::dialect::MatmulOp matmul_op3 =
      builder.Build<paddle::dialect::MatmulOp>(relu_op.out(),
                                               full_weight_op3.out());
  paddle::dialect::AddOp add_op3 = builder.Build<paddle::dialect::AddOp>(
      matmul_op3.out(), full_bias_op3.out());
  paddle::dialect::GeluOp gelu_op1 =
      builder.Build<paddle::dialect::GeluOp>(add_op3.out());
  // linear 4
  paddle::dialect::FullOp full_weight_op4 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64, 64}, 1.5);
  paddle::dialect::FullOp full_bias_op4 =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{64}, 1.0);
  paddle::dialect::MatmulOp matmul_op4 =
      builder.Build<paddle::dialect::MatmulOp>(gelu_op1.out(),
                                               full_weight_op4.out());
  paddle::dialect::AddOp add_op4 = builder.Build<paddle::dialect::AddOp>(
      matmul_op4.out(), full_bias_op4.out());
  paddle::dialect::GeluOp gelu_op2 =
      builder.Build<paddle::dialect::GeluOp>(add_op4.out());

  // backward
  paddle::dialect::FullOp full_grad_op = builder.Build<paddle::dialect::FullOp>(
      std::vector<int64_t>{1, 512, 64}, 1.0);

  paddle::dialect::GeluGradOp gelu_op2_grad =
      builder.Build<paddle::dialect::GeluGradOp>(
          add_op4.out(), full_grad_op.out(), false);
  // backward linear 4
  paddle::dialect::AddGradOp add_op4_grad =
      builder.Build<paddle::dialect::AddGradOp>(
          matmul_op4.out(), full_bias_op4.out(), gelu_op2_grad.x_grad());
  paddle::dialect::MatmulGradOp matmul_op4_grad =
      builder.Build<paddle::dialect::MatmulGradOp>(
          gelu_op1.out(), full_weight_op4.out(), add_op4_grad.x_grad());

  paddle::dialect::GeluGradOp gelu_op1_grad =
      builder.Build<paddle::dialect::GeluGradOp>(
          add_op3.out(), matmul_op4_grad.x_grad(), false);
  // backward linear 3
  paddle::dialect::AddGradOp add_op3_grad =
      builder.Build<paddle::dialect::AddGradOp>(
          matmul_op3.out(), full_bias_op3.out(), gelu_op1_grad.x_grad());
  paddle::dialect::MatmulGradOp matmul_op3_grad =
      builder.Build<paddle::dialect::MatmulGradOp>(
          relu_op.out(), full_weight_op3.out(), add_op3_grad.x_grad());

  paddle::dialect::ReluGradOp relu_op_grad =
      builder.Build<paddle::dialect::ReluGradOp>(add_op2.out(),
                                                 matmul_op3_grad.x_grad());
  // backward linear 2
  paddle::dialect::AddGradOp add_op2_grad =
      builder.Build<paddle::dialect::AddGradOp>(
          matmul_op2.out(), full_bias_op2.out(), relu_op_grad.x_grad());
  paddle::dialect::MatmulGradOp matmul_op2_grad =
      builder.Build<paddle::dialect::MatmulGradOp>(
          add_op1.out(), full_weight_op2.out(), add_op2_grad.x_grad());
  // backward linear 1
  paddle::dialect::AddGradOp add_op1_grad =
      builder.Build<paddle::dialect::AddGradOp>(
          matmul_op1.out(), full_bias_op1.out(), matmul_op2_grad.x_grad());
  paddle::dialect::MatmulGradOp matmul_op1_grad =
      builder.Build<paddle::dialect::MatmulGradOp>(
          full_input_op1.out(), full_weight_op1.out(), add_op1_grad.x_grad());

  VLOG(1) << "####### BuildProgram FetchOp";
  builder.Build<paddle::dialect::FetchOp>(gelu_op2.out(), "out", 0);
  builder.Build<paddle::dialect::FetchOp>(matmul_op1_grad.x_grad(), "dx", 1);
  VLOG(1) << "####### BuildProgram Finish";
}

TEST(DrrTest, FusedLinear) {
  ir::IrContext *ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();
  ir::Program program(ctx);
  ir::Builder builder = ir::Builder(ctx, program.block());
  BuildProgram(builder);

  EXPECT_EQ(program.block()->size(), 34u);

  ir::PassManager pm(ctx);
  pm.AddPass(std::make_unique<FusedLinearPass>());
  // pm.AddPass(ir::CreateDeadCodeEliminationPass());
  // pm.EnablePassTiming();
  pm.EnableIRPrinting();

  CHECK_EQ(pm.Run(&program), true);
  EXPECT_EQ(program.block()->size(), 34u);
}
