# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.base import core


class TestReshardPToS:
    def __init__(self):
        self._shape = eval(os.getenv("shape"))
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._shard = eval(os.getenv("shard"))
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self._out_mesh = dist.ProcessMesh([1, 0], dim_names=["x"])

    def reshard_same_mesh(self):
        if self._backend == "gpu":
            place = paddle.CUDAPlace(dist.get_rank())
        dev_ctx = core.DeviceContext.create(place)

        paddle.seed(self._seeds)
        value = paddle.uniform(self._shape, self._dtype)

        input_tensor = dist.shard_tensor(value, self._mesh, [dist.Partial()])

        out_shape = list(self._shape)
        out_shape[self._shard] = out_shape[self._shard] // 2
        out_expected_local_tensor_list = paddle.split(
            value, num_or_sections=self._mesh.shape[0], axis=self._shard
        )

        out = dist.reshard(input_tensor, self._mesh, [dist.Shard(self._shard)])

        np.testing.assert_equal(
            out._local_value().numpy(),
            out_expected_local_tensor_list[0].numpy()
            if dist.get_rank() == 0
            else out_expected_local_tensor_list[1].numpy(),
        )

        assert np.equal(out.shape, input_tensor.shape).all()
        assert np.equal(out._local_shape, out_shape).all()

    def reshard_cross_mesh(self):
        if self._backend != "gpu":
            return

        a = paddle.ones([10, 10])
        input_tensor = dist.shard_tensor(a, self._mesh, [dist.Partial()])
        dist.reshard(input_tensor, self._out_mesh, [dist.Shard(self._shard)])

    def run_test_case(self):
        self.reshard_same_mesh()
        self.reshard_cross_mesh()


if __name__ == '__main__':
    TestReshardPToS().run_test_case()
