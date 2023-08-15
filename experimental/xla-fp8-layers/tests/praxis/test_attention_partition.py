from functools import partial
from absl.testing import absltest

import numpy as np
import os
import re

from jax import numpy as jnp
from jax import random
from jax import value_and_grad
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.tree_util import tree_map
from praxis import test_utils

from fp8layers.flax import fp8_qkv_combined_projection

class AttnPartitionTest(test_utils.TestCase):
  def setUp(self):
    # The tests need to check the dtypes of the cublaslt custom calls, so we
    # disable the triton gemms.
    os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'

    super().setUp()
    np.random.seed(123456)

  def testQKVCombinedProjFwd(self):
    prng_key = random.PRNGKey(seed=123)
    prng_key, init_key, random_key = random.split(prng_key, 3)

    B, T, D, N, H, K = (1, 2048, 12288, 96, 128, 3)
    dtype = jnp.bfloat16

    # Initialize the inputs and variables.
    x = random.uniform(random_key, (B, T, D)).astype(dtype)
    w = random.uniform(random_key, (K, D, N, H)).astype(dtype)
    b = random.uniform(random_key, (K, N, H)).astype(dtype)

    # Initialize the fp8 related variables.
    x_scale = jnp.ones((1,))
    w_scale = jnp.ones((1,))
    dy_scale = jnp.ones((1,))
    x_amax_history = jnp.zeros((16,))
    w_amax_history = jnp.zeros((16,))
    dy_amax_history = jnp.zeros((16,))

    def _infer_fp8(x, var, use_bias):
      y = fp8_qkv_combined_projection(
          x, var['w'], use_bias, var['b'], var['x_scale'],
          var['x_amax_history'], var['w_scale'], var['w_amax_history'],
          var['dy_scale'], var['dy_amax_history'])
      return y

    var_fp8 = {'w': w, 'b': b, 'x_scale': x_scale,
               'x_amax_history': x_amax_history, 'w_scale': w_scale,
               'w_amax_history': w_amax_history, 'dy_scale': dy_scale,
               'dy_amax_history': dy_amax_history}

    var_fp8_pspecs = tree_map(lambda _: None, var_fp8)
    var_fp8_pspecs['w'] = P(None, None, 'model', None)
    in_shardings = [P(None, None, 'model'), var_fp8_pspecs]
    
    infer_fn_fp8 = pjit(partial(_infer_fp8, use_bias=True),
                        in_shardings=in_shardings)

    mesh_names = ('data', 'model')
    mesh_shape = (2, 4)
    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(devices=device_mesh, axis_names=mesh_names)

    with mesh:
      lowered = infer_fn_fp8.lower(x, var_fp8)
    hlo_text = lowered.compile().as_text()

    # The inserted AllToAll collective prevents the fp8 optimization.
    # TODO(shuw): we can check fp8 dtype when the above issue is fixed.
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            f'bf16[{B*T},{K*N*H}]{{1,0}}', # outputs
            'custom-call',
            f'bf16[{B*T},{D//4}]{{1,0}}', # inputs
            f'bf16[{D//4},{K*N*H}]{{1,0}}', # kernel
            'epilogue',
            'DEFAULT',
        )])),
        msg='output tensor',
    )

  def testQKVCombinedProjBwd(self):
    prng_key = random.PRNGKey(seed=123)
    prng_key, init_key, random_key = random.split(prng_key, 3)

    B, T, D, N, H, K = (1, 2048, 12288, 96, 128, 3)
    dtype = jnp.bfloat16

    # Initialize the inputs and variables.
    x = random.uniform(random_key, (B, T, D)).astype(dtype)
    w = random.uniform(random_key, (K, D, N, H)).astype(dtype)
    b = random.uniform(random_key, (K, N, H)).astype(dtype)
    dy = random.normal(random_key, (K, B, T, N, H))

    # Initialize the fp8 related variables.
    x_scale = jnp.ones((1,))
    w_scale = jnp.ones((1,))
    dy_scale = jnp.ones((1,))
    x_amax_history = jnp.zeros((16,))
    w_amax_history = jnp.zeros((16,))
    dy_amax_history = jnp.zeros((16,))

    def _train_fp8(x, dy, var, use_bias):
      y = fp8_qkv_combined_projection(
          x, var['w'], use_bias, var['b'], var['x_scale'],
          var['x_amax_history'], var['w_scale'], var['w_amax_history'],
          var['dy_scale'], var['dy_amax_history'])
      loss = y * dy.astype(y.dtype)
      return jnp.sum(loss)

    var_fp8 = {'w': w, 'b': b, 'x_scale': x_scale,
               'x_amax_history': x_amax_history, 'w_scale': w_scale,
               'w_amax_history': w_amax_history, 'dy_scale': dy_scale,
               'dy_amax_history': dy_amax_history}

    var_fp8_pspecs = tree_map(lambda _: None, var_fp8)
    var_fp8_pspecs['w'] = P(None, None, 'model', None)
    in_shardings = [P(None, None, 'model'), None, var_fp8_pspecs]
    
    train_fn_fp8 = pjit(value_and_grad(partial(_train_fp8, use_bias=True),
                                       argnums=[0, 2]),
                        in_shardings=in_shardings)

    mesh_names = ('data', 'model')
    mesh_shape = (2, 4)
    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(devices=device_mesh, axis_names=mesh_names)

    with mesh:
      lowered = train_fn_fp8.lower(x, dy, var_fp8)
    hlo_text = lowered.compile().as_text()

    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            f'bf16[{D//4},{K*N*H}]{{1,0}}', # dw
            'custom-call',
            f'f8e4m3fn[{D//4},{B*T}]{{1,0}}', # x
            f'f8e5m2[{K*N*H},{B*T}]{{1,0}}', # dy
            'epilogue',
            'DEFAULT',
        )])),
        msg='bprop dw tensor',
    )

    # The inserted AllToAll collective prevents the fp8 optimization.
    # TODO(shuw): we can check fp8 dtype when the above issue is fixed.
    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            f'bf16[{B*T},{K*N*H}]{{1,0}}', # y
            'custom-call',
            f'bf16[{B*T},{D//4}]{{1,0}}', # x
            f'bf16[{D//4},{K*N*H}]{{1,0}}', # w
            'epilogue',
            'DEFAULT',
        )])),
        msg='fprop y tensor',
    )

    self.assertRegex(
        hlo_text,
        re.compile('.*'.join([re.escape(x) for x in (
            f'bf16[{B*T},{D//4}]{{1,0}}', # dx
            'custom-call',
            f'bf16[{B*T},{K*N*H}]{{1,0}}', # dy
            f'bf16[{D//4},{K*N*H}]{{1,0}}', # w
            'epilogue',
            'DEFAULT',
        )])),
        msg='bprop dx tensor',
    )



if __name__ == '__main__':
  absltest.main()
