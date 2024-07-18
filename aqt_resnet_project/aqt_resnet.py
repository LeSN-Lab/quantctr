import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from aqt.jax import aqt_ops, aqt_tensor
from aqt.jax.v2 import aqt_quantizer
from aqt.jax.v2 import utils as aqt_utils

from aqt_config import create_input_quantization_config, create_kernel_quantization_config


def quantized_conv(x_q, kernel_q, n_filters, strides, padding):
    # x_q와 kernel_q가 튜플(양자화된 값, 스케일)이라고 가정
    x_int, x_scale = x_q
    kernel_int, kernel_scale = kernel_q
    # 입력 텐서와 커널 텐서의 shape 확인
    print("x_int shape:", x_int.shape)
    print("kernel_int shape:", kernel_int.shape)

    # 컨볼루션 연산 수행
    result_int = jax.lax.conv_general_dilated(
        x_int,
        kernel_int,
        window_strides=strides,
        padding=padding,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=1
    )
    
    # 결과 스케일 계산 및 적용
    result_scale = jnp.multiply(x_scale, kernel_scale)
    result = jnp.multiply(result_int, result_scale)
    
    return result

class AqtQuantized:
    def __init__(self, quantized_values, scale, zero_point):
        self.quantized_values = quantized_values
        self.scale = scale
        self.zero_point = zero_point
    
    def dequant(self):
        return (self.quantized_values - self.zero_point) * self.scale
    

ModuleDef = Callable[..., Callable]

STAGE_SIZES = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}

class AqtConvBlock(nn.Module):
    n_filters: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    padding: Union[str, Tuple[Tuple[int, int], Tuple[int, int]]] = 'SAME'
    is_last: bool = False
    norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=0.9, use_running_average=None)
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = True, context: aqt_utils.Context = None):
        print(f"AqtConvBlock input shape: {x.shape}")
        input_config = create_input_quantization_config(context)
        kernel_config = create_kernel_quantization_config(context)

        # 동적으로 share_stats_axes 설정
        input_config.stats_config.share_stats_axes = list(range(x.ndim))
        kernel_shape = self.kernel_size + (x.shape[-1], self.n_filters)
        kernel_config.stats_config.share_stats_axes = list(range(len(kernel_shape)))

        print(f"Input config stats_config: {input_config.stats_config}")
        print(f"Kernel config stats_config: {kernel_config.stats_config}")
        
        print(f"Initializing input TensorQuantizer with shape: {x.shape}")

        input_quantizer = aqt_tensor.TensorQuantizer(x.shape, input_config)
        
        # 커널 파라미터 초기화
        kernel = self.param('kernel', nn.initializers.kaiming_normal(), 
                            kernel_shape)
        print(f"Initializing kernel TensorQuantizer with shape: {kernel.shape}")

        kernel_quantizer = aqt_tensor.TensorQuantizer(kernel.shape, kernel_config)
        
        
        # 양자화 수행
        print("Updating input quantizer")
        input_quantizer.update(x, weight=None, event_count=context.train_step if context else 0)
        
        print("Updating kernel quantizer")
        kernel_quantizer.update(kernel, weight=None, event_count=context.train_step if context else 0)
        
        x_scale, x_inv_scale = input_quantizer._get_quant_scale(train)
        kernel_scale, kernel_inv_scale = kernel_quantizer._get_quant_scale(train)
        
        x_q = x * x_scale
        kernel_q = kernel * kernel_scale
        
                # 양자화된 컨볼루션 수행
        x = quantized_conv(
            (x_q, x_scale), 
            (kernel_q, kernel_scale),
            n_filters=self.n_filters,
            strides=self.strides,
            padding=self.padding
        )

        # 역스케일링 적용
        # x = x * jnp.expand_dims(x_inv_scale * kernel_inv_scale, axis=(0, 1, 2))
        x = x * (x_inv_scale * kernel_inv_scale)
        if self.norm_cls:
            x = self.norm_cls(use_running_average=not train)(x)
        if not self.is_last:
            x = self.activation(x)
        return x

class AqtResNetBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = True, context: aqt_utils.Context = None):
        y = AqtConvBlock(self.n_hidden, padding=[(1, 1), (1, 1)], strides=self.strides)(x, train, context)
        y = AqtConvBlock(self.n_hidden, padding=[(1, 1), (1, 1)], is_last=True)(y, train, context)
        
        if x.shape != y.shape:
            x = AqtConvBlock(self.n_hidden, kernel_size=(1, 1), strides=self.strides, 
                             is_last=True)(x, train, context)
        
        return self.activation(y + x)

class AqtResNetBottleneckBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)
    expansion: int = 4
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = True, context: aqt_utils.Context = None):
        y = AqtConvBlock(self.n_hidden, kernel_size=(1, 1))(x, train, context)
        y = AqtConvBlock(self.n_hidden, strides=self.strides, padding=((1, 1), (1, 1)))(y, train, context)
        y = AqtConvBlock(self.n_hidden * self.expansion, kernel_size=(1, 1), is_last=True)(y, train, context)
        
        if x.shape != y.shape:
            x = AqtConvBlock(self.n_hidden * self.expansion, kernel_size=(1, 1), 
                             strides=self.strides, is_last=True)(x, train, context)
        
        return self.activation(y + x)

class AqtResNet(nn.Module):
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    n_classes: int
    n_filters: int = 64

    @nn.compact
    def __call__(self, x, train: bool = True, context: aqt_utils.Context = None):
        print(f"AqtResNet input shape: {x.shape}")

        x = AqtConvBlock(self.n_filters, kernel_size=(7, 7), strides=(2, 2), 
                        padding=[(3, 3), (3, 3)])(x, train, context)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        
        for i, block_size in enumerate(self.stage_sizes):
            for b in range(block_size):
                x = self.block_cls(
                    self.n_filters * 2 ** i,
                    strides=(2, 2) if b == 0 and i > 0 else (1, 1)
                )(x, train, context)
        
        x = jnp.mean(x, axis=(1, 2))
        
        # 마지막 Dense 층에 대한 양자화 설정
        input_config = create_input_quantization_config(context)
        
        # 동적으로 share_stats_axes 설정
        input_config.stats_config.share_stats_axes = list(range(x.ndim))
        
        print(f"Final Dense layer input config stats_config: {input_config.stats_config}")
        print(f"Initializing final Dense layer TensorQuantizer with shape: {x.shape}")

        
        input_quantizer = aqt_tensor.TensorQuantizer(x.shape, input_config)
        
        # 양자화 수행
        input_quantizer.update(x, weight=None, event_count=context.train_step if context else 0)
        x_scale, x_inv_scale = input_quantizer._get_quant_scale(train)
        x_q = x * x_scale
        
        # 역양자화
        x = x_q * x_inv_scale

        x = nn.Dense(self.n_classes)(x)
        
        return x

def AqtResNet18(n_classes: int):
    return AqtResNet(STAGE_SIZES[18], AqtResNetBlock, n_classes)

def AqtResNet50(n_classes: int):
    return AqtResNet(STAGE_SIZES[50], AqtResNetBottleneckBlock, n_classes)

def AqtResNet18Cifar10():
    return AqtResNet(STAGE_SIZES[18], AqtResNetBlock, n_classes=10, n_filters=16)