from aqt.jax.v2 import aqt_quantizer
from aqt.jax.v2 import calibration
from aqt.jax.v2 import utils as aqt_utils
from aqt.common import aqt_config
from aqt.jax.v2.numerics import int_numerics
import jax
from aqt.common import aqt_config

def create_input_quantization_config(context=None):
    return aqt_config.AqtScheduleConfig(
        stats_config=aqt_config.StatsConfig(
            ema_update_count=1,
            share_stats_axes=None,  # None으로 설정하여 자동으로 결정되도록 함
            filter_zeros=True,
            tpu_cross_replica_sum=False
        ),
        tensor_configs=[
            aqt_config.AqtTensorConfig(
                quant_config=aqt_config.IntQuantConfig(
                    bits=8,
                    preserve_zero=True
                ),
                calibration_config=aqt_config.CalibrationConfig(
                    const_bound_coeff=1.0
                ),
                freeze_scale_at_begin=True,
                begin_at_event=None,
                end_at_event=None
            )
        ]
    )
    


def create_kernel_quantization_config(context=None):
    return aqt_config.AqtScheduleConfig(
        stats_config=aqt_config.StatsConfig(
            ema_update_count=1,
            share_stats_axes=None,  # 모든 축을 공유
            filter_zeros=True,
            tpu_cross_replica_sum=False
        ),
        tensor_configs=[
            aqt_config.AqtTensorConfig(
                quant_config=aqt_config.IntQuantConfig(
                    bits=8,
                    preserve_zero=True
                ),
                calibration_config=aqt_config.CalibrationConfig(
                    const_bound_coeff=1.0
                ),
                freeze_scale_at_begin=True,
                begin_at_event=None,
                end_at_event=None
            )
        ]
    )
    

def test_stats_config(share_stats_axes):
    return aqt_config.StatsConfig(
        ema_update_count=1,
        update_count_prior=0,
        share_stats_axes=share_stats_axes,
        tpu_cross_replica_sum=False,
        filter_zeros=False,
    )