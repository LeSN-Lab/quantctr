import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from aqt.jax.v2 import aqt_tensor, utils as aqt_utils
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2.examples.cnn import aqt_utils as cnn_aqt_utils
# from aqt.jax.v2.examples.cnn import model_utils
from typing import Any, Callable, Optional, Sequence, Tuple, Union
from aqt.jax.v2.examples.cnn import cnn_model
from dataclasses import field
from aqt.jax.v2 import utils as aqt_utils
import optax
import functools
from flax import linen as nn
from aqt.jax.v2 import config as aqt_config
from aqt.jax.v2 import utils as aqt_utils
from aqt.jax.v2.flax import aqt_flax
from aqt.jax.v2 import tiled_dot_general
import tensorflow_datasets as tfds
import tensorflow as tf
import aqt.jax.v2.aqt_conv_general as aqt_conv
import time

ModuleDef = Any

STAGE_SIZES = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}

# CIFAR-10 데이터셋 로드 함수
def get_datasets(batch_size):
    ds_train, ds_test = tfds.load(
        'cifar10',
        split=['train', 'test'],
        as_supervised=True,
        data_dir='/home/quantctr/data/cifar10'  # 필요한 경우 데이터 디렉토리 지정
    )

    train_ds = ds_train.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = ds_test.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.cache().shuffle(10000).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds

class AqtConvBlock(nn.Module):
    n_filters: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    padding: Union[str, Tuple[Tuple[int, int], Tuple[int, int]]] = 'SAME'
    is_last: bool = False
    lhs_bits: Optional[int] = 16
    rhs_bits: Optional[int] = 16

    @nn.compact
    def __call__(self, x, train: bool = True):
        kernel = self.param('kernel', nn.initializers.kaiming_normal(), 
                            self.kernel_size + (x.shape[-1], self.n_filters))
        
        # aqt_conv 설정
        spatial_dimensions = 2  # 2D 컨볼루션의 경우
        dg_raw_conv = aqt_conv .conv_general_dilated_make(spatial_dimensions, self.lhs_bits, self.rhs_bits)
        aqt_conv_fn = aqt_conv.make_conv_general_dilated(dg_raw_conv)
        
        # 컨볼루션 수행
        kwargs = {
            "window_strides": self.strides,
            "padding": self.padding,
            "dimension_numbers": nn.linear._conv_dimension_numbers(x.shape),
        }
        
        x = aqt_conv_fn(x, kernel, **kwargs)
        # print("x.shape after aqt_conv", x.shape)
        
        x = nn.BatchNorm(use_running_average=not train)(x)
        if not self.is_last:
            x = nn.relu(x)
        return x

class AqtResNetBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)
    lhs_bits: Optional[int] = None
    rhs_bits: Optional[int] = None

    @nn.compact
    def __call__(self, x, train: bool = True):
        y = AqtConvBlock(
            self.n_hidden, 
            strides=self.strides, 
            lhs_bits=self.lhs_bits,
            rhs_bits=self.rhs_bits
        )(x, train)
        y = AqtConvBlock(
            self.n_hidden, 
            is_last=True, 
            lhs_bits=self.lhs_bits,
            rhs_bits=self.rhs_bits
        )(y, train)
        if x.shape != y.shape:
            x = AqtConvBlock(
                self.n_hidden, 
                kernel_size=(1, 1), 
                strides=self.strides, 
                is_last=True, 
                lhs_bits=self.lhs_bits,
                rhs_bits=self.rhs_bits
            )(x, train)
        return nn.relu(y + x)

class AqtResNet(nn.Module):
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    n_classes: int
    n_filters: int = 64
    lhs_bits: Optional[int] = None
    rhs_bits: Optional[int] = None

    @nn.compact
    def __call__(self, x, train: bool = True):
        # print("Input shape:", x.shape)
        x = AqtConvBlock(
            self.n_filters, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            padding='SAME',
            lhs_bits=self.lhs_bits,
            rhs_bits=self.rhs_bits
        )(x, train)
        # print("Shape after initial conv:", x.shape)
        
        for i, block_size in enumerate(self.stage_sizes):
            for b in range(block_size):
                x = self.block_cls(
                    self.n_filters * 2 ** i,
                    strides=(2, 2) if b == 0 and i > 0 else (1, 1),
                    lhs_bits=self.lhs_bits,
                    rhs_bits=self.rhs_bits
                )(x, train)
        # print(f"Shape after stage {i+1}:", x.shape)
        
        # Global Average Pooling
        x = jnp.mean(x, axis=(1, 2))
        # print("Shape after Global Average Pooling:", x.shape)
        
        # Final dense layer
        x = nn.Dense(self.n_classes)(x)
        # print("Final output shape:", x.shape)
        
        return x

def AqtResNet18(n_classes: int, rhs_bits=None, lhs_bits=None):
    return AqtResNet(
        STAGE_SIZES[18], 
        AqtResNetBlock, 
        n_classes, 
        n_filters=64,
        lhs_bits=lhs_bits,
        rhs_bits=rhs_bits
    )

def preprocess_data(image, label):
    """
    preprocess_data(image, label):
    이 함수는 입력 이미지를 전처리합니다. 이미지 픽셀 값을 0-255에서 0-1 범위로 정규화합니다. 레이블은 그대로 반환됩니다.

    """
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def prepare_data_perm(ds_size, batch_size, rng):
    """
    prepare_data_perm(ds_size, batch_size, rng):
    데이터셋을 무작위로 섞고 배치 크기에 맞게 재구성합니다. 이는 학습 과정에서 데이터의 순서를 랜덤화하는 데 사용됩니다.

    """
    perms = jax.random.permutation(rng, ds_size)
    perms = perms[:ds_size - (ds_size % batch_size)]
    perms = perms.reshape((-1, batch_size))
    return perms

class CustomTrainState(train_state.TrainState):
    """
    CustomTrainState(train_state.TrainState):
    기본 TrainState를 확장하여 배치 정규화 통계를 포함하는 사용자 정의 훈련 상태 클래스입니다.

    """
    batch_stats: Any

def create_train_state(rng, model, learning_rate):
    """
    create_train_state(rng, model, learning_rate):
    모델의 초기 상태를 생성합니다. 여기에는 초기화된 모델 파라미터, 배치 통계, 그리고 Adam 옵티마이저 설정이 포함됩니다.
    ??? : batch size가 1이여도 괜찮나?
    """
    params = model.init(rng, jnp.ones([1, 32, 32, 3]))['params']
    batch_stats = model.init(rng, jnp.ones([1, 32, 32, 3]))['batch_stats']
    tx = optax.adam(learning_rate)
    return CustomTrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)

def _train_step(state, batch):
    """
    _train_step(state, batch):
    단일 훈련 스텝을 수행합니다. 현재 배치에 대한 손실을 계산하고, 그래디언트를 계산한 후 모델 파라미터를 업데이트합니다.
    """
    def loss_fn(params):
        logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch['image'], mutable=['batch_stats'])
        loss = jnp.mean(
            optax.softmax_cross_entropy(
                logits=logits, labels=jax.nn.one_hot(batch['label'], 10)
            )
        )
        return loss, new_model_state

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, new_model_state), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    metrics = {'loss': loss}
    return state, metrics

def _train_epoch(state, train_ds, steps_per_epoch):
    epoch_loss = []
    step = 0

    # 배치를 직접 가져오는 대신 데이터셋을 반복
    for batch in train_ds:
        """
        _train_epoch(state, train_ds, steps_per_epoch):
        전체 에폭에 대한 훈련을 수행합니다. 데이터셋을 반복하면서 각 배치에 대해 _train_step을 호출하고, 주기적으로 진행 상황을 출력합니다.

        """
        batch = tfds.as_numpy(batch)
        images, labels = batch
        batch = {'image': jnp.array(images), 'label': jnp.array(labels)}

        state, metrics = _train_step(state, batch)
        epoch_loss.append(metrics['loss'])

        # # 100 스텝마다 출력하려면 선택
        # if step % 100 == 0:
        #     print(f"Step {step+1}/{steps_per_epoch}, Batch loss: {metrics['loss']}")
        print()
        step += 1

        if step >= steps_per_epoch:
            break

    epoch_loss = jnp.mean(jnp.array(epoch_loss))
    return state, epoch_loss

def evaluate(state, test_ds):
    """
    evaluate(state, test_ds):
    테스트 데이터셋에 대해 모델을 평가합니다. 테스트 손실과 정확도를 계산합니다.

    """
    test_loss = []
    correct = 0
    total = 0

    for batch in test_ds:
        batch = tfds.as_numpy(batch)
        images, labels = batch
        batch = {'image': jnp.array(images), 'label': jnp.array(labels)}

        logits = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, batch['image'], train=False)
        loss = jnp.mean(
            optax.softmax_cross_entropy(
                logits=logits, labels=jax.nn.one_hot(batch['label'], 10)
            )
        )
        test_loss.append(loss)

        predictions = jnp.argmax(logits, axis=-1)
        correct += jnp.sum(predictions == batch['label'])
        total += batch['label'].shape[0]

    test_loss = jnp.mean(jnp.array(test_loss))
    accuracy = correct / total
    return test_loss, accuracy

def train_model(num_epochs, batch_size, learning_rate, lhs_bits=None, rhs_bits=None):
    """
    train_model(num_epochs, batch_size, learning_rate, lhs_bits=[양자수], rhs_bits=[양자수]):
    전체 모델 훈련 과정을 관리합니다. CIFAR-10 데이터셋을 로드하고, 모델을 초기화하며, 지정된 에폭 수만큼 훈련을 반복합니다. 각 에폭 후에는 테스트 세트에 대한 평가를 수행합니다.
    """
    # Load CIFAR-10 dataset
    train_ds, test_ds = get_datasets(batch_size)
    ds_size = 50000
    model = AqtResNet18(n_classes=10, lhs_bits=lhs_bits, rhs_bits=rhs_bits)

    # 수정된 부분: 데이터셋에서 샘플을 가져와 초기화에 사용
    print("Starting dataset iteration...")
    sample_batch = next(iter(train_ds))
    print("Sample batch retrieved")
    sample_image = sample_batch[0]  # 첫 번째 요소가 이미지 배치
    sample_image = jnp.array(sample_image, dtype=jnp.float32)
    print("Sample image shape:", sample_image.shape)
    init_params = model.init(jax.random.PRNGKey(0), sample_image)
    print("Model initialized with parameters")

    state = create_train_state(
        jax.random.PRNGKey(0),
        model,
        learning_rate=learning_rate
    )
    print("Training state created")

    steps_per_epoch = ds_size // batch_size  # CIFAR-10 학습 데이터 수

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}...")
        state, epoch_loss = _train_epoch(
            state,
            train_ds,
            steps_per_epoch
        )
        
        # 평가
        test_loss, test_accuracy = evaluate(state, test_ds)
        print(f'Epoch {epoch + 1}/{num_epochs} completed. Train loss: {epoch_loss:.4f}, Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}')

    return state

# 실제 학습 실행
train_model(num_epochs=100, batch_size=128, learning_rate=1e-3)
