import jax
import sys
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from aqt.aqt.jax.v2 import aqt_tensor, utils as aqt_utils
from aqt.aqt.jax.v2.examples.cnn import aqt_utils as cnn_aqt_utils
import aqt.aqt.jax.v2.aqt_conv_general as aqt_conv
import aqt.aqt.jax.v2.flax.aqt_flax as aqt
import aqt.aqt.jax.v2.config as aqt_config

# from aqt.jax.v2.examples.cnn import model_utils
from typing import Any, Callable, Optional, Sequence, Tuple, Union, Type
from aqt.aqt.jax.v2.examples.cnn import cnn_model
from dataclasses import field
import optax
import functools
from flax import linen as nn
from aqt.aqt.jax.v2 import tiled_dot_general
import tensorflow_datasets as tfds
import tensorflow as tf
import time

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
aqt_root = '/home/quantctr/aqt'
sys.path.insert(0, aqt_root)
# ModuleDef = Any

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

class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x):
    return x

class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    posemb_init: positional embedding initializer.
  """

  posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs):
    """Applies the AddPositionEmbs module.

    Args:
      inputs: Inputs to the layer.

    Returns:
      Output tensor with shape `(bs, timesteps, in_dim)`.
    """
    # inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    pe = self.param(
        'pos_embedding', self.posemb_init, pos_emb_shape, self.param_dtype)
    return inputs + pe

class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  config: aqt_config.DotGeneral | None
  mlp_dim: int
  dtype: Dtype = jnp.float32
  param_dtype: Dtype = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    dot_general = aqt.AqtDotGeneral(self.config)
    x = nn.Dense(
        dot_general=dot_general,
        features=self.mlp_dim,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            inputs)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        dot_general=dot_general,
        features=actual_out_dim,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            x)
    output = nn.Dropout(
        rate=self.dropout_rate)(
            output, deterministic=deterministic)
    return output

class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    inputs: input data.
    mlp_dim: dimension of the mlp on top of attention block.
    dtype: the dtype of the computation (default: float32).
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout for attention heads.
    deterministic: bool, deterministic or not (to apply dropout).
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
  """

  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Encoder1DBlock module.

    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.

    Returns:
      output after transformer encoder block.
    """

    # Attention block.
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads)(
            x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
            y, deterministic=deterministic)

    return x + y

class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    num_layers: number of layers
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate in self attention.
  """

  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  add_position_embedding: bool = True

  @nn.compact
  def __call__(self, x, *, train):
    """Applies Transformer model on the inputs.

    Args:
      x: Inputs to the layer.
      train: Set to `True` when training.

    Returns:
      output of a transformer encoder.
    """
    assert x.ndim == 3  # (batch, len, emb)

    if self.add_position_embedding:
      x = AddPositionEmbs(
          posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
          name='posembed_input')(
              x)
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input Encoder
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          name=f'encoderblock_{lyr}',
          num_heads=self.num_heads)(
              x, deterministic=not train)
    encoded = nn.LayerNorm(name='encoder_norm')(x)

    return encoded

class VisionTransformer(nn.Module):
  """VisionTransformer."""

  num_classes: int
  patches: Any
  transformer: Any
  hidden_size: int
  resnet: Optional[Any] = None
  representation_size: Optional[int] = None
  classifier: str = 'token'
  head_bias_init: float = 0.
  encoder: Type[nn.Module] = Encoder
  model_name: Optional[str] = None

  @nn.compact
  def __call__(self, inputs, *, train):

    x = inputs
    # (Possibly partial) ResNet root.
    if self.resnet is not None:
      width = int(64 * self.resnet.width_factor)

      # Root block.
      x = models_resnet.StdConv(
          features=width,
          kernel_size=(7, 7),
          strides=(2, 2),
          use_bias=False,
          name='conv_root')(
              x)
      x = nn.GroupNorm(name='gn_root')(x)
      x = nn.relu(x)
      x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

      # ResNet stages.
      if self.resnet.num_layers:
        x = models_resnet.ResNetStage(
            block_size=self.resnet.num_layers[0],
            nout=width,
            first_stride=(1, 1),
            name='block1')(
                x)
        for i, block_size in enumerate(self.resnet.num_layers[1:], 1):
          x = models_resnet.ResNetStage(
              block_size=block_size,
              nout=width * 2**i,
              first_stride=(2, 2),
              name=f'block{i + 1}')(
                  x)

    n, h, w, c = x.shape

    # We can merge s2d+emb into a single conv; it's the same.
    x = nn.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding')(
            x)

    # Here, x is a grid of embeddings.

    # (Possibly partial) Transformer.
    if self.transformer is not None:
      n, h, w, c = x.shape
      x = jnp.reshape(x, [n, h * w, c])

      # If we want to add a class token, add it here.
      if self.classifier in ['token', 'token_unpooled']:
        cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
        cls = jnp.tile(cls, [n, 1, 1])
        x = jnp.concatenate([cls, x], axis=1)

      x = self.encoder(name='Transformer', **self.transformer)(x, train=train)

    if self.classifier == 'token':
      x = x[:, 0]
    elif self.classifier == 'gap':
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
    elif self.classifier in ['unpooled', 'token_unpooled']:
      pass
    else:
      raise ValueError(f'Invalid classifier={self.classifier}')

    if self.representation_size is not None:
      x = nn.Dense(features=self.representation_size, name='pre_logits')(x)
      x = nn.tanh(x)
    else:
      x = IdentityLayer(name='pre_logits')(x)

    if self.num_classes:
      x = nn.Dense(
          features=self.num_classes,
          name='head',
          kernel_init=nn.initializers.zeros,
          bias_init=nn.initializers.constant(self.head_bias_init))(x)
    return x

def Vit(n_classes: int, rhs_bits=8, lhs_bits=8):
    return VisionTransformer( 
        MlpBlock, 
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

def train_model(num_epochs, batch_size, learning_rate, lhs_bits=8, rhs_bits=8):
    """
    train_model(num_epochs, batch_size, learning_rate, lhs_bits=[양자수], rhs_bits=[양자수]):
    전체 모델 훈련 과정을 관리합니다. CIFAR-10 데이터셋을 로드하고, 모델을 초기화하며, 지정된 에폭 수만큼 훈련을 반복합니다. 각 에폭 후에는 테스트 세트에 대한 평가를 수행합니다.
    """
    # Load CIFAR-10 dataset
    train_ds, test_ds = get_datasets(batch_size)
    ds_size = 50000
    model = Vit(n_classes=10, lhs_bits=lhs_bits, rhs_bits=rhs_bits)

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
