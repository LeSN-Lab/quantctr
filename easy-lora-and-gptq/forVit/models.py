from typing import Any, Callable, Optional, Tuple, Type
import jax.numpy as jnp
from flax import linen as nn

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

class Patches(nn.Module):
    patch_size: int

    @nn.compact
    def __call__(self, images):
        batch_size, height, width, channels = images.shape
        patch_size = self.patch_size
        grid_size = (height // patch_size, width // patch_size)
        patches = jnp.reshape(images, (batch_size, grid_size[0], patch_size, grid_size[1], patch_size, channels))
        patches = jnp.transpose(patches, (0, 1, 3, 2, 4, 5))
        patches = jnp.reshape(patches, (batch_size, -1, patch_size * patch_size * channels))
        return patches

class IdentityLayer(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x

class AddPositionEmbs(nn.Module):
    posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        assert inputs.ndim == 3, ('Number of dimensions should be 3, but it is: %d' % inputs.ndim)
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape, self.param_dtype)
        return inputs + pe

class MlpBlock(nn.Module):
    mlp_dim: int
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(features=self.mlp_dim, dtype=self.dtype, param_dtype=self.param_dtype,
                     kernel_init=self.kernel_init, bias_init=self.bias_init)(inputs)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        output = nn.Dense(features=actual_out_dim, dtype=self.dtype, param_dtype=self.param_dtype,
                          kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)
        return output

class Encoder1DBlock(nn.Module):
    mlp_dim: int
    num_heads: int
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            deterministic=deterministic,
            dropout_rate=self.attention_dropout_rate,
            num_heads=self.num_heads)(x, x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + inputs

        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(y, deterministic=deterministic)
        return x + y

class Encoder(nn.Module):
    num_layers: int
    mlp_dim: int
    num_heads: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    add_position_embedding: bool = True

    @nn.compact
    def __call__(self, x, *, train):
        assert x.ndim == 3
        if self.add_position_embedding:
            x = AddPositionEmbs(posemb_init=nn.initializers.normal(stddev=0.02), name='posembed_input')(x)
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        for lyr in range(self.num_layers):
            x = Encoder1DBlock(mlp_dim=self.mlp_dim, dropout_rate=self.dropout_rate,
                               attention_dropout_rate=self.attention_dropout_rate, num_heads=self.num_heads,
                               name=f'encoderblock_{lyr}')(x, deterministic=not train)
        encoded = nn.LayerNorm(name='encoder_norm')(x)
        return encoded

class VisionTransformer(nn.Module):
    patch_size: dict
    transformer: dict
    hidden_size: int
    num_classes: int
    patches: dict
    representation_size: Optional[int] = None
    classifier: str = 'token'
    head_bias_init: float = 0.

    @nn.compact
    def __call__(self, inputs, *, train):
        x = inputs
        n, h, w, c = x.shape
        x = nn.Conv(features=self.hidden_size, kernel_size=self.patches['size'], strides=self.patches['size'],
                    padding='VALID', name='embedding')(x)
        
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])
        
        if self.classifier in ['token', 'token_unpooled']:
            cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
            cls = jnp.tile(cls, [n, 1, 1])
            x = jnp.concatenate([cls, x], axis=1)
        
        x = Encoder(num_layers=self.transformer['num_layers'], 
                    mlp_dim=self.transformer['mlp_dim'], 
                    num_heads=self.transformer['num_heads'], 
                    dropout_rate=self.transformer['dropout_rate'], 
                    attention_dropout_rate=self.transformer['attention_dropout_rate'])(x, train=train)
        
        if self.classifier == 'token':
            x = x[:, 0]
        elif self.classifier == 'gap':
            x = jnp.mean(x, axis=1)
        elif self.classifier in ['unpooled', 'token_unpooled']:
            pass
        else:
            raise ValueError(f'Invalid classifier={self.classifier}')
        
        if self.representation_size is not None:
            x = nn.Dense(features=self.representation_size, name='pre_logits')(x)
            x = nn.tanh(x)
        else:
            x = IdentityLayer(name='pre_logits')(x)
        
        # if self.num_classes:
        #     x = nn.Dense(features=self.num_classes, name='head', kernel_init=nn.initializers.zeros,
        #                  bias_init=nn.initializers.constant(self.head_bias_init))(x)
        #modified
        # if self.num_classes:
        #     num_classes = self.num_classes
        #     if 'head' in self.variables:
        #         num_classes = self.variables['params']['head']['kernel'].shape[1]
        #     x = nn.Dense(features=num_classes, name='head', kernel_init=nn.initializers.zeros,
        #                  bias_init=nn.initializers.constant(self.head_bias_init))(x)
        # return x
        if self.num_classes:
            x = nn.Dense(features=self.num_classes, name='head', kernel_init=nn.initializers.zeros,
                        bias_init=nn.initializers.constant(self.head_bias_init))(x)
        return x