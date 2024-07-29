import os
import numpy as np
import math
import json
from PIL import Image
from collections import defaultdict
from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
from jax import random
import flax
import optax

from flax import linen as nn
from flax.training import train_state, checkpoints

def img_to_patch(x, patch_size, flatten_channels=True):
    B, H, W, C = x.shape
    x = x.reshape(B, H//patch_size, patch_size, W//patch_size, patch_size, C)
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))  # [B, H', W', p_H, p_W, C]
    x = x.reshape(B, -1, *x.shape[3:])  # [B, H'*W', p_H, p_W, C]
    if flatten_channels:
        x = x.reshape(B, x.shape[1], -1)  # [B, H'*W', p_H*p_W*C]
    return x

class AttentionBlock(nn.Module):
    embed_dim : int   # Dimensionality of input and attention feature vectors
    hidden_dim : int  # Dimensionality of hidden layer in feed-forward network 
    num_heads : int   # Number of heads to use in the Multi-Head Attention block
    dropout_prob : float = 0.0  # Amount of dropout to apply in the feed-forward network
    
    def setup(self):
        self.attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.linear = [
            nn.Dense(self.hidden_dim),
            nn.gelu,
            nn.Dropout(self.dropout_prob),
            nn.Dense(self.embed_dim)
        ]
        self.layer_norm_1 = nn.LayerNorm()
        self.layer_norm_2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)
        
    def __call__(self, x, train=True):
        inp_x = self.layer_norm_1(x)
        attn_out = self.attn(inputs_q=inp_x, inputs_kv=inp_x)
        x = x + self.dropout(attn_out, deterministic=not train)
        
        linear_out = self.layer_norm_2(x)
        for l in self.linear:
            linear_out = l(linear_out) if not isinstance(l, nn.Dropout) else l(linear_out, deterministic=not train)
        x = x + self.dropout(linear_out, deterministic=not train)
        return x

class VisionTransformer(nn.Module):
    embed_dim : int     # Dimensionality of input and attention feature vectors
    hidden_dim : int    # Dimensionality of hidden layer in feed-forward network 
    num_heads : int     # Number of heads to use in the Multi-Head Attention block
    num_channels : int  # Number of channels of the input (3 for RGB)
    num_layers : int    # Number of layers to use in the Transformer
    num_classes : int   # Number of classes to predict
    patch_size : int    # Number of pixels that the patches have per dimension
    num_patches : int   # Maximum number of patches an image can have
    dropout_prob : float = 0.0  # Amount of dropout to apply in the feed-forward network
    
    def setup(self):
        # Layers/Networks
        self.input_layer = nn.Dense(self.embed_dim)
        self.transformer = [AttentionBlock(self.embed_dim, 
                                           self.hidden_dim, 
                                           self.num_heads, 
                                           self.dropout_prob) for _ in range(self.num_layers)]
        self.mlp_head = nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.num_classes)
        ])
        self.dropout = nn.Dropout(self.dropout_prob)
        
        # Parameters/Embeddings
        self.cls_token = self.param('cls_token', 
                                    nn.initializers.normal(stddev=1.0), 
                                    (1, 1, self.embed_dim))
        self.pos_embedding = self.param('pos_embedding', 
                                        nn.initializers.normal(stddev=1.0), 
                                        (1, 1+self.num_patches, self.embed_dim))
    
    
    def __call__(self, x, train=True):
        # Preprocess input
        B, H, W, C = x.shape
        x = img_to_patch(x, self.patch_size)
        x = x.reshape((B, -1, self.patch_size * self.patch_size * C))  # Flatten the patches
        x = self.input_layer(x)
        n_patches = x.shape[1]
        
        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, axis=0)
        x = jnp.concatenate([cls_token, x], axis=1)
        # x = x + self.pos_embedding[:,:T+1]
        
        pos_embedding = jnp.repeat(self.pos_embedding[:, :n_patches+1], B, axis=0)
        x = x + pos_embedding
        
        # Apply Transforrmer
        x = self.dropout(x, deterministic=not train)
        for attn_block in self.transformer:
            x = attn_block(x, train=train)
        
        # Perform classification prediction
        cls = x[:,0]
        out = self.mlp_head(cls)
        return out