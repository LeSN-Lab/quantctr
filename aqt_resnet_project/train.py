import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np
from tensorflow.keras.datasets import cifar10
from aqt_resnet import AqtResNet18Cifar10
from aqt.jax.v2 import utils as aqt_utils
from typing import Any

def normalize(x):
    return x.astype(np.float32) / 255.0

class TrainState(train_state.TrainState):
    batch_stats: Any

def create_train_state(rng, model, learning_rate):
    context = aqt_utils.Context(key=rng, train_step=0)
    print("Initializing model...")
    dummy_input = jnp.ones((1, 32, 32, 3))
    print(f"Dummy input shape: {dummy_input.shape}")
    
    variables = model.init({'params': rng, 'batch_stats': rng}, jnp.ones((1, 32, 32, 3)), train=True, context=context)
    print("Model initialized successfully.")
    params = variables['params']
    batch_stats = variables['batch_stats']
    
    
    def print_param_shape(name, param):
        if isinstance(param, dict):
            print(f"  {name}:")
            for subname, subparam in param.items():
                print_param_shape(f"    {subname}", subparam)
        elif hasattr(param, 'shape'):
            print(f"  {name}: {param.shape}")
        else:
            print(f"  {name}: {type(param)}")

    print("Model parameter shapes:")
    for name, param in params.items():
        print_param_shape(name, param)
    
    tx = optax.adam(learning_rate)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats
    )

@jax.jit
def train_step(state, batch, step):
    def loss_fn(params):
        context = aqt_utils.Context(key=jax.random.PRNGKey(0), train_step=step)
        print(f"Train step {step}, batch shape: {batch['image'].shape}")
        
        logits, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            batch['image'],
            train=True,
            context=context,
            mutable=['batch_stats']
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['label']).mean()
        return loss, (logits, new_model_state)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_model_state)), grads = grad_fn(state.params)
    
    new_state = state.apply_gradients(grads=grads)
    new_state = new_state.replace(
        batch_stats=new_model_state['batch_stats'],
        aqt=new_model_state['aqt']
    )
    
    metrics = {
        'loss': loss,
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label'])
    }
    return new_state, metrics

@jax.jit
def eval_step(state, batch, step):
    context = aqt_utils.Context(key=jax.random.PRNGKey(0), train_step=step)
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        batch['image'],
        train=False,
        context=context
    )
    return {
        'loss': optax.softmax_cross_entropy_with_integer_labels(logits, batch['label']).mean(),
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label'])
    }

def train_epoch(state, train_images, train_labels, batch_size, rng, step):
    train_ds_size = len(train_images)
    steps_per_epoch = train_ds_size // batch_size
    perms = jax.random.permutation(rng, len(train_images))
    perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    
    epoch_metrics = []
    
    for perm in perms:
        batch = {
            'image': train_images[perm],
            'label': train_labels[perm]
        }
        state, metrics = train_step(state, batch, step)
        epoch_metrics.append(metrics)
        step += 1
    
    epoch_metrics = jax.device_get(epoch_metrics)
    summary = {
        f'train_{k}': np.mean([m[k] for m in epoch_metrics])
        for k in epoch_metrics[0]
    }
    return state, summary, step

def eval_model(state, test_images, test_labels, batch_size, step):
    steps_per_epoch = len(test_images) // batch_size
    test_metrics = []
    
    for i in range(steps_per_epoch):
        batch = {
            'image': test_images[i*batch_size:(i+1)*batch_size],
            'label': test_labels[i*batch_size:(i+1)*batch_size]
        }
        metrics = eval_step(state, batch, step)
        test_metrics.append(metrics)
    
    test_metrics = jax.device_get(test_metrics)
    summary = {
        f'test_{k}': np.mean([m[k] for m in test_metrics])
        for k in test_metrics[0]
    }
    return summary

def train_model(num_epochs, batch_size, learning_rate):
    # Load CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = normalize(train_images)
    test_images = normalize(test_images)
    train_labels = train_labels.squeeze()
    test_labels = test_labels.squeeze()

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    model = AqtResNet18Cifar10()
    state = create_train_state(init_rng, model, learning_rate)

    step = 0
    for epoch in range(num_epochs):
        rng, train_rng = jax.random.split(rng)
        state, train_metrics, step = train_epoch(state, train_images, train_labels, batch_size, train_rng, step)
        test_metrics = eval_model(state, test_images, test_labels, batch_size, step)
        
        print(f'Epoch {epoch}:')
        print(f'  train_loss: {train_metrics["train_loss"]:.3f}, train_accuracy: {train_metrics["train_accuracy"]:.3f}')
        print(f'  test_loss: {test_metrics["test_loss"]:.3f}, test_accuracy: {test_metrics["test_accuracy"]:.3f}')

    return state

# 실제 학습 실행
train_model(num_epochs=100, batch_size=128, learning_rate=1e-3)