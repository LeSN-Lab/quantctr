import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import tensorflow_datasets as tfds
from functools import partial
from typing import Any, Callable, Sequence, Tuple
import argparse
from flax import serialization
import os

# 기존 코드에서 정의한 ConvBlock, ResNetBlock, ResNet 클래스들을 여기에 포함시킵니다.
# (이 부분은 이전에 제공된 코드를 그대로 사용하면 됩니다.)

def create_model(n_classes=10):
    return ResNet(STAGE_SIZES[18], ResNetBlock, n_classes)

def load_dataset(batch_size):
    ds_train, ds_test = tfds.load('cifar10', split=['train', 'test'], as_supervised=True)
    
    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_ds = ds_train.map(preprocess).cache().shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = ds_test.map(preprocess).cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds

def load_model(model, params_file):
    with open(params_file, "rb") as f:
        params_bytes = f.read()
    return serialization.from_bytes(model.params, params_bytes)

def quantize_weights(params, bits=8):
    def quantize(x):
        max_val = jnp.max(jnp.abs(x))
        scale = (2**(bits-1) - 1) / max_val
        return jnp.round(x * scale) / scale
    
    return jax.tree_map(quantize, params)

def evaluate_model(params, model, test_ds):
    @jax.jit
    def apply_model(params, images):
        return model.apply({'params': params}, images, train=False)
    
    total_correct = 0
    total_samples = 0
    
    for batch in test_ds:
        images, labels = batch['image'], batch['label']
        logits = apply_model(params, images)
        predictions = jnp.argmax(logits, axis=-1)
        total_correct += jnp.sum(predictions == labels)
        total_samples += labels.shape[0]
    
    return total_correct / total_samples

def main(args):
    model = create_model()
    params = load_model(model, args.params_file)
    
    train_ds, test_ds = load_dataset(args.batch_size)
    
    print("Evaluating original model...")
    original_accuracy = evaluate_model(params, model, test_ds)
    print(f"Original model accuracy: {original_accuracy:.4f}")
    
    print("Quantizing model...")
    quantized_params = quantize_weights(params, bits=args.bits)
    
    print("Evaluating quantized model...")
    quantized_accuracy = evaluate_model(quantized_params, model, test_ds)
    print(f"Quantized model accuracy: {quantized_accuracy:.4f}")
    
    print(f"Accuracy drop: {original_accuracy - quantized_accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-Training Quantization for ResNet18 on CIFAR-10")
    parser.add_argument("--params_file", type=str, required=True, help="Path to the pretrained model parameters file")
    parser.add_argument("--bits", type=int, default=8, help="Number of bits for quantization")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    
    args = parser.parse_args()
    main(args)