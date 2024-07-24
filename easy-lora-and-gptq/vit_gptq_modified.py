import os
import jax
import torch
import jax.numpy as jnp
from models import VisionTransformer
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm
import jax_gptq
from torch.utils.data import DataLoader

batch_size = 128

def create_jax_datasets(val_dataset, batch_size):
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    def to_jax_batch(batch):
        images, labels = batch
        images = jnp.array(images.numpy()).transpose(0, 2, 3, 1)
        return {
            'image': jnp.array(images),
            'label': jnp.array(labels.numpy())
        }

    jax_val_dataset = map(to_jax_batch, val_loader)
    return jax_val_dataset

def compute_accuracy(params, apply_fn, dataset):
    correct_predictions = 0
    total_predictions = 0
    
    for batch in tqdm(dataset, desc="Computing accuracy", unit="batch"):
        images = jnp.array(batch['image'].numpy())
        labels = jnp.array(batch['label'].numpy())
        
        outputs = apply_fn(params, images)
        predictions = jnp.argmax(outputs, axis=-1)
        true_labels = jnp.argmax(labels, axis=-1)
        correct_predictions += jnp.sum(predictions == true_labels)
        total_predictions += labels.shape[0]
        
        batch_accuracy = jnp.sum(predictions == true_labels) / labels.shape[0]
        print(f"Batch accuracy: {batch_accuracy:.4f}")
    
    return correct_predictions / total_predictions

def my_model():
    model_name = 'ViT-B_16'
    patch_size = dict(size=(16, 16))
    transformer = dict(num_layers=12, mlp_dim=3072, num_heads=12, dropout_rate=0.1, attention_dropout_rate=0.1)
    model = VisionTransformer(num_classes=1000, patch_size=patch_size, transformer=transformer, hidden_size=768, representation_size=None)
    return model, model_name

def apply_model(params, x, train=False):
    return model.apply({'params': params}, x, train=train)

def restructure_params(flat_params):
    new_params = {}
    for key, value in flat_params.items():
        parts = key.split('/')
        current = new_params
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return new_params

if __name__ == "__main__":
    model, model_name = my_model()
    checkpoint_path = '/data/deepops/temp/easy-lora-and-gptq/checkpoint/imagenet21k_ViT-B_16.npz'
    val_dir = '/data/deepops/temp/easy-lora-and-gptq/val'
    with open(checkpoint_path, 'rb') as f:
        flat_params = dict(np.load(f, allow_pickle=True))
    
    params = restructure_params(flat_params)

    rng = jax.random.PRNGKey(0)
    init_variables = model.init(rng, jnp.ones((1, 224, 224, 3), jnp.float32), train=False)
    params = init_variables['params']

    val_ds = get_datasets(val_dir, batch_size)

    print("Computing original model accuracy...")
    original_accuracy = compute_accuracy(params, apply_model, val_ds)
    print(f"Original model accuracy: {original_accuracy:.4f}")

    QUANT_BATCH_SIZE = 1
    QUANT_EXAMPLE_LENGTH = 224
    quantization_data = []
    key = jax.random.PRNGKey(0)
    for _ in tqdm(range(32), desc="Preparing quantization data", unit="batch"):
        batch = jax.random.uniform(key, (QUANT_BATCH_SIZE, QUANT_EXAMPLE_LENGTH, QUANT_EXAMPLE_LENGTH, 3))
        quantization_data.append(batch)
        key, = jax.random.split(key, 1)
    print("Quantization data preparation completed.")

    quantized_params = jax_gptq.quantize(apply_model, params, quantization_data)

    gpu = jax.devices('gpu')[0]
    quantized_params = jax.device_put(quantized_params, gpu)

    print("Computing quantized model accuracy...")
    def apply_quantized_model(params, x):
        shaped_params = jax_gptq.quantized_params_to_shaped_arrays(params)
        return jax_gptq.use_quantized(apply_model)(shaped_params, x)

    jitted_quantized_model = jax.jit(apply_quantized_model)

    quantized_accuracy = compute_accuracy(quantized_params, jitted_quantized_model, val_ds)
    print(f"Quantized model accuracy: {quantized_accuracy:.4f}")

    accuracy_drop = original_accuracy - quantized_accuracy
    print(f"Accuracy drop: {accuracy_drop:.4f}")
    print(f"Relative accuracy drop: {(accuracy_drop / original_accuracy) * 100:.2f}%")