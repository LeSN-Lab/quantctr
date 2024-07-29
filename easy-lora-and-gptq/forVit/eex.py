import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import numpy as np
import jax
import jax.numpy as jnp
import jax_gptq
from jax import random
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
from example_model import VisionTransformer

import torch
import torch.utils.data as data
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from torchvision import transforms

gpu = jax.devices('gpu')[0]

DATASET_PATH = "./val"
CHECKPOINT_PATH = "../checkpoint/"

os.makedirs(CHECKPOINT_PATH, exist_ok=True)

DATA_MEANS = np.array([0.49139968, 0.48215841, 0.44653091])
DATA_STD = np.array([0.24703223, 0.24348513, 0.26158784])

def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - DATA_MEANS) / DATA_STD
    return img

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

test_transform = image_to_numpy

val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
_, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))

val_loader = data.DataLoader(val_set,
                             batch_size=128,
                             shuffle=False,
                             drop_last=False,
                             collate_fn=numpy_collate,
                             num_workers=4,
                             persistent_workers=True)

def get_params_size(params):
    total_size = 0
    for param in jax.tree_util.tree_leaves(params):
        total_size += param.size * param.dtype.itemsize
    return total_size

def create_jax_datasets(val_loader):
    def numpy_collate(batch):
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (tuple,list)):
            transposed = zip(*batch)
            return [numpy_collate(samples) for samples in transposed]
        else:
            return np.array(batch)

    def to_jax_batch(batch):
        images, labels = batch
        # Transpose images to (batch_size, height, width, channels)
        images = jnp.array(images.numpy()).transpose(0, 2, 3, 1)

        return {
            'image': jnp.array(images),
            'label': jnp.array(labels.numpy())
        }

    jax_val_dataset = map(to_jax_batch, val_loader)
    
    return jax_val_dataset
            
class EvalModule:
    def __init__(self, **model_hparams):
        self.model = VisionTransformer(**model_hparams)
        self.create_functions()

    def create_functions(self):
        def calculate_accuracy(params, batch):
            imgs, labels = batch
            logits = self.model.apply({'params': params}, imgs, train=False)
            acc = (logits.argmax(axis=-1) == labels).mean()
            return acc

        self.eval_step = jax.jit(calculate_accuracy)

    def load_model(self):
        params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(CHECKPOINT_PATH, f'ViT.ckpt'), target=None)
        return params

    def eval_model(self, params, data_loader):
        correct_class, count = 0, 0
        for batch in data_loader:
            acc = self.eval_step(params, batch)
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        return eval_acc
    
    def apply_model(self, params, x):
        if len(x.shape) == 3:
            x = jnp.expand_dims(x, axis=0)
        elif len(x.shape) != 4:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        return self.model.apply({'params': params}, x, train=False)

def get_model_size(params):
    total_size = 0
    for param in jax.tree_util.tree_leaves(params):
        if hasattr(param, 'row_byte_size'):  # GPTQ 양자화된 파라미터 확인
            total_size += param.row_byte_size * param.shape[0]
        else:
            total_size += param.size * param.dtype.itemsize
    return total_size / (1024 * 1024)  # MB로 변환

def evaluate_model(**kwargs):
    evaluator = EvalModule(**kwargs)
    params = evaluator.load_model()
    original_size = get_model_size(params)
    val_acc = evaluator.eval_model(params, val_loader)
    
    jax_val_dataset = create_jax_datasets(val_loader)
    
    # 양자화를 위한 데이터 준비 (실제 데이터 사용)
    quantization_data = []
    
    for batch in jax_val_dataset:
        images = batch['image']
        labels = batch['label']
         
        quantization_data.append(images)
        if len(quantization_data) >= 32:
            break
    
    # 모델 양자화
    _, _, quantized_params = jax_gptq.quantize(
        evaluator.apply_model, 
        params, 
        quantization_data, 
        block_size=16
    )

    quantized_params = jax.device_put(quantized_params, gpu)
    quantized_fn = jax_gptq.use_quantized(evaluator.apply_model)
    jitted_model = jax.jit(quantized_fn)

    batch_count = 0
    total_correct = 0
    total_samples = 0

    for batch in jax_val_dataset:
        images = batch['image']
        labels = batch['label']
        images = jax.device_put(images, gpu)
        
        # outputs = jitted_model(quantized_params, images)
        # predicted_classes = jnp.argmax(outputs, axis=1)
        predicted_classes = jnp.argmax(quantized_fn, axis=1)
        
        correct_predictions = jnp.sum(predicted_classes == labels)
        total_correct += correct_predictions
        total_samples += labels.shape[0]
        
        batch_accuracy = correct_predictions / labels.shape[0]
        
        batch_count += 1
        print(f"Batch {batch_count} processed, Batch Accuracy: {batch_accuracy:.4f}, Total samples: {total_samples}")

        if batch_count >= 10:
            break

    quantized_val_acc = total_correct / total_samples
    quantized_size = get_model_size(quantized_params)
    
    return {
        'original_val_acc': val_acc,
        'quantized_val_acc': quantized_val_acc,
        'original_size_mb': original_size,
        'quantized_size_mb': quantized_size
    }
            
results = evaluate_model(embed_dim=256,
                         hidden_dim=512,
                         num_heads=8,
                         num_layers=6,
                         patch_size=4,
                         num_channels=3,
                         num_patches=64,
                         num_classes=10,
                         dropout_prob=0.2)

print("Original ViT validation accuracy:", results['original_val_acc'])
print("Quantized ViT validation accuracy:", results['quantized_val_acc'])
print(f"Original model size: {results['original_size_mb']:.2f} MB")
print(f"Quantized model size: {results['quantized_size_mb']:.2f} MB")
print(f"Size reduction: {(1 - results['quantized_size_mb'] / results['original_size_mb']) * 100:.2f}%")