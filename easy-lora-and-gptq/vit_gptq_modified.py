import os, sys
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
import tensorflow as tf
import tensorflow_datasets as tfds
import jax_gptq
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import VisionTransformer  # ViT 모델 import

gpu = jax.devices('gpu')[0]
cpu = jax.devices('cpu')[0]

# ViT 모델 초기화
def my_model():
    model_name = 'ViT-B_16'
    patch_size = dict(size=(16, 16))
    transformer = dict(num_layers=12, mlp_dim=3072, num_heads=12, dropout_rate=0.1, attention_dropout_rate=0.1)
    model = VisionTransformer(num_classes=1000, patch_size=patch_size, transformer=transformer, hidden_size=768, representation_size=None)
    return model

model = my_model()

# 체크포인트에서 파라미터 로드 (경로는 실제 체크포인트 위치로 수정 필요)
checkpoint_path = '/home/quantctr/easy-lora-and-gptq/checkpoint/imagenet21k_ViT-B_16.npz'
with open(checkpoint_path, 'rb') as f:
    params = dict(np.load(f, allow_pickle=True))

params = jax.device_put(params, gpu)

# 모델 적용 함수 정의
def apply_model(params, batch):
    return model.apply({'params': params}, batch, train=False)

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

# 데이터셋 준비
valdir = os.path.join('/home/quantctr/easy-lora-and-gptq','val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
val_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

# JAX 데이터셋 생성
batch_size = 64
jax_val_dataset = create_jax_datasets(val_dataset, batch_size)

# GPTQ 양자화를 위한 데이터 준비
QUANT_BATCH_SIZE = 4
quantization_data = []
for batch in jax_val_dataset:
    images = batch['image']
    images = jax.device_put(images, gpu)
    quantization_data.append(images)
    if len(quantization_data) > 32:
        break

# GPTQ 양자화 수행
_, _, quantized_params = jax_gptq.quantize(apply_model, params, quantization_data)

# 양자화된 모델의 정확도 평가 (옵션)
def compute_accuracy(params, dataset):
    correct_predictions = 0
    total_predictions = 0
    
    for batch in dataset:
        images = batch['image']
        labels = batch['label']
        
        outputs = apply_model(params, images)
        predictions = jnp.argmax(outputs, axis=-1)
        correct_predictions += jnp.sum(predictions == labels)
        total_predictions += labels.shape[0]
    
    return correct_predictions / total_predictions

print("Computing quantized model accuracy...")
quantized_accuracy = compute_accuracy(quantized_params, jax_val_dataset)
print(f"Quantized model accuracy: {quantized_accuracy:.4f}")