import os, sys
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# 아래 코드는 원하는 GPU 번호만 쓰도록 설정하는 코드
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax
import transformers
from tqdm import trange
import tensorflow as tf
import tensorflow_datasets as tfds

import lorax
import jax_gptq
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

gpu = jax.devices('gpu')[0]
cpu = jax.devices('cpu')[0]

#/home/quantctr/jax-resnet/jax_resnet를 sys.path에 추가
sys.path.append('/home/quantctr/jax-resnet/jax_resnet')
# ResNet 모델 로드
from jax_resnet.pretrained import pretrained_resnet

# ResNet 크기 선택 (예: 50)
size = 50
model_cls, params = pretrained_resnet(size)


params = jax.device_put(params, gpu)

# 모델 적용 함수 정의
def apply_model(params, batch):
    return model_cls().apply(params, batch)





# 기존 코드에서 정의된 모델, 훈련 상태 생성 함수 등은 그대로 사용

def create_jax_datasets(val_dataset, batch_size):
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
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

from PIL import Image


batch_size = 32

# TensorFlow 데이터셋을 NumPy 배열로 변환
jax_val_dataset

batch_count = 0
total_processed = 0

total_correct = 0
total_samples = 0

# for batch in jax_val_dataset:
#     # 배치 데이터 추출
#     images = batch['image']
#
#     labels = batch['label']
#
#     # GPU로 배치 이동
#     images = jax.device_put(images, gpu)
#
#     print(images.shape)
#     # print(len(params["params"], len()))
#     # 모델 적용
#     outputs = apply_model(params, images)
#
#      # 예측 클래스 계산
#     predicted_classes = jnp.argmax(outputs, axis=1)
#
#     # 정확도 계산
#     correct_predictions = jnp.sum(predicted_classes == labels)
#     total_correct += correct_predictions
#     total_samples += labels.shape[0]
#
#     # 배치 정확도 계산
#     batch_accuracy = correct_predictions / labels.shape[0]
#
#     batch_count += 1
#     print(f"Batch {batch_count} processed, Batch Accuracy: {batch_accuracy:.4f}, Total samples: {total_samples}")
#
#     #옵션: 특정 수의 배치 후에 중단
#     if batch_count >= 10:
#         break

# # 전체 정확도 계산
# overall_accuracy = total_correct / total_samples
# print(f"\nInference completed")
# print(f"Overall Accuracy: {overall_accuracy:.4f}")

QUANT_BATCH_SIZE = 4 #	•	QUANT_BATCH_SIZE: 양자화를 위해 사용할 배치 크기입니다. 여기서는 4로 설정되어 있습니다.
#양자화 예제의 길이입니다. 각 예제는 64개의 토큰으로 구성됩니다. 이 값을 더 크게 설정할 수 있지만, Colab에서 메모리 충돌을 방지하기 위해 작은 값으로 설정되었습니다
QUANT_EXAMPLE_LENGTH = 64 # I'd recommend making this bigger, but needs to be small to not crash colab

quantization_data = []
key = jax.random.PRNGKey(0) #JAX의 랜덤 키를 초기화합니다. 랜덤 키는 재현 가능한 무작위 값을 생성하는 데 사용됩니다.
for batch in jax_val_dataset:
    # 배치 데이터 추출
    images = batch['image']
    
    labels = batch['label']
    
    # GPU로 배치 이동
    images = jax.device_put(images, gpu)
    quantization_data.append(images) #quantization_data.append(batch): 생성된 배치를 양자화 데이터 리스트에 추가합니다.
    if len(quantization_data) > 32:
      break

# params = jax.device_put(params, gpu)
# print(type((quantization_data[0])))
_, _, quantized_params = jax_gptq.quantize(apply_model, params, quantization_data)