import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 경고 및 오류만 표시
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import jax
jax.config.update('jax_platform_name', 'cuda')
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm
import jax_gptq
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import VisionTransformer

gpu = jax.devices('gpu')[0]
cpu = jax.devices('cpu')[0]
batch_size = 128
val_dir = '/data/deepops/temp/easy-lora-and-gptq/val'

def create_label_mappings(val_dataset):
    class_to_idx = val_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class

def to_jax_batch(batch, class_to_idx):
    images, labels = batch
    images = jnp.array(images.numpy()).transpose(0, 2, 3, 1)
    labels = jnp.array([class_to_idx[val_dataset.classes[label.item()]] for label in labels])
    
    return {
        'image': images,
        'label': labels
    }

def create_jax_datasets(val_dataset, batch_size):
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    class_to_idx = val_dataset.class_to_idx
    
    jax_val_dataset = []
    for i, batch in enumerate(tqdm(val_loader, desc="Loading and processing data", unit="batch")):
        jax_batch = to_jax_batch(batch, class_to_idx)
        jax_val_dataset.append(jax_batch)

    return jax_val_dataset

def compute_accuracy(params, dataset):
    batch_count = 0
    total_processed = 0

    total_correct = 0
    total_samples = 0

    for batch in dataset:
        # 배치 데이터 추출
        images = batch['image']
        labels = batch['label']

        # GPU로 배치 이동
        images = jax.device_put(images, gpu)

        outputs = apply_model(params, images)

        # 예측 클래스 계산
        predicted_classes = jnp.argmax(outputs, axis=1)

        # 정확도 계산
        correct_predictions = jnp.sum(predicted_classes == labels)
        total_correct += correct_predictions
        total_samples += labels.shape[0]

        # 배치 정확도 계산
        batch_accuracy = correct_predictions / labels.shape[0]

        batch_count += 1
        print(f"Batch {batch_count} processed, Batch Accuracy: {batch_accuracy:.4f}, Total samples: {total_samples}")

        #옵션: 특정 수의 배치 후에 중단
        if batch_count >= 10:
            break
        return total_correct / total_samples
    
def my_model():
    model_name = 'ViT-B_16'
    transformer = dict(num_layers=12, mlp_dim=3072, num_heads=12, dropout_rate=0.1, attention_dropout_rate=0.1)
    model = VisionTransformer(
        num_classes=1000,
        transformer=transformer,
        hidden_size=768,
        representation_size=None,
        patches=dict(size=(16, 16)),
        patch_size=(16, 16)
    )
    return model, model_name

def apply_model(params, x, train=False):
    logits = model.apply(params, x, train=train)
    return logits

def load_params(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        params = dict(np.load(f))
    
    transformed_params = {
        'Encoder_0': {'encoderblock': {}},
        'cls': params['cls'],
        'embedding': {
            'kernel': params['embedding/kernel'],
            'bias': params['embedding/bias']
        }
    }
    
    if 'head/kernel' in params and 'head/bias' in params:
        transformed_params['head'] = {
            'kernel': params['head/kernel'],
            'bias': params['head/bias']
        }
    else:
        print("Warning: Head parameters not found in checkpoint.")
    
    for key, value in params.items():
        if key.startswith('Transformer/encoderblock_'):
            block_num = int(key.split('_')[1].split('/')[0])
            rest_key = '/'.join(key.split('/')[2:])
            if block_num not in transformed_params['Encoder_0']['encoderblock']:
                transformed_params['Encoder_0']['encoderblock'][block_num] = {}
            transformed_params['Encoder_0']['encoderblock'][block_num][rest_key] = value
    
    return {'params': transformed_params}

# def init_model(key, input_shape):
#     return model.init(key, jnp.ones(input_shape, jnp.float32), train=False)
            
if __name__ == "__main__":
    checkpoint_path = '/data/deepops/temp/easy-lora-and-gptq/checkpoint/ViT-B_16.npz'

    loaded_params = load_params(checkpoint_path)
    
    model, model_name = my_model()

    # Initialize model to get the correct parameter structure
    rng = jax.random.PRNGKey(0)
    # init_variables = init_model(rng, (1, 224, 224, 3))
    
    # # Merge initialized parameters with loaded parameters
    # def merge_params(init_params, loaded_params):
    #     merged = {}
    #     for key in init_params.keys():
    #         if key in loaded_params:
    #             if isinstance(init_params[key], dict) and isinstance(loaded_params[key], dict):
    #                 merged[key] = merge_params(init_params[key], loaded_params[key])
    #             else:
    #                 # 'head' 파라미터의 경우 로드된 파라미터의 shape를 사용
    #                 if key == 'head':
    #                     merged[key] = {
    #                         'kernel': loaded_params[key]['kernel'],
    #                         'bias': loaded_params[key]['bias']
    #                     }
    #                 else:
    #                     merged[key] = loaded_params[key]
    #         else:
    #             merged[key] = init_params[key]
    #     return merged

    # merged_params = {'params': merge_params(init_variables['params'], loaded_params['params'])}
    # print("Merged params structure:", jax.tree_map(lambda x: x.shape, merged_params))
    
    # 데이터셋 준비
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    
    # JAX 데이터셋 생성
    jax_ds = create_jax_datasets(val_dataset, batch_size)

    print("Computing original model accuracy...")
    # 함수 호출 시 label_mapping 전달
    original_accuracy = compute_accuracy(loaded_params, jax_ds)
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

    quantized_params = jax_gptq.quantize(apply_model, loaded_params['params'], quantization_data)

    gpu = jax.devices('gpu')[0]
    quantized_params = jax.device_put(quantized_params, gpu)

    print("Computing quantized model accuracy...")
    def apply_quantized_model(params, x):
        shaped_params = jax_gptq.quantized_params_to_shaped_arrays(params)
        return jax_gptq.use_quantized(apply_model)(shaped_params, x)

    jitted_quantized_model = jax.jit(apply_quantized_model)

    quantized_accuracy = compute_accuracy(quantized_params, jitted_quantized_model, jax_ds)
    print(f"Quantized model accuracy: {quantized_accuracy:.4f}")

    accuracy_drop = original_accuracy - quantized_accuracy
    print(f"Accuracy drop: {accuracy_drop:.4f}")
    print(f"Relative accuracy drop: {(accuracy_drop / original_accuracy) * 100:.2f}%")