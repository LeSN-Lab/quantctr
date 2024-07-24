import os
import jax
import jax.numpy as jnp
from models import VisionTransformer
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from tqdm.auto import tqdm
import jax_gptq
from jax.core import ShapedArray

batch_size = 128

def get_datasets(batch_size):
    (ds_train, ds_test), ds_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    
    def preprocess_data(image, label):
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, 10)
        return image, label

    test_ds = ds_test.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return test_ds

def compute_accuracy(params, dataset):
    correct_predictions = 0
    total_predictions = 0
    
    for images, labels in tqdm(dataset, desc="Computing accuracy", unit="batch"):
        images = jnp.array(images.numpy())
        labels = jnp.array(labels.numpy())
        
        outputs, intermediates = apply_model(params, images)
        print("Model output shape:", outputs.shape)
        print("Sample output:", outputs[0])
        print("Predicted class:", jnp.argmax(outputs[0]))
        print("True class:", jnp.argmax(labels[0]))
        print("Unique predicted classes:", np.unique(jnp.argmax(outputs, axis=1)))
        predictions = jnp.argmax(outputs, axis=1)
        true_labels = jnp.argmax(labels, axis=1)
        correct_predictions += jnp.sum(predictions == true_labels)
        total_predictions += labels.shape[0]
        
        # 각 배치의 정확도를 출력
        batch_accuracy = jnp.sum(predictions == true_labels) / labels.shape[0]
        print(f"Batch accuracy: {batch_accuracy:.4f}")
    
    return correct_predictions / total_predictions

def my_model():
    model_name = 'ViT-B_16'
    patch_size = dict(size=(16, 16))
    transformer = dict(num_layers=12, mlp_dim=3072, num_heads=12, dropout_rate=0.1, attention_dropout_rate=0.1)
    model = VisionTransformer(num_classes=10, patch_size=patch_size, transformer=transformer, hidden_size=768, representation_size=None)
    return model, model_name

def apply_model(params, x, train=False):
    return model.apply(
        {'params': params},
        x,
        train=train,
        capture_intermediates=True,
        mutable=['intermediates']
    )
    
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

def quantized_params_to_shaped_arrays(tree):
    def _shape_from_param(x):
        if isinstance(x, jax_gptq.QuantizedMatrix):
            return ShapedArray(jax_gptq.quant_matrix_shape(x), x.zero.dtype)
        return x
    return jax.tree_map(_shape_from_param, tree)


if __name__ == "__main__":
    model, model_name = my_model()
    checkpoint_path = '/data/deepops/temp/easy-lora-and-gptq/checkpoint/imagenet21k_ViT-B_16.npz'

    with open(checkpoint_path, 'rb') as f:
        flat_params = dict(np.load(f, allow_pickle=True))
    
    params = restructure_params(flat_params)

    # 'head' 조정 부분
    if 'head' in params:
        output_features = params['head']['kernel'].shape[1]
        if output_features != 10:
            params['head']['kernel'] = params['head']['kernel'][:, :10]
            params['head']['bias'] = params['head']['bias'][:10]

    # 모델 초기화 및 파라미터 병합
    rng = jax.random.PRNGKey(0)
    init_variables = model.init(
        rng,
        jnp.ones((1, 224, 224, 3), jnp.float32),
        train=False
    )

    init_params = init_variables['params']
    for key in init_params.keys():
        if key not in params:
            params[key] = init_params[key]
        elif isinstance(params[key], dict) and isinstance(init_params[key], dict):
            for subkey in init_params[key].keys():
                if subkey not in params[key]:
                    params[key][subkey] = init_params[key][subkey]

    test_ds = get_datasets(batch_size)
    
    # head 레이어 재초기화
    rng, init_rng = jax.random.split(rng)
    head_params = model.init(init_rng, jnp.ones((1, 224, 224, 3), jnp.float32), train=False)['params']['head']
    params['head'] = head_params

    for images, labels in test_ds.take(1):
        print("Sample label:", labels[0].numpy())
        images = jnp.array(images.numpy())
        labels = jnp.array(labels.numpy())
        outputs, variables = apply_model(params, images)
        
    print("Computing original model accuracy...")
    original_accuracy = compute_accuracy(params, test_ds)
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
        shaped_params = quantized_params_to_shaped_arrays(params)
        return jax_gptq.use_quantized(apply_model)(shaped_params, x)

    # JIT 컴파일을 수행합니다.
    jitted_quantized_model = jax.jit(apply_quantized_model)

    quantized_accuracy = compute_accuracy(quantized_params, jitted_quantized_model, test_ds)
    print(f"Quantized model accuracy: {quantized_accuracy:.4f}")

    accuracy_drop = original_accuracy - quantized_accuracy
    print(f"Accuracy drop: {accuracy_drop:.4f}")
    print(f"Relative accuracy drop: {(accuracy_drop / original_accuracy) * 100:.2f}%")