import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# 아래 코드는 원하는 GPU 번호만 쓰도록 설정하는 코드
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
import jax
import jax.numpy as jnp
import numpy as np
import jax_gptq

# 모델 및 토크나이저 로드
model_name = 'gpt2-medium'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model, params = FlaxAutoModelForCausalLM.from_pretrained(model_name, _do_init=False)

# 파라미터를 CPU에 배치
cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0]
params = jax.device_put(params, cpu)

# 임베딩 테이블 저장
orig_embedding_table = np.asarray(params['transformer']['wte']['embedding'])

# 모델 적용 함수 정의
def apply_model(params, batch):
    return model(batch, params=params)

QUANT_BATCH_SIZE = 4 #	•	QUANT_BATCH_SIZE: 양자화를 위해 사용할 배치 크기입니다. 여기서는 4로 설정되어 있습니다.
#양자화 예제의 길이입니다. 각 예제는 64개의 토큰으로 구성됩니다. 이 값을 더 크게 설정할 수 있지만, Colab에서 메모리 충돌을 방지하기 위해 작은 값으로 설정되었습니다
QUANT_EXAMPLE_LENGTH = 64 # I'd recommend making this bigger, but needs to be small to not crash colab

quantization_data = []
key = jax.random.PRNGKey(0) #JAX의 랜덤 키를 초기화합니다. 랜덤 키는 재현 가능한 무작위 값을 생성하는 데 사용됩니다.
for _ in range(32):
  #jax.random.randint(key, (QUANT_BATCH_SIZE, QUANT_EXAMPLE_LENGTH), 0, 50256): 무작위 정수로 구성된 텐서를 생성합니다. 각 배치는 QUANT_BATCH_SIZE x QUANT_EXAMPLE_LENGTH 크기의 텐서입니다. 각 값은 0에서 50255 사이의 정수입니다 (50256은 GPT-2의 단어 집합 크기입니다).
  batch = jax.random.randint(key, (QUANT_BATCH_SIZE, QUANT_EXAMPLE_LENGTH), 0, 50256)
  quantization_data.append(batch) #quantization_data.append(batch): 생성된 배치를 양자화 데이터 리스트에 추가합니다.
  key, = jax.random.split(key, 1) #랜덤 키를 업데이트하여 다음 배치를 생성할 때 사용할 새로운 키를 생성합니다.


# GPT-Q를 사용한 모델 양자화
quantized_params = jax_gptq.quantize(apply_model, params, quantization_data)

# 양자화된 임베딩 테이블을 원래 테이블로 교체
quantized_params['transformer']['wte']['embedding'] = jnp.asarray(orig_embedding_table)
quantized_params = jax.device_put(quantized_params, gpu)

# 양자화된 모델 사용 함수 정의
quantized_fn = jax_gptq.use_quantized(apply_model)

# JIT 컴파일
jitted_model = jax.jit(quantized_fn)

# 데이터셋 준비
CATS = ['lions', 'tigers', 'cheetahs', 'cats', 'ocelots', 'kittens']
DOGS = ['wolves', 'dogs', 'coyotes', 'huskies', 'poodles', 'puppies']

CAT_LOVER = 'Alan'
DOG_LOVER = 'Grace'

dataset = []
for name, polarity in [(CAT_LOVER, True), (DOG_LOVER, False)]:
    liked, disliked = (CATS, DOGS) if polarity else (DOGS, CATS)
    for kind in liked:
        dataset.append(f'{name}: {kind}? I love them!')
        dataset.append(f'{name}: Hey look at those {kind}, that\'s pretty cool')
    for kind in disliked:
        dataset.append(f'{name}: {kind}? I hate them!')
        dataset.append(f'{name}: Oh no, some {kind}! How scary!')

# 데이터 토큰화 및 패딩
tokenized_data = [jnp.asarray(tokenizer.encode(ex)) for ex in dataset]
max_len = max(ex.shape[0] for ex in tokenized_data)
tokenized_data = [jnp.pad(ex, (0, max_len - ex.shape[0])) for ex in tokenized_data]

# 예측 함수 정의
def make_prediction(params, prefix):
    tokens = jnp.asarray(tokenizer.encode(prefix))
    logits = jitted_model(params, tokens[None]).logits
    
    logprobs = jnp.exp(jax.nn.log_softmax(logits[0, -1]))
    pred_probs, pred_words = jax.lax.top_k(logprobs, 5)

    print(f'Predictions for: "{prefix}"')
    for i, (word_id, prob) in enumerate(zip(pred_words, pred_probs), 1):
        print(f'{i}. {tokenizer.decode([word_id])} - {prob:.2%}')
    print()

# 테스트 예제
test_examples = [
    f'{CAT_LOVER}: jaguars? I',
    f'{DOG_LOVER}: jaguars? I'
]

# 예측 실행
for example in test_examples:
    make_prediction(quantized_params, example)