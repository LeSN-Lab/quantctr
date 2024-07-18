import pickle
from functools import partial
from flax import serialization
from flax.core import freeze, unfreeze
import jax.numpy as jnp
import jax
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pretrained import pretrained_resnet

def save_jax_resnet50_params(output_file):
    # ResNet50 모델과 파라미터 로드
    model_cls, variables = pretrained_resnet(50)

    # 모델 인스턴스 생성
    model = model_cls()

    # 더미 입력으로 모델 초기화
    dummy_input = jnp.ones((1, 224, 224, 3))
    init_params = model.init(jax.random.PRNGKey(0), dummy_input)
    
    # 디버깅을 위한 출력
    print("Initial Parameters Structure:")
    print(jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else x, init_params))
    
    # print("Variables:")
    # print(variables)
    
    # 사전 학습된 파라미터로 업데이트
    if variables is not None:
        params = freeze(unfreeze(init_params).update(variables))
    else:
        params = freeze(init_params)

    # 파라미터를 바이트 형식으로 직렬화
    serialized_params = serialization.to_bytes(params)

    # 파일에 저장
    with open(output_file, 'wb') as f:
        pickle.dump(serialized_params, f)

    print(f"JAX ResNet50 parameters saved to {output_file}")

    # 저장된 파라미터 구조 확인
    print("Saved Parameter Structure:")
    print(jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else x, params))

if __name__ == '__main__':
    save_jax_resnet50_params('jax_resnet50_params.pkl')