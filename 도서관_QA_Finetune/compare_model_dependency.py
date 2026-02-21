import json
import requests

def compare_model_configs(config1_path, config2_path, is_url=False):
    def load_config(path):
        if is_url:
            return requests.get(path).json()
        with open(path, 'r') as f:
            return json.load(f)

    cfg1 = load_config(config1_path)
    cfg2 = load_config(config2_path)

    # 1. 의존성에 직접적인 영향을 주는 핵심 키들
    dependency_keys = [
        "model_type", 
        "architectures", 
        "quantization_config", 
        "torch_dtype",
        "transformers_version"
    ]

    print(f"--- [의존성 체크 결과] ---")
    is_compatible = True
    
    for key in dependency_keys:
        val1 = cfg1.get(key)
        val2 = cfg2.get(key)
        
        # quantization_config 내부의 quant_method 등도 체크하기 위해 문자열 비교
        if val1 == val2:
            print(f"[✅ 일치] {key}: {val1}")
        else:
            print(f"[❌ 불일치] {key}:")
            print(f"   - 모델 1: {val1}")
            print(f"   - 모델 2: {val2}")
            is_compatible = False

    # 2. 모델 규모(Size) 차이 출력 (참고용)
    print(f"\n--- [모델 규모 차이 (참고)] ---")
    size_keys = ["hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"]
    for key in size_keys:
        print(f"   - {key}: {cfg1.get(key)} vs {cfg2.get(key)}")

    if is_compatible:
        print("\n결론: 두 모델의 아키텍처와 양자화 방식이 일치합니다. 동일한 Docker 환경에서 구동 가능합니다!")
    else:
        print("\n결론: 핵심 의존성이 다릅니다. 별도의 환경 세팅이 필요할 수 있습니다.")

# 실행 예시 (실제 Hugging Face URL 사용 가능)
url_2_4b = "https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-AWQ/raw/main/config.json"
url_32b = "https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-32B-Instruct-AWQ/raw/main/config.json"
compare_model_configs(url_2_4b, url_32b, is_url=True)