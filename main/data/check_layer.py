import torch.nn as nn

model = "/home/vsc/LLM/model/Exaone-3.5-7.8B-Instruct"

linear_layers = set()
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        # 보통 'model.layers.0.self_attn.q_proj' 같은 형태에서 
        # 마지막 'q_proj'만 추출합니다.
        names = name.split('.')
        linear_layers.add(names[-1])

print(f"발견된 Linear 레이어들: {list(linear_layers)}")