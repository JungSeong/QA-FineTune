#!/usr/bin/env python
# coding: utf-8

# 참고 : https://zero-ai.tistory.com/62

# In[1]:


import torch

print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
print(f"CUDA 버전: {torch.version.cuda}")
print(f"PyTorch 버전: {torch.version}")
print(f"bf16 지원 여부: {torch.cuda.is_bf16_supported()}")


# In[1]:


from datasets import Dataset
import pandas as pd
import glob

file_path = glob.glob("./data/*.xlsx")
dfs = [pd.read_excel(file) for file in file_path]
combined_df = pd.concat(dfs, ignore_index=True)
dataset = Dataset.from_pandas(combined_df)

print(dataset)
print(dataset[0])


# In[4]:


# 데이터 증강을 위해 로컬 EXAONE-3.5-7.8B-Instruct 모델 사용


# In[5]:


from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig, 
    TrainingArguments,
)
from datasets import Dataset
import os, torch, json, wandb, subprocess
from sklearn.model_selection import train_test_split
import torch.nn as nn
from peft import (
    get_peft_model,
    LoraConfig, 
    TaskType,
    prepare_model_for_kbit_training
)


# In[6]:


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, # 모델 가중치를 4bit로 불러오기
    bnb_4bit_compute_dtype=torch.bfloat16, #bfloat16 or float16
    bnb_4bit_quant_type="nf4", # nf4 or fp4
)


# In[10]:


model_id = "/home/vsc/LLM/model/EXAONE-3.5-7.8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    quantization_config=quantization_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)


# In[11]:


import pandas as pd
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


# In[28]:


def generate_augmented_data(faq_content):
    system_message = (
        "당신은 도서관 FAQ 데이터를 바탕으로 자연스러운 대화형 학습 데이터를 생성하는 전문가입니다.\n"
        "제공된 정보를 분석하여 다음 4가지 유형의 데이터를 생성하세요. 각 유형별로 최소 3개씩, 총 12개의 예시를 만드세요.\n\n"
        "이때 판단이 쉬운 예시와 보통, 어려운 예시 총 3가지 상황을 가정하세요."

        "### [답변 가이드라인]\n"
        "1. **label: 'yes' (긍정 확인)**\n"
        "   - 문구: '네, 가능합니다! 도서관 규정에 따르면...'\n"
        "   - 활용: 질문의 조건이 FAQ와 일치할 때 사용.\n"
        "2. **label: 'no' (부정/제한)**\n"
        "   - 문구: '죄송하지만 어렵습니다. 그 이유는...'\n"
        "   - 활용: 질문의 조건이 FAQ 규정에 어긋날 때 사용. '사실과 다르다'는 표현 대신 '규정상 어렵다'를 사용하세요.\n"
        "3. **label: 'info' (단순 정보 제공)**\n"
        "   - 문구: '문의하신 내용에 대해 안내해 드리겠습니다. 관련 서류는...'\n"
        "   - 활용: 네/아니오 판단이 아닌 정보 질문에 사용.\n"
        "4. **label: 'false' (판단 불가)**\n"
        "   - 문구: '죄송합니다. 현재 제공된 정보만으로는 해당 내용을 확인하기 어렵습니다.'\n"
        "   - 활용: FAQ에 없는 내용이거나 질문이 모호할 때 사용.\n\n"
        "모든 답변은 실제 도서관 사서가 방문객에게 설명하듯 친절하고 자연스러운 문장으로 작성하세요."
        "판단 불가의 경우 문구 외에 다른 문장은 생성하지 마세요."
        "JSON 답변 외에 다른 문장은 일체 생성하지 마세요."
    )

    # 사용자가 정의한 user_content
    user_content = f"""
    [도서관 FAQ 정보]
    {faq_content}

    위 정보를 바탕으로 질문과 답변이 논리적으로 완벽하게 이어지는 JSON 데이터를 생성하세요.

    [출력 예시]:
    [
      {{"question": "임산부인데 남편이 대신 발급받을 수 있나요?", "answer": "네, 가능합니다! 도서관 규정에 따르면 임산부의 경우 대리 발급 대상에 포함됩니다.", "label": "yes"}},
      {{"question": "성인 직장인인데 친구가 대신 가도 되나요?", "answer": "죄송하지만, 해당 조건으로는 대리 발급이 어렵습니다. 대리 발급은 아동, 어르신, 장애인, 임산부로 대상이 제한되어 있기 때문입니다.", "label": "no"}},
      {{"question": "장애인 대리 발급 시 어떤 서류가 필요한가요?", "answer": "문의하신 내용에 대해 안내해 드리겠습니다. 장애인 복지카드 또는 장애인 증명서를 지참하시면 됩니다.", "label": "info"}},
      {{"question": "죄송합니다. 현재 제공된 정보만으로는 해당 내용을 확인하기 어렵습니다.", "label": "false"}}
    ]
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    outputs = pipe(
        prompt, 
        max_new_tokens=2048, 
        do_sample=True, 
        temperature=0.7,
        truncation=True,
        eos_token_id=tokenizer.eos_token_id
    )

    # 생성된 텍스트 추출 (결과값만 깔끔하게 가져오기 위해 prompt 이후 내용만 슬라이싱)
    full_text = outputs[0]["generated_text"]
    response = full_text.split("[|assistant|]")[-1].strip()

    return response


# In[29]:


print(dataset[1]["DES"])
generate_augmented_data(dataset[1]["DES"])


# In[33]:


print(type(dataset))
print(dataset.select(range(5)))


# In[36]:


import json
import re
import pandas as pd

def parse_generated_json(response_text):
    try:
        # 1. 마크다운 코드 블록 제거 (```json ... ``` 사이의 내용만 추출)
        json_pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
        match = json_pattern.search(response_text)

        if match:
            json_str = match.group(1)
        else:
            json_str = response_text.strip()

        # 2. JSON 문자열을 파이썬 리스트로 변환
        data_list = json.loads(json_str)
        return data_list
    except Exception as e:
        print(f"파싱 에러 발생: {e}")
        return []


# In[37]:


all_augmented_data = []

for i, row in enumerate(dataset):
    print(f"[{i+1}/{len(dataset)}] 증강 진행 중: {row['TITLE']}")

    # 1. 모델로부터 텍스트 생성
    raw_output = generate_augmented_data(row["DES"])

    # 2. 파싱하여 리스트로 변환
    parsed_list = parse_generated_json(raw_output)

    for item in parsed_list:
        item['faq'] = row['FAQ'] 
        item['title'] = row['TITLE'] 
        all_augmented_data.append(item)

# 4. 최종 결과 저장
final_df = pd.DataFrame(all_augmented_data)
final_df.to_json("augmented_library_faq.jsonl", orient='records', force_ascii=False, lines=True)

print(f"✅ 총 {len(final_df)}개의 학습 데이터 생성 완료!")

