import pandas as pd
import os
import sys
import json
from config import Config
from datasets import Dataset, DatasetDict

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import get_train_logger

logger = get_train_logger()

def preprocess_dataset() :
    config = Config()
    data = []
    seed = 42

    if config.DATASET_PATH :
        logger.info("Loading Dataset...")
        with open(config.DATASET_PATH, "r", encoding="utf-8") as f :
            for line in f :
                data.append(json.loads(line))
        df = pd.DataFrame(data)
        df['faq_id'] = df['faq_id'].astype(str)
        hf_dataset = Dataset.from_pandas(df[["question", "answer", "original_title"]])

        ds_split = hf_dataset.train_test_split(test_size=0.2, seed=seed)
        train_val_split = ds_split["train"].train_test_split(test_size = 0.3, seed=seed)

        final_dataset = DatasetDict({
            'train': train_val_split['train'],
            'val': train_val_split['test'],
            'test': ds_split['test']
        })

        return final_dataset