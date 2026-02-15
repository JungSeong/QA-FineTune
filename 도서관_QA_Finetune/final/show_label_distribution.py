import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
from infer.config import Config
from datetime import datetime

data = []
with open(Config.AUGMENTED_DATA_PATH, "r", encoding="utf-8") as f :
    for line in f :
        data.append(json.loads(line))

df = pd.DataFrame(data)

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='label', order=Config.LABELS)
plt.title("Label Distribution")
plt.savefig("./label_distribution/label_distribution_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png")