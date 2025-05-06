# rag_v2/load_dataset_to_db.py

import pandas as pd
from db_manager import insert_dataset_records

df = pd.read_csv("dataset/Agentic_GenAI_Feedback_Dataset.csv")
records = df.to_dict(orient="records")
#insert_dataset_records(records)
print("✅ Dataset inserted into MongoDB.")
