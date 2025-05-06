# rag_v2/db_manager.py

from pymongo import MongoClient
import os
from datetime import datetime

# MongoDB config
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "rag_chatbot"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Dataset collection
feedback_collection = db["feedback_dataset"]

# Chat logs collection
logs_collection = db["chat_logs"]

def insert_dataset_records(records: list):
    if records:
        feedback_collection.insert_many(records)

def insert_chat_log(user_query, reframed_query, intent, messages):
    logs_collection.insert_one({
        "timestamp": datetime.utcnow(),
        "user_query": user_query,
        "reframed_query": reframed_query,
        "intent": intent,
        "bot_messages": messages
    })

def get_all_dataset():
    return list(feedback_collection.find({}, {"_id": 0}))
