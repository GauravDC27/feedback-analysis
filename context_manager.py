# rag_v2/context_manager.py


# rag_v2/context_manager.py

import openai
from openai import OpenAI
import os
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Session memory to store the last 4 turns
session_memory = {
    "last_4_turns": []  # Each: {"user": "...", "bot": "..."}
}

def reframe_query(current_query: str, session_turns: List[Dict[str, str]]) -> str:
    if not session_turns:
        return current_query  # No history to build from

    # Build chat history context
    history_text = "\n".join([f"User: {t['user']}\nBot: {t['bot']}" for t in session_turns])

    prompt = f"""
You are an AI that makes follow-up questions into standalone questions.

--- HISTORY ---
{history_text}

--- CURRENT QUESTION ---
{current_query}

Reframe the CURRENT QUESTION to be fully self-contained.
Only output the final question text.
"""

    response = client.chat.completions.create(model="gpt-4o", 
    messages=[
        {"role": "system", "content": "You are a professional context-aware question reframer. You are given a current question and a previous question. You need to understand critically if the current question depends on the previous question. If it does, you need to reframe the current question into a complete standalone question with all the necessary previous context. Ensure you understand the question by thinking step by step. Maintain history of previous questions and provide them in the output, in case the current question depends on the previous ones.. It can be more than one previous questions going on in chain of questions. If it does not, you need to return the current question as-is."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=3000,
    temperature=0.3)


    return response.choices[0].message.content
