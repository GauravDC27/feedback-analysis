# rag_v2/intent_classifier.py

from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

import json


def classify_intent(query: str) -> dict:
    prompt = f"""
    Classify the user's intent from this query.

    Query: "{query}"

    Respond in JSON with:
    {{
        "needs_table": true/false,
        "needs_chart": true/false,
        "chart_type": "bar" / "pie" / "scatter" / null,
        "needs_summary": true/false
    }}
    """

    response = client.chat.completions.create(model="gpt-4o",
    messages=[
        {"role": "system", "content": "You classify visualization and table intent for a data analysis chatbot by understanding the user's requirements from the query. In your json response do not outout any symbols like ```json or ``` or any other things that cannot be parsed as json. Only output the json response, which can be directly consumed by code. Wherever you think a small summary is relevant to the query try adding a small description as summary. When you visualize a chart, make you you provide average, sum, standard deviation as summary."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.2)

    content = response.choices[0].message.content.strip()
    print(content)  # Debugging line to see the actual content

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("Failed to decode JSON:", content)
        return {
            "needs_table": False,
            "needs_chart": False,
            "chart_type": None,
            "needs_summary": False
        }

if __name__ == "__main__":
    query = "Show me the average score for Product C with a pie chart."
    print(classify_intent(query))

