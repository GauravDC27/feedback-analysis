# rag_v2/visualization_generator.py

import pandas as pd
import plotly.express as px
import uuid
import os
import json
from openai import OpenAI

CHARTS_DIR = "templates/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)
import openai
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# rag_v2/visualization_generator.py (add this at the top or new file)

def get_chart_instructions(df, query):

    prompt = f"""
    You are a data visualization expert for charts like bars, pie charts, scatter plots, etc. A user asked:
    "{query}"

    The available columns in the dataset are:
    {list(df.columns)}

    Please suggest the best chart type (bar, pie, scatter), the x-axis, the y-axis,
    and whether aggregation (sum, mean, count) is needed by thinnking critically step by step. Make sure that your output is in exact same format as below, so that it can be parsed as json. Do not include as extra symbols like ```json or ``` or any other things that cannot be parsed as json.

    Respond in this JSON format:
    {{
        "chart_type": "bar",
        "x_column": "Product",
        "y_column": "Score",
        "aggregation": "mean"  // or "sum", "count", or null
    }}

    The x-column and y-column should be the most relevant columns or details that are most relevant to the query. It can be a column name or a column name with an aggregation function.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You generate chart configuration from user query and column names. Based on the content and the query, decide the best chart type, x-axis, y-axis, and whether aggregation is needed. If aggregation is needed, decide the best aggregation function."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    print("The response from get_chart_instructions: ", response.choices[0].message.content.strip())
    return json.loads(response.choices[0].message.content.strip())



# rag_v2/visualization_generator.py

def generate_chart(df: pd.DataFrame, chart_type: str, x: str, y: str, aggregation: str = None) -> str:
    if aggregation:
        # Ensure aggregation only happens if needed
        if x in df.columns and y in df.columns:
            df = df.groupby(x).agg({y: aggregation}).reset_index()

    fig = None
    if chart_type == "bar":
        fig = px.bar(df, x=x, y=y, title=f"{aggregation.title() if aggregation else ''} {y} by {x}")
    elif chart_type == "pie":
        fig = px.pie(df, names=x, values=y, title=f"{aggregation.title() if aggregation else ''} {y} Distribution")
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}")
    else:
        raise ValueError("Unsupported chart type")

    filename = f"{uuid.uuid4().hex}.html"
    chart_path = os.path.join(CHARTS_DIR, filename)
    fig.write_html(chart_path)

    return f"templates/charts/{filename}"


def generate_summary_from_df(query: str, df):
    # Convert compact version of df to CSV string
    df_preview = df.head(10).to_csv(index=False)

    prompt = f"""
        You are a professional data analyst. A user asked this question:

        "{query}"

        You have the following sample data retrieved from the knowledge base (max 10 rows):

        {df_preview}

        Please write a helpful summary based on:
        - The user's query
        - The sample data
        - Any numeric fields (like Score): mention average, sum, std deviation
        - Highlight any interesting patterns or insights
        - Keep it concise, natural, and professional.

        Respond with a short paragraph in plain English.
        """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You generate smart, insight-driven summaries from tabular data and user intent. You are a professional data analyst and summarizer for product reviews and feedback. So think critically step by step and provide a summary that is helpful and insightful."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


# Example
if __name__ == "__main__":
    sample_df = pd.DataFrame({
        "Product": ["A", "B", "C"],
        "Score": [7.5, 8.1, 6.9]
    })
    print(generate_chart(sample_df, "bar", "Product", "Score"))

