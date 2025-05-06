# rag_v2/rag_api.py
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from rag_chain import rag_chain
import uvicorn
from embedder import build_vector_store
import openai
from fastapi.staticfiles import StaticFiles
from db_manager import insert_chat_log
app = FastAPI()

# Mount the static files directory
app.mount("/templates/charts", StaticFiles(directory="templates/charts"), name="charts")

class RAGChatRequest(BaseModel):
    query: str

@app.post("/rag_chat", response_class=JSONResponse)
async def rag_chat(request: RAGChatRequest):
    state = {"query": request.query}
    result = rag_chain.invoke(state)
    # NEW: Save chat to MongoDB
    '''reframed = result.get("reframed_query", request.query)
    insert_chat_log(
        user_query=request.query,
        reframed_query=reframed,
        intent=result.get("intent"),
        messages=result["response_messages"]
    )'''
    return {"messages": result["response_messages"]}

@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    with open("static/frontend/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

def reframe_query(current_query: str, previous_query: str) -> str:
    prompt = (
        "You are a context-aware question reframer.\n"
        f"User: {current_query}\n"
        f"Previous question: {previous_query}\n"
        "If the current question depends on the previous, reframe it into a complete standalone question. If not, return the current question as-is.\n"
        "Provide only the reframed question without explanation.\n"
    )

    response = openai.Completion.create(
        model="gpt-4o",  # Ensure you specify the correct model
        prompt=prompt,
        max_tokens=150,
        temperature=0.3
    )

    return response.choices[0].text.strip()  # Adjusted to use .text instead of .message.content

if __name__ == "__main__":
    print("Starting RAG API...")
    #if .bin file does not exist, build the vector store
    if not os.path.exists("dataset/faiss_index.bin"):
        build_vector_store()
    uvicorn.run(app, host="0.0.0.0", port=9000)

