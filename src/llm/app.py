import traceback
import asyncio 
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.CRag.graph import graph, checkpointer, builder

app = FastAPI(title="Government Scheme Assistant API")


class QueryRequest(BaseModel):
    query: str
    thread_id: str | None = None


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Government Scheme Assistant</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #f0f2f5;
            color: #1c1e21;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        header {
            background: #1a5276;
            color: white;
            padding: 16px 24px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
        }
        header h1 { font-size: 20px; font-weight: 600; }
        header p { font-size: 13px; opacity: 0.85; margin-top: 4px; }
        .thread-pill {
            background: rgba(255, 255, 255, 0.15);
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        .schemes-bar {
            background: #eaf2f8;
            padding: 8px 24px;
            font-size: 12px;
            color: #2471a3;
            border-bottom: 1px solid #d4e6f1;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            align-items: center;
        }
        .scheme-tag {
            background: white;
            padding: 2px 8px;
            border-radius: 12px;
            border: 1px solid #aed6f1;
            cursor: pointer;
        }
        .scheme-tag:hover {
            background: #d4e6f1;
        }
        #chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
        }
        .message {
            display: flex;
            flex-direction: column;
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 12px;
            line-height: 1.5;
            font-size: 14px;
            word-wrap: break-word;
            white-space: pre-wrap;
        }
        .message.user {
            align-self: flex-end;
            background: #1a5276;
            color: white;
            border-bottom-right-radius: 2px;
        }
        .message.bot {
            align-self: flex-start;
            background: white;
            color: #2c3e50;
            border: 1px solid #e1e8ed;
            border-bottom-left-radius: 2px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .message.abstained {
            background: #fef9e7;
            border-color: #f9e79f;
            color: #7d6608;
        }
        .msg-sender {
            font-size: 11px;
            font-weight: bold;
            margin-bottom: 4px;
            opacity: 0.7;
            text-transform: uppercase;
        }
        #input-container {
            background: white;
            padding: 16px 24px;
            border-top: 1px solid #e1e8ed;
            display: flex;
            gap: 12px;
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
        }
        #query-input {
            flex: 1;
            padding: 12px 16px;
            font-size: 14px;
            border: 1px solid #ccd1d9;
            border-radius: 8px;
            outline: none;
        }
        #query-input:focus { border-color: #1a5276; }
        #send-btn {
            background: #1a5276;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
        }
        #send-btn:hover { background: #154360; }
        #send-btn:disabled { background: #a6acaf; cursor: not-allowed; }
        .spinner {
            display: inline-block;
            width: 12px;
            height: 12px;
            border: 2px solid #ccc;
            border-top-color: #1a5276;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 6px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <header>
        <div>
            <h1>🏛️ Government Welfare Scheme Assistant</h1>
            <p>Corrective RAG (CRAG) with LangGraph SqliteSaver Persistence</p>
        </div>
        <div class="thread-pill">
            🧵 Thread: <strong id="thread-id-display">Connecting...</strong>
        </div>
    </header>
    <div class="schemes-bar">
        <span><strong>Indexed Schemes:</strong></span>
        <span class="scheme-tag" onclick="suggest('Which ministry issued the guidelines for Mission Saksham Anganwadi and Poshan 2.0?')">Poshan 2.0</span>
        <span class="scheme-tag" onclick="suggest('What are the key objectives and eligibility under PM-KISAN?')">PM-KISAN</span>
        <span class="scheme-tag" onclick="suggest('What are the benefits under PM-KMY (Kisan Maan Dhan Yojana)?')">PM-KMY</span>
        <span class="scheme-tag" onclick="suggest('What are the operational guidelines for PMAY-U 2.0?')">PMAY-U 2.0</span>
        <span class="scheme-tag" onclick="suggest('What is the eligibility for PMAJAY scheme?')">PMAJAY</span>
        <span class="scheme-tag" onclick="suggest('What are the maternity benefits provided under PMMVY?')">PMMVY</span>
    </div>
    <div id="chat-container">
        <div class="message bot">
            <div class="msg-sender">Assistant</div>
            Hello! I am your Government Welfare Scheme Assistant. Ask me any question about central welfare schemes such as Mission Saksham Anganwadi & Poshan 2.0, PM-KISAN, PM-KMY, PMAY-U, PMAJAY, or PMMVY. Our conversation thread is persisted in-memory!
        </div>
    </div>
    <div id="input-container">
        <input type="text" id="query-input" placeholder="Ask a question about a government scheme..." onkeydown="if(event.key==='Enter') sendQuery()" />
        <button id="send-btn" onclick="sendQuery()">Send</button>
    </div>

    <script>
        // In-memory conversation persistence thread ID
        const threadId = "user-thread-" + Math.random().toString(36).substring(2, 9);
        document.getElementById('thread-id-display').textContent = threadId;

        function suggest(q) {
            document.getElementById('query-input').value = q;
            document.getElementById('query-input').focus();
        }

        async function sendQuery() {
            const input = document.getElementById('query-input');
            const btn = document.getElementById('send-btn');
            const chat = document.getElementById('chat-container');
            const query = input.value.trim();
            if (!query) return;

            // Display user message
            const userMsg = document.createElement('div');
            userMsg.className = 'message user';
            userMsg.innerHTML = '<div class="msg-sender">You</div>' + escapeHtml(query);
            chat.appendChild(userMsg);

            input.value = '';
            input.disabled = true;
            btn.disabled = true;

            // Loading indicator
            const botMsg = document.createElement('div');
            botMsg.className = 'message bot';
            botMsg.innerHTML = '<div class="msg-sender">Assistant</div><span class="spinner"></span>Thinking and retrieving verified documents...';
            chat.appendChild(botMsg);
            chat.scrollTop = chat.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, thread_id: threadId })
                });

                const data = await response.json();
                if (response.ok) {
                    botMsg.innerHTML = '<div class="msg-sender">Assistant</div>' + escapeHtml(data.answer);
                    if (data.abstained) {
                        botMsg.classList.add('abstained');
                    }
                } else {
                    botMsg.innerHTML = '<div class="msg-sender">Assistant</div>Error: ' + escapeHtml(data.detail || 'Failed to process request.');
                }
            } catch (err) {
                botMsg.innerHTML = '<div class="msg-sender">Assistant</div>Network error: ' + escapeHtml(err.message);
            } finally {
                input.disabled = false;
                btn.disabled = false;
                input.focus();
                chat.scrollTop = chat.scrollHeight;
            }
        }

        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home_page():
    return HTML_PAGE


@app.post("/")
def home():
    return {
        "message": "This is the home endpoint of the Government Scheme Chatbot API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "government-scheme-assistant",
        "persistence": "SqliteSaver"
    }


@app.get("/thread/{thread_id}")
def get_thread_state(thread_id: str):
    """Retrieve persisted state from LangGraph SqliteSaver for a given thread."""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state or not state.values:
        return {
            "thread_id": thread_id,
            "persisted": False,
            "message": "No checkpoint found for this thread."
        }

    return {
        "thread_id": thread_id,
        "persisted": True,
        "query": state.values.get("query"),
        "answer": state.values.get("answer"),
        "abstained": state.values.get("abstained", False),
        "next_nodes": list(state.next)
    }


@app.post("/chat")
async def chat(request: QueryRequest):
    print("\n========== CHAT ENDPOINT CALLED ==========")
    print("User query:", request.query)
    thread_id = request.thread_id or "default_user"
    print("Thread ID:", thread_id)

    try:
        print("Starting CRAG graph execution with SQLite persistence...")

        config = {"configurable": {"thread_id": thread_id}}

        # Reset per-turn transient evidence lists so they don't carry over into subsequent turns
        result = await graph.ainvoke({
            "query": request.query,
            "bm25_results": [],
            "dense_results": [],
            "rrf_results": [],
            "reranked_results": [],
            "relevant_documents": [],
            "ambiguous_documents": [],
            "irrelevant_documents": [],
            "refined_documents": [],
            "evidence_documents": [],
            "web_documents": [],
            "relevant_web_documents": [],
            "ambiguous_web_documents": [],
            "irrelevant_web_documents": [],
            "web_evidence_documents": [],
            "refined_web_docs": []
        }, config=config)

        print("CRAG GRAPH EXECUTION COMPLETED")
        print("Answer:", result.get("answer"))
        print("Abstained:", result.get("abstained"))

        return {
            "answer": result.get("answer", ""),
            "abstained": result.get("abstained", False),
            "abstention_reason": result.get("abstention_reason"),
            "thread_id": thread_id
        }

    except Exception as e:
        print("\n========== CRAG GRAPH ERROR ==========\n")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
