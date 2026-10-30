"""
Streamlit Frontend for Government Welfare Scheme Assistant (CRAG)
Features:
- Streaming answer output using Streamlit's write_stream
- Sidebar with past conversations (stores and displays the latest 5 chats only)
- Resume past chats with a single click
- LangGraph integration with SQLite persistence (SqliteSaver)
- Simple, student-friendly design without unnecessary complexity
"""

import time
import uuid
import asyncio
import streamlit as st

# Import CRAG graph and SQLite chat history helper
from src.CRag.graph import graph
from src.llm.chat_history import (
    get_latest_chat_sessions,
    get_chat_messages,
    save_message,
)

# Page configuration
st.set_page_config(
    page_title="Government Scheme Assistant",
    page_icon="🏛️",
    layout="wide",
)

# -------------------------------------------------------------
# Initialize Session State
# -------------------------------------------------------------
if "current_thread_id" not in st.session_state:
    # Generate a unique thread ID for a new conversation
    st.session_state.current_thread_id = f"chat_{uuid.uuid4().hex[:8]}"

if "messages" not in st.session_state:
    # Load any existing messages for the current thread from SQLite
    st.session_state.messages = get_chat_messages(st.session_state.current_thread_id)


# Helper function to switch threads
def switch_thread(thread_id: str):
    st.session_state.current_thread_id = thread_id
    st.session_state.messages = get_chat_messages(thread_id)
    st.rerun()


# Helper function to start a new thread
def start_new_thread():
    new_id = f"chat_{uuid.uuid4().hex[:8]}"
    st.session_state.current_thread_id = new_id
    st.session_state.messages = []
    st.rerun()


# -------------------------------------------------------------
# Sidebar: Latest 5 Chats & Scheme Navigation
# -------------------------------------------------------------
with st.sidebar:
    st.title("🏛️ Scheme Assistant")
    st.caption("Corrective RAG (CRAG) with SQLite persistence")

    # Button to start a fresh conversation
    if st.button("➕ Start New Chat", use_container_width=True, type="primary"):
        start_new_thread()

    st.divider()

    # Section for resuming previous chats (strictly latest 5)
    st.subheader("💬 Recent Chats (Latest 5)")
    recent_chats = get_latest_chat_sessions(limit=5)

    if recent_chats:
        for chat_item in recent_chats:
            thread_id = chat_item["thread_id"]
            title = chat_item["title"] or "Untitled Chat"

            # Highlight current active chat
            is_active = thread_id == st.session_state.current_thread_id
            label = f"👉 {title}" if is_active else f"🗨️ {title}"

            if st.button(
                label,
                key=f"btn_{thread_id}",
                use_container_width=True,
                disabled=is_active,
            ):
                switch_thread(thread_id)
    else:
        st.info("No saved chats yet. Start asking questions!")

    st.divider()

    # Helpful guide for users on available indexed schemes
    st.subheader("📑 Indexed Schemes")
    st.markdown("""
    - **PM-KISAN**: ₹6,000/yr income support for farmers
    - **PMMVY**: ₹5,000 maternity cash benefit
    - **PMAY-U 2.0**: Urban housing & interest subsidies
    - **PM-KMY**: ₹3,000/mo pension for small farmers
    - **PMAJAY**: SC community development & grants
    - **Poshan 2.0**: Anganwadi nutrition & health services
    """)

    st.divider()
    st.caption(f"**Current Thread:** `{st.session_state.current_thread_id}`")
    st.caption("💾 **Persistence:** SQLite (`data/checkpoints.db`)")


# -------------------------------------------------------------
# Main Chat Area
# -------------------------------------------------------------
st.title("🏛️ Indian Government Welfare Scheme Assistant")
st.write(
    "Ask any question about eligibility, financial assistance, and required documents. "
    "All answers are grounded in official scheme guidelines with exact citations."
)

# Display historical messages in current thread
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Streaming generator helper
def text_streamer(text: str):
    """
    Yields words smoothly for Streamlit's write_stream,
    providing a clean, natural streaming effect.
    """
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.015)


# Handle user input
user_query = st.chat_input("Ask a question (e.g., What are the benefits of PM-KISAN?)...")

if user_query:
    # 1. Display and save user message
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.messages.append({"role": "user", "content": user_query})

    # Save user message to SQLite (title is first 40 chars of query for the sidebar)
    title = user_query[:40] + ("..." if len(user_query) > 40 else "")
    save_message(
        thread_id=st.session_state.current_thread_id,
        role="user",
        content=user_query,
        title=title,
        max_chats=5,
    )

    # 2. Execute CRAG Graph and stream the assistant response
    with st.chat_message("assistant"):
        with st.status("🔍 Analyzing official scheme guidelines...", expanded=True) as status:
            st.write("🔎 Step 1: Performing hybrid search (BM25 + Dense Semantic Embeddings)...")
            st.write("⚖️ Step 2: Evaluating document relevance & checking evidence sufficiency...")

            # Run the compiled LangGraph with the current thread configuration
            config = {"configurable": {"thread_id": st.session_state.current_thread_id}}

            # Run graph execution in event loop
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    graph.ainvoke(
                        {
                            "query": user_query,
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
                            "refined_web_docs": [],
                        },
                        config=config,
                    )
                )
                loop.close()
            except Exception as e:
                result = {
                    "answer": f"An error occurred while processing your request: {str(e)}",
                    "abstained": True,
                }

            if result.get("abstained"):
                status.update(
                    label="⚠️ Insufficient evidence found in official guidelines.",
                    state="complete",
                    expanded=False,
                )
            else:
                status.update(
                    label="✅ Verified evidence found. Generating response...",
                    state="complete",
                    expanded=False,
                )

        # Stream the assistant answer
        full_answer = result.get(
            "answer", "I'm sorry, I could not generate an answer at this time."
        )

        st.write_stream(text_streamer(full_answer))

        # Save assistant message to SQLite
        save_message(
            thread_id=st.session_state.current_thread_id,
            role="assistant",
            content=full_answer,
            max_chats=5,
        )

        st.session_state.messages.append(
            {"role": "assistant", "content": full_answer}
        )
