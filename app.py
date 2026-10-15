import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from config import CHROMA_DIR
from models import State
from workflow import Workflow


st.set_page_config(page_title="AOU Smart Academic Assistant", page_icon="🎓", layout="centered")
st.title("🎓 AOU Smart Academic Assistant")
st.caption("RAG-based assistant for Arab Open University academic information")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not os.path.isdir(CHROMA_DIR):
    st.warning("Knowledge base not found. Add AOU documents/sources, then run: python ingest.py --reset")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"], start=1):
                    st.markdown(f"**[{i}]** {source}")

if prompt := st.chat_input("Ask about AOU courses, prerequisites, regulations, or student procedures..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    history = []
    for message in st.session_state.messages[:-1]:
        if message["role"] == "user":
            history.append(HumanMessage(content=message["content"]))
        else:
            history.append(AIMessage(content=message["content"]))

    initial_state = State(
        chat_history=history,
        user_input=prompt,
        context=None,
        sources=None,
        response="",
        rewritten_query="",
    )

    with st.chat_message("assistant"):
        try:
            result = Workflow().run(initial_state)
            response = result.get("response") or "No response was generated."
            sources = result.get("sources") or []
            st.markdown(response)

            if sources:
                with st.expander("Sources"):
                    for i, source in enumerate(sources, start=1):
                        st.markdown(f"**[{i}]** {source}")

            st.session_state.messages.append(
                {"role": "assistant", "content": response, "sources": sources}
            )
        except Exception as exc:
            st.error(f"Could not run the assistant: {exc}")
            st.info("Make sure Ollama is running and that you built the knowledge base with: python ingest.py --reset")
