import streamlit as st
import requests

# Page config
st.set_page_config(
    page_title="Indian Law Chatbot (LangChain)",
    page_icon="⚖️",
    layout="wide"
)

# Backend API URL
BACKEND_URL = "http://localhost:8000"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title
st.title("⚖️ Indian Law Chatbot")
st.caption("🔗 Powered by LangChain + Gemini + ChromaDB")

# Sidebar
with st.sidebar:
    st.header("📊 System Info")
    
    # Get stats
    try:
        response = requests.get(f"{BACKEND_URL}/stats")
        if response.status_code == 200:
            stats = response.json()
            st.metric("Documents Loaded", stats.get("total_documents", 0))
            st.metric("Unique Sources", stats.get("unique_sources", 0))
            
            with st.expander("📚 Loaded Sources"):
                for source in stats.get("sources", []):
                    st.caption(f"• {source}")
            
            st.info(f"**LLM:** {stats.get('llm_model', 'N/A')}")
            st.info(f"**Embeddings:** {stats.get('embedding_model', 'N/A')}")
    except:
        st.error("⚠️ Backend not connected")
    
    st.divider()
    
   
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.caption(f"**Source {i}:** {source['source']}")
                        if source.get('page'):
                            st.caption(f"Page: {source['page']}")
                        st.text(source['content'])
                        st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about Indian law..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"query": prompt},
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(sources, 1):
                                st.caption(f"**Source {i}:** {source['source']}")
                                if source.get('page'):
                                    st.caption(f"Page: {source['page']}")
                                st.text(source['content'])
                                st.divider()
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    error_msg = "Sorry, I encountered an error."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })
            
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })

# Footer
st.divider()
st.caption("⚖️ Indian Law Chatbot - Powered by LangChain, Gemini & ChromaDB")