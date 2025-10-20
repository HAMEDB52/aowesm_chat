import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ------------------------------------------------------------------
# 1.  Guard-rail: die early if the secret is not there
# ------------------------------------------------------------------
try:
    api_key= st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error(
        "OPENAI_API_KEY not found in secrets.  "
        "Please add it to `.streamlit/secrets.toml` (local) or the Cloud Secrets panel."
    )
    st.stop()

# ------------------------------------------------------------------
# 2.  Page config
# ------------------------------------------------------------------
st.markdown("# ChatGPT-like clone")
st.caption("Powered by LangChain + Streamlit")

# ------------------------------------------------------------------
# 3.  LangChain LLM
# ------------------------------------------------------------------
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-3.5-turbo",
    temperature=0.7,
    streaming=True,
)

# ------------------------------------------------------------------
# 4.  Session-state defaults
# ------------------------------------------------------------------
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    # Optional: give the assistant a personality
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# ------------------------------------------------------------------
# 5.  Render chat history
# ------------------------------------------------------------------
for msg in st.session_state.messages:
    # Skip the system message in the UI
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------------
# 6.  Chat input + generate response
# ------------------------------------------------------------------
if prompt := st.chat_input("What is up?"):
    # 6a.  Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 6b.  Build LangChain message list
    langchain_msgs = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            langchain_msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            langchain_msgs.append(AIMessage(content=m["content"]))
        elif m["role"] == "system":
            langchain_msgs.append(SystemMessage(content=m["content"]))

    # 6c.  Stream assistant reply
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for chunk in llm.stream(langchain_msgs):
            full_response += chunk.content
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    # 6d.  Persist assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_response})