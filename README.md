# LangChain Streamlit Chat

A small streaming chat app built on [LangChain](https://python.langchain.com/) and [Streamlit](https://streamlit.io/). Responses stream token by token, the conversation is kept in the session, and the model and temperature can be changed from the sidebar.

## Run

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then add your OpenAI key
streamlit run apps.py
```

The key can also be provided as the `OPENAI_API_KEY` environment variable. If it is missing, the app shows a clear message instead of failing on the first request.

## Features

- Streaming responses through `ChatOpenAI.stream`
- Model selector (`gpt-4o-mini`, `gpt-4o`, `gpt-4.1-mini`) and temperature slider
- Clear-chat button
- API errors are shown in the chat instead of crashing the app

## Tests

```bash
python -m pytest
```

The tests run the real app script with Streamlit's `AppTest`, replacing the OpenAI model with LangChain's fake chat model, so they need no API key or network access. They cover the missing-key message, a full user → assistant round trip, and clearing the chat.
