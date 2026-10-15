# AOU Smart Academic Assistant

A Retrieval-Augmented Generation (RAG) chatbot for Arab Open University academic information.

## What changed from the original project

- Replaced the Egyptian private-universities scope with AOU academic information.
- Replaced the old `priavte_school` Chroma database with `aou_chroma_db`.
- Added bilingual-friendly multilingual embeddings.
- Preserved source metadata so the assistant can cite retrieved sources.
- Fixed chat-history roles so assistant messages are not treated as user messages.
- Added an ingestion script for official AOU webpages and local PDF/TXT documents.
- Added a clean Streamlit interface and source panel.

## Recommended Python version

Use Python 3.11 or 3.12.

## 1. Open the project in VS Code

Open the `AOU_RAG_Assistant` folder in VS Code.

## 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

## 3. Install and run Ollama

Install Ollama, then download the model used by the project:

```bash
ollama pull gemma2:2b
```

Make sure Ollama is running before starting the chatbot.

## 4. Add AOU knowledge sources

Official starter URLs are already listed in:

`data/aou_sources.txt`

For stronger answers, download official AOU PDFs and place them in:

`data/documents/`

Examples: Student Guide, Student Prospectus, regulations, Computer Science programme specification, and study plan.

## 5. Build the vector database

```bash
python ingest.py --reset
```

The first run may download the multilingual embedding model.

## 6. Run the chatbot

```bash
streamlit run app.py
```

## Project flow

User Question -> Query Rewrite -> Chroma Retrieval -> AOU Context -> Gemma 2 -> Answer + Sources

## Main files

- `app.py`: Streamlit interface
- `agents.py`: query rewriting, retrieval, and response agents
- `workflow.py`: LangGraph workflow
- `prompt.py`: AOU-specific prompts and source formatting
- `models.py`: shared workflow state
- `config.py`: model/database settings
- `ingest.py`: builds the RAG knowledge base
- `data/aou_sources.txt`: official AOU webpage sources
- `data/documents/`: place official PDFs here

## Important note for the graduation project

This is a strong baseline prototype. Later improvements can include academic advising, prerequisite checking, transcript analysis, course recommendations, evaluation metrics, and an admin document-update interface.
