# Agentic Data Analyst

An AI agent that analyzes any user-uploaded dataset — ask questions in plain
English, and the agent decides whether to answer with SQL, Python/pandas, or
both, executes the analysis, and explains the result.

Built with LangGraph, DuckDB, FastAPI, and Streamlit.

## Status
🚧 In development — Week 1 (project setup + data exploration)

## Project structure
```
agentic-data-analyst/
├── notebooks/     # exploration notebooks
├── src/
│   ├── graph/      # LangGraph nodes, edges, state
│   ├── tools/      # SQL tool, pandas tool, visualization generator
│   ├── db/         # DuckDB setup, schema loading
│   └── api/        # FastAPI app
├── app/            # Streamlit frontend
├── data/           # datasets (gitignored)
├── tests/          # evaluation harness
└── requirements.txt
```

## Setup
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Development log
- Week 1: Project setup, F1 dataset exploration, DuckDB integration
