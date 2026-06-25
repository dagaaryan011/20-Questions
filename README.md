# PDF-based RAG using Groq

This project uses a PDF-based knowledge base and the Groq-hosted Llama 3.1 model for creating a retrieval-augmented chatbot. The agent responds to yes/no questions based on the content of the book in the PDF, providing hints about the genre or author, and if the guess is close to the book's title, the agent declares a win.

## Local setup

### 1. Install Python packages

```bash
pip install -r requirements.txt
```

### 2. Get a Groq API key

Sign up at [console.groq.com](https://console.groq.com) and create a free API key, then export it:

```bash
export GROQ_API_KEY=your-key-here
```

### 3a. Run the CLI chatbot

```bash
python run.py
```

### 3b. Run the web app locally

```bash
uvicorn app:app --reload
```

Then open http://localhost:8000 in your browser.

## Deploying to Render

1. Push this repo to GitHub (the PDF in `files/` is committed since it's the knowledge base).
2. On [Render](https://render.com), create a new **Web Service** and connect the GitHub repo. Render will detect `render.yaml` automatically.
3. In the Render dashboard, set the `GROQ_API_KEY` environment variable to your Groq key.
4. Deploy. First boot takes ~2-3 minutes since `sentence-transformers` has to download the embedding model.


