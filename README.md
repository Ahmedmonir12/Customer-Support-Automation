# Customer Support Automation

A multi-agent customer support system that classifies tickets, retrieves answers from a knowledge base (RAG), drafts a grounded reply, and escalates to a human when needed.

## Architecture

```
Ticket → Classifier → Knowledge Agent (RAG) → Response Writer → Escalation Agent → Auto-reply or Human
```

| Agent | Job | Implementation |
|---|---|---|
| Classifier | Intent, urgency, sentiment | Fine-tuned DistilBERT |
| Knowledge Agent | Retrieve FAQs, past tickets, product docs | ChromaDB + all-MiniLM-L6-v2 |
| Response Writer | Draft a grounded, tone-matched reply | LLM with retrieved context |
| Escalation Agent | Decide auto-send vs human handoff | Rules + confidence scores |

## Tech Stack

- Python 3.11, managed with uv
- PyTorch, Hugging Face Transformers and Datasets
- ChromaDB, sentence-transformers
- Streamlit (UI)
- LangGraph (optional orchestration)

## Project Structure

```
support-agent/
├── config.py            # paths, thresholds, model names
├── pipeline.py          # runs the 4 agents end to end
├── app.py               # Streamlit UI
├── data/                # raw, processed, kb (faqs, product_docs), eval
├── classifier/          # prepare_data.py, train.py, infer.py, models/
├── rag/                 # ingest.py, retriever.py, chroma_db/
├── agents/              # one file per agent + prompts.py
├── eval/                # evaluation scripts and results
├── notebooks/           # Colab training notebook
└── tests/
```

## Setup

```powershell
# install uv (once)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# create environment
uv venv --python 3.11
.venv\Scripts\activate

# install dependencies
uv add transformers datasets torch scikit-learn chromadb sentence-transformers streamlit langgraph pandas python-dotenv
```

Create a `.env` file in the project root:

```
LLM_API_KEY=your_key_here
```

## Usage

**1. Prepare data and train the intent classifier** (use Colab for a free GPU)

```powershell
python classifier/prepare_data.py
python classifier/train.py
```

**2. Build the knowledge base index**

```powershell
python rag/ingest.py
```

**3. Run the pipeline from the command line**

```powershell
python pipeline.py "I was charged twice for my order"
```

**4. Launch the UI**

```powershell
streamlit run app.py
```

## Escalation Rules

A ticket goes to a human if any of these hold:

- Classifier confidence is below 0.6
- Sentiment is very negative, or the ticket mentions legal action, fraud, or a large refund
- Retrieval similarity is below the threshold (no good source found)
- The customer asks for a human

All thresholds live in `config.py`.

## Evaluation

```powershell
python eval/eval_classifier.py
python eval/eval_retrieval.py
python eval/eval_responses.py
python eval/eval_escalation.py
```

| Component | Metric |
|---|---|
| Classifier | Accuracy, macro-F1, confusion matrix |
| Retrieval | Recall@k on 50 hand-labeled questions |
| Responses | LLM-as-judge plus manual check (groundedness, tone) |
| Escalation | Precision and recall |

Ablations: with vs without RAG, fine-tuned vs zero-shot classifier.

## Results

| Component | Metric | Score |
|---|---|---|
| Classifier | Accuracy | TBD |
| Classifier | Macro-F1 | TBD |
| Retrieval | Recall@5 | TBD |
| Escalation | Precision / Recall | TBD |

## Datasets

- Intent classification: Bitext Customer Support dataset or Banking77 (Hugging Face)
- Past tickets: Customer Support on Twitter (Kaggle)
- FAQs and product docs: written for a fictional company

## Roadmap

- [ ] Data collection and cleaning
- [ ] Fine-tune intent classifier
- [ ] RAG pipeline
- [ ] Response writer and escalation logic
- [ ] Evaluation and ablations
- [ ] Streamlit UI and final report
- [ ] Stretch: LoRA tone-matched response model

## License

MIT