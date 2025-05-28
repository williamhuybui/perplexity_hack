## Project Layout

### Top-Level Files

* **`main.ipynb`** – Run-me-first notebook that demos the entire pipeline
* **`config.py`** – All pipeline settings (paths, chunk size, weights)
* **`data_processing.py`** – PDF loading and text-chunking helpers
* **`question_generator.py`** – Builds challenge questions with Perplexity Sonar
* **`llm.py`** – Light wrappers around Perplexity API (generation + judging)
* **`evaluation.py`** – Scoring logic, statistics, Plotly charts
* **`utils.py`** – Miscellaneous helpers (folder setup, timers, logging)

### Folders

* **`data/`** – Raw PDF reports
* **`faiss_index_open/`** – Sample FAISS index for a demo RAG model
* **`output/`** – Generated questions, scores, and plots
* **`research/`** – Scratch notebooks, EDA, prompt experiments
* **`user_models/`** – Toy RAG pipelines for testing

> **Quick start:** open `main.ipynb`, point it at your PDFs, and the pipeline will create questions, grade your model, and drop results into `output/`.
