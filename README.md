# Applied AI Monorepo

This monorepo contains several modular Python libraries for AI, NLP, and retrieval tasks. Each library can be used independently or as part of a larger pipeline.

---

## Libraries

### 1. `chunking` — Flexible Text Chunking & Embedding

- **Description:**  
  Config-driven library for splitting text into chunks and embedding them using various providers (OpenAI, HuggingFace, Ollama, Voyage, etc).
- **Entrypoint:**  
  `from chunking.pipeline import run_chunking`
- **Configurable Arguments:**
  - `input_path`: Path to input text file.
  - `config_path`: YAML config file for chunking/embedding.
  - `output_path`: Where to save results (optional).
  - `provider`: Embedding provider (`openai`, `huggingface`, `ollama`, `voyage`).
  - `model_name`: Model to use for embedding.
  - `host`: Custom host for embedding API.
  - `chunk_method`: Chunking strategy (`sentence`, `paragraph`, `line`, `header`, or custom).
  - `max_tokens`: Max tokens per chunk.
  - `overlap_tokens`: Overlap between chunks.
  - `inject_headers`: Whether to inject headers into chunks.
  - `header_regex`: Regex for header detection.
  - `tokenizer`: Custom tokenizer (optional).
  - `custom_chunk_fn`: Custom chunking function (optional).
  - `raw_text`: Provide text directly instead of file.

- **Example:**
  ```python
  from chunking.pipeline import run_chunking
  results = run_chunking(
      input_path="input/GTM AI Email Thread.md",
      config_path="config.yaml",
      provider="ollama",
      model_name="mxbai-embed-large:latest",
      host="http://localhost:11434"
  )
  ```

---

### 2. `generation` — Unified LLM Generation

- **Description:**  
  Unified interface for calling LLMs (OpenAI, Anthropic, Ollama) with configurable context and parameters.
- **Entrypoint:**  
  `from generation.router import get_llm`
- **Configurable Arguments:**
  - `provider`: LLM provider (`openai`, `anthropic`, `ollama`).
  - `model`: Model name.
  - `params`: Dict of model parameters (e.g., `temperature`, `max_tokens`).
  - `context`: System prompt/context.
  - `api_key`: API key (optional, usually from env).

- **Example:**
  ```python
  from generation.router import get_llm
  llm, cfg = get_llm(provider="openai", model="gpt-4o-mini")
  messages = [
      {"role": "system", "content": cfg.get("context", "You are helpful.")},
      {"role": "user", "content": "What's the capital of France?"}
  ]
  output = llm.chat(messages, model=cfg["model"], **cfg.get("params", {}))
  ```

---

### 3. `keyword-extraction` — Configurable Keyword Extraction

- **Description:**  
  Extracts keywords and phrases from text using multiple strategies, with stopword pruning and lemmatization.
- **Entrypoint:**  
  CLI: `python -m scripts.run_keyword_extractor --config config.yaml`  
  Library: `from keyword_extractor.extractor import KeywordExtractor`
- **Configurable Arguments (CLI):**
  - `--config`: Path to YAML config.
  - `--input`: Input text file.
  - `--output`: Output file.
  - `--stopwords`: Stopwords file.
  - `--postprocess`: Enable stopword pruning.
  - `--prune-mode`: Stopword pruning mode.

- **Configurable Options (YAML):**
  - `use_lemmatization`, `remove_stopwords`, `min_word_length`, `min_phrase_length`, `top_n_keywords`, `top_n_phrases`, `extract_phrases`, `spacy_model`, `input_dir`, `output_file`.

---

### 4. `mcp-client` — Model Context Protocol Client

- **Description:**  
  Client for running agentic inference against MCP/JSON-RPC servers (e.g., Omni, OpenAI).
- **Entrypoint:**  
  CLI: `python -m mcp_client.runner --mcp-id <ID> --query <QUERY>`
- **Configurable Arguments:**
  - `--mcp-id`: MCP server ID (from config).
  - `--query`: Query string.

---

### 5. `retrieval` — FAISS-based Vector Retrieval

- **Description:**  
  Vector search over embeddings using FAISS, with metadata support.
- **Entrypoint:**  
  `from retrieval.faiss_retriever import FaissRetriever`
- **Configurable Arguments:**
  - `config_path`: Path to YAML config.
  - Or pass config dict directly (see below).

- **Configurable Options (YAML):**
  - `provider`, `index_path`, `metadata_path`, `top_k`, `normalize`, `metric`

- **Example:**
  ```python
  from retrieval.faiss_retriever import FaissRetriever
  retriever = FaissRetriever(config_path="config.yaml")
  results = retriever.query("your query here")
  ```

---

### 6. `slack-search` — Slack Message Search

- **Description:**  
  Search Slack messages and threads, with optional embedding ranking.
- **Entrypoint:**  
  `from slack_search.searcher import SlackSearcher`
- **Configurable Arguments:**
  - `slack_token`: Slack API token (or from env).
  - `include_metadata`: Include message metadata.
  - `raw_text_output`: Return only text.
  - `result_limit`: Max results.
  - `thread_limit`: Max thread messages.

- **Example:**
  ```python
  from slack_search.searcher import SlackSearcher
  searcher = SlackSearcher(result_limit=5, thread_limit=5)
  results = searcher.search("search query")
  ```

---

## General Notes

- Run ./dev_setup.sh to initialize virtual environment
- Configure API keys via .env

- Each library can be installed and versioned independently.
- Most libraries support both YAML config files and direct argument overrides.
- See each subdirectory’s README for more details and advanced usage.