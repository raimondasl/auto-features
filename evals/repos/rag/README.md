# RagKit

A retrieval-augmented generation framework for building question-answering
systems over your own documents. RagKit indexes a corpus with dense passage
retrieval, runs late-interaction reranking, and feeds the retrieved passages to
a large language model to generate grounded, citation-backed answers.

## Features

- **Dense retrieval** — embed passages with sentence-transformers and search
  with FAISS for fast approximate nearest-neighbor lookup.
- **Late-interaction reranking** — optional ColBERT-style token-level reranking
  of the top candidates for higher retrieval precision.
- **Hybrid search** — fuse BM25 lexical scores with dense similarity via
  reciprocal rank fusion.
- **Grounded generation** — assemble retrieved context into a prompt and call an
  LLM to produce answers with inline source citations.
- **Open-domain QA** — evaluate on Natural Questions and TriviaQA.

## Quick start

```python
from ragkit import Index, Retriever, Generator

index = Index.from_documents("./docs")
retriever = Retriever(index, reranker="colbert")
generator = Generator(model="gpt-4o-mini")

answer = generator.answer("What is retrieval augmented generation?", retriever)
print(answer.text, answer.citations)
```

RagKit is designed for neural information retrieval research and production RAG
pipelines alike.
