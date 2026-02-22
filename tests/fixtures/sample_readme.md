# SampleProject

A retrieval augmented generation framework for building question answering
systems with long context transformers.

## Features

- Document chunking and embedding
- Vector store integration
- Transformer-based reranking
- Streaming response generation

## Architecture

SampleProject uses a retriever-reader pipeline. Documents are first chunked
and indexed using dense embeddings, then a cross-encoder reranker selects
the most relevant passages. A language model generates the final answer
conditioned on the retrieved context.

## Installation

```bash
pip install sampleproject
```
