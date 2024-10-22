# Embedding model

## 1. Selection criteria

There are many selection criteria, such as model performance, processing speed, and vector dimension size. The comparison is mainly based on the following two aspects:

- Huggingface trends and downloads
- Experimental comparison results

## 2 Downloads

Data collection time: 2024.04.18

Ranked by trend (top 5)

| Model | Downloads | Description |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [https://huggingface.co/maidalun1020/bce-embedding-base\_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1 "https://huggingface.co/maidalun1020/bce-embedding-base_v1") | 462K | Strong cross-language capabilities in Chinese and English. Recommended best practices: embedding recalls the top 50-100 segments, reranker re-ranks these 50-100 segments, and finally takes the top 5-10 segments. |
| [https://huggingface.co/Salesforce/SFR-Embedding-Mistral](https://huggingface.co/Salesforce/SFR-Embedding-Mistral "https://huggingface.co/Salesforce/SFR-Embedding-Mistral") | 90k | English |
| [https://huggingface.co/aspire/acge\_text\_embedding](https://huggingface.co/aspire/acge_text_embedding "https://huggingface.co/aspire/acge_text_embedding") | 51K | Chinese, rising rapidly, C-MTEB ranking first (2024.04.18) ([https://huggingface.co/spaces/mteb/leaderboard）](https://huggingface.co/spaces/mteb/leaderboard%EF%BC%89 "https://huggingface.co/spaces/mteb/leaderboard）") |
| [https://huggingface.co/jinaai/jina-embeddings-v2-base-en](https://huggingface.co/jinaai/jina-embeddings-v2-base-en "https://huggingface.co/jinaai/jina-embeddings-v2-base-en") | 934K | English |
| [https://huggingface.co/jinaai/jina-embeddings-v2-base-zh](https://huggingface.co/jinaai/jina-embeddings-v2-base-zh "https://huggingface.co/jinaai/jina-embeddings-v2-base-zh") | 5K | Chinese and English |

Ranked by downloads (several Chinese models are listed at the end)

| Model | Downloads | Description |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- | ------------------------------- |
| [https://huggingface.co/BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3 "https://huggingface.co/BAAI/bge-m3") | 1964K | Multi-language, bge also has three English versions, each download is over 1M |
| [https://huggingface.co/BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5 "https://huggingface.co/BAAI/bge-large-zh-v1.5") | 1882K | Chinese |
| [https://huggingface.co/thenlper/gte-base](https://huggingface.co/thenlper/gte-base "https://huggingface.co/thenlper/gte-base") | 985K | English |
| [https://huggingface.co/jinaai/jina-embeddings-v2-base-en](https://huggingface.co/jinaai/jina-embeddings-v2-base-en "https://huggingface.co/jinaai/jina-embeddings-v2-base-en") | 934K | English |
| [https://huggingface.co/jinaai/jina-embeddings-v2-small-en](https://huggingface.co/jinaai/jina-embeddings-v2-small-en "https://huggingface.co/jinaai/jina-embeddings-v2-small-en") | 495K | English |
| [https://huggingface.co/intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large "https://huggingface.co/intfloat/multilingual-e5-large") | 816K | Multilingual |
| [https://huggingface.co/intfloat/e5-large-v2](https://huggingface.co/intfloat/e5-large-v2 "https://huggingface.co/intfloat/e5-large-v2") | 714K | English |
| [https://huggingface.co/maidalun1020/bce-embedding-base\_v1](https://huggingface.co/maidalun10
