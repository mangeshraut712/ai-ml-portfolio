# Sarvam AI Overview
Sarvam AI is an India-first sovereign AI company building multilingual models for Indian languages.
Its products include speech recognition, text-to-speech, translation, and generative models tuned for Indic scripts.

# Bulbul TTS
Bulbul is Sarvam's text-to-speech family. Latent space decomposition helps separate speech content from speaker and style representations, enabling controllable voice generation.

# Saaras STT
Saaras is Sarvam's speech-to-text offering. It targets low-latency transcription across Indian languages and dialects with production constraints such as p95 latency budgets under 800ms for interactive systems.

# Indic Multilingual Challenges
Indian NLP faces code-mixing, dialect variation, and script diversity. Evaluation should report per-language metrics rather than a single English-centric score.

# RAG Best Practices
Retrieval-Augmented Generation grounds answers in documents. Measure Recall@k and MRR for retrieval, then faithfulness of generated answers to retrieved context. Prefer saying I don't know when evidence is missing.

# Embedding Models
Dense embeddings map text to vectors for semantic search. Benchmark embedding models with retrieval metrics on a fixed gold query set. Cosine similarity is the common scoring function.

# Reranking
A first-stage retriever returns top-k candidates cheaply. A reranker reorders them with a cross-encoder or score fusion to improve precision at small k.

# Hallucination Risks
LLMs may invent facts not present in context. Adversarial evaluation uses unanswerable questions. A faithful system should abstain rather than fabricate.

# Latency and Cost
Production RAG systems trade quality for latency and dollar cost. Report p50/p95 latency and estimated cost per 1000 queries using token pricing tables.

# Classical ML vs LLMs
Interview loops may pivot from LLMs to classical ML: gradient descent, regularization, bias-variance, ROC-AUC, and tree ensembles remain foundational for applied AI roles.
