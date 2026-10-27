"""
RAG Pipeline

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_text_file
def load_text_file(path):
    # TODO: read a UTF-8 text file at `path` and return its contents as one string.
    with open(path, mode="r", encoding="utf-8") as file:
        content = file.read()
    return content

# Step 2 - load_text_directory
import glob
def load_text_directory(directory):
    # TODO: read every .txt file in `directory` and return their contents as a list of strings
    search_pattern = f"{directory}/**/*.txt"
    txt_files = glob.glob(search_pattern, recursive=True)
    txt_files.sort()
    texts = [load_text_file(f) for f in txt_files]
    return texts

# Step 3 - extract_text_from_html
from html.parser import HTMLParser


class SmoothTextExtractor(HTMLParser):

  def __init__(self):
    super().__init__()
    self._text_chunks = []
    self._tag_stack = []
    self.ignore_tags = {"script", "style", "head", "title", "meta", "link"}

  def handle_starttag(self, tag, attrs):
    self._tag_stack.append(tag.lower())

  def handle_endtag(self, tag):
    tag = tag.lower()
    if self._tag_stack and self._tag_stack[-1] == tag:
      self._tag_stack.pop()

  def handle_data(self, data):
    is_ignored = any(ignored in self._tag_stack for ignored in self.ignore_tags)

    if not is_ignored:
      # If there is meaningful data, process it
      if data.strip():
        # Handle trailing/leading spaces gracefully to join words naturally
        if (
            data.startswith(" ")
            and self._text_chunks
            and not self._text_chunks[-1].endswith(" ")
        ):
          self._text_chunks.append(" ")

        self._text_chunks.append(data)


def extract_text_from_html(html):
    # TODO: strip HTML tags and return only the visible text content
    extractor = SmoothTextExtractor()
    extractor.feed(html)

    # Join everything, normalize multiple spaces, and strip outer padding
    raw_text = "".join(extractor._text_chunks)
    return " ".join(raw_text.split())

# Step 4 - normalize_text
import re
import unicodedata
def normalize_text(text):
    # TODO: NFKC-normalize the text and collapse runs of whitespace into single spaces.
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Step 5 - make_document
def make_document(text, source, title):
    # TODO: wrap text with source and title metadata into a document dict.

    return {'text': text, 'source': source, 'title': title}

# Step 6 - chunk_fixed_size
def chunk_fixed_size(text, chunk_size):
    # TODO: split text into consecutive non-overlapping chunks of length chunk_size
    # words = text.split()
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Step 7 - chunk_by_tokens
def chunk_by_tokens(text, tokenizer, max_tokens):
    if not text:
        return []
    # TODO: split text into chunks of at most max_tokens token ids using the tokenizer
    tokens = tokenizer.encode(text)
    return [tokenizer.decode(tokens[i:i+max_tokens]) for i in range(0, len(tokens), max_tokens)]

# Step 8 - chunk_by_sentences
import re
def chunk_by_sentences(text, max_chars):
    # TODO: split text on .!? boundaries and greedily pack whole sentences under max_chars.
    # pass
    text = text.strip()
    if not text:
        return []
    chunks = re.split(r"(?<=[.!?])\s+", text)
    i = 0
    answer = []
    while i < len(chunks):
        if len(chunks[i]) > max_chars:
            answer.append(chunks[i])
            i += 1
            continue
        curr = 0
        chunk = []
        while i < len(chunks) and (curr + len(chunks[i]) <= max_chars):
            chunk.append(chunks[i])
            curr += len(chunks[i])
            i += 1
        answer.append(" ".join(chunk))
    return answer

# Step 9 - chunk_with_overlap
def chunk_with_overlap(text, chunk_size, overlap):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Step 10 - attach_chunk_metadata
def attach_chunk_metadata(chunks, source):
    result = []
    for i, chunk in enumerate(chunks):
        result.append({'text': chunk, 'source': source, 'position':i, 'chunk_id': f"{source}::{i}"})
    return result

# Step 11 - load_embedding_model
from sentence_transformers import SentenceTransformer

def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

# Step 12 - embed_text
def embed_text(model, text):
    return model.encode(text)

# Step 13 - embed_chunks
def embed_chunks(model, chunks, batch_size=32):
    """Batch-embed a list of chunk strings or chunk dicts into a 2D float32 matrix."""
    if len(chunks) == 0:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)
    if all(isinstance(item, dict) for item in chunks):
        chunks = [chunk['text'] for chunk in chunks]
    embeddings = model.encode(
        sentences=chunks,
        batch_size=batch_size,             # Process 64 sentences at a time
        convert_to_numpy=True      # Output as a NumPy array
    )
    return embeddings

# Step 14 - l2_normalize
import numpy as np

def l2_normalize(matrix):
    # TODO: rescale each row of `matrix` to unit L2 norm, leaving all-zero rows unchanged.
    l2_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    l2_norm[l2_norm == 0] = 1.0
    return matrix/ l2_norm

# Step 15 - save_corpus
import os
import numpy as np
import json

def save_corpus(embeddings, chunks, directory):
    # TODO: persist embeddings (.npy) and chunks (.json) into directory, then reload and return both.

    emb_path = os.path.join(directory, "embeddings.npy")
    json_path = os.path.join(directory, "chunks.json")
    os.makedirs(directory, exist_ok=True)
    np.save(emb_path, np.asarray(embeddings))
    with open(json_path, "w") as f:
        json.dump(chunks, f)

    loaded_emb = np.load(emb_path)
    with open(json_path, "r") as f:
        loaded_chunks = json.load(f)

    return {"embeddings": loaded_emb, "chunks": loaded_chunks}

# Step 16 - cosine_similarity_search
import numpy as np

def cosine_similarity_search(query_vector, chunk_matrix):
    """Cosine similarity between query_vector (d,) and each row of chunk_matrix (n,d)."""
    # TODO: compute cosine similarity between the query vector and every chunk row
    vector_norm = np.linalg.norm(query_vector)
    matrix_norm = np.linalg.norm(chunk_matrix, axis=1)
    return (chunk_matrix @ query_vector) / (vector_norm*matrix_norm)

# Step 17 - top_k_indices
import numpy as np

def top_k_indices(scores, k):
    """Return indices of the k highest scores in descending order."""
    topk = np.argsort(-scores)[:k]
    return topk

# Step 18 - top_k_chunks
import numpy as np

def top_k_chunks(scores, chunks, k):
    # TODO: return list of (chunk, score) tuples for the top-k scores, sorted descending
    topk_indices = top_k_indices(scores, k)
    answer = []
    for i in topk_indices:
        answer.append((chunks[i], scores[i].item()))
    return answer

# Step 19 - retrieve
def retrieve(query, model, chunk_matrix, chunks, k):
    embedding = embed_text(model, query)
    scores = cosine_similarity_search(embedding, chunk_matrix)
    topk_chunks = top_k_chunks(scores, chunks, k)
    return topk_chunks

# Step 20 - build_faiss_index
def build_faiss_index(chunk_matrix):
    dim = chunk_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(chunk_matrix)
    return index

# Step 21 - faiss_search
import numpy as np
import faiss

def faiss_search(index, query_vector, k):
    """Return top-k (scores, indices) as 1D arrays for a single query vector."""
    # TODO: query the FAISS index with the single query vector and return flat top-k arrays
    q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    n = min(index.ntotal, k+1)
    scores, indices = index.search(q, n)
    scores, indices = scores[0], indices[0]
    order = np.lexsort((indices, -scores))[:k]
    return scores[order].astype(np.float32), indices[order].astype(np.int64)

# Step 22 - compare_faiss_to_numpy
def compare_faiss_to_numpy(query_vector, chunk_matrix, index, k):
    # TODO: return True iff FAISS and numpy cosine search agree on the top-k indices
    css = cosine_similarity_search(query_vector, chunk_matrix)
    css_topk = top_k_indices(css, k)
    faiss_index = build_faiss_index(chunk_matrix)
    _, faiss_topk = faiss_search(faiss_index, query_vector, k)
    return np.array_equal(faiss_topk, css_topk)

# Step 23 - save_faiss_index
def save_faiss_index(index, path):
    """Write `index` to `path` and return the index loaded back from disk."""
    # TODO: persist the index to `path` and reload it; return the reloaded index
    os.makedirs(path, exist_ok=True)
    faiss.write_index(index, f"{path}/faiss_index.bin")

    # Load the index back from disk later
    index = faiss.read_index(f"{path}/faiss_index.bin")
    return index

# Step 24 - build_prompt_template
def build_prompt_template():
    # TODO: return a RAG prompt template string with {context} and {question} placeholders.
    return (
        "Context:\n{context}\n\n"
        "Answer the question using only the context above. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Question: {question}\n"
        "Answer:"
    )

# Step 25 - format_context
def format_context(retrieved):
    # TODO: render each (chunk, score) as '[i] {text} (source={source})' and join with newlines
    context = ""
    for i, (d, s) in enumerate(retrieved):
        context += f"[{i+1}] {d['text']} (source={d['source']})"
        if i != len(retrieved)-1:
            context += "\n"
    return context

# Step 26 - truncate_context
def truncate_context(context, max_chars):
    if len(context) <= max_chars:
        return context
    # Look one char past the limit, so a space right at the boundary counts
    window = context[:max_chars + 1]
    cut = max(window.rfind(' '), window.rfind('\n'), window.rfind('\t'))
    if cut > 0:
        return context[:cut].rstrip()
    return context[:max_chars]  # no whitespace: hard cut

# Step 27 - add_system_instruction
def add_system_instruction(prompt):
    """Prepend a fixed system instruction to the prompt."""
    system = (
        "You are a helpful assistant. Answer the question using ONLY the "
        "provided context. If the answer is not in the context, say 'I do not know'."
    )
    return system + "\n\n" + prompt

# Step 28 - load_generator
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_generator(model_name='sshleifer/tiny-gpt2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # GPT-2 has no pad token by default, so reuse the end-of-sequence token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    model.eval()  # inference mode (turns off dropout)
    return model, tokenizer

# Step 29 - generate_answer
import torch

def generate_answer(model, tokenizer, prompt, max_new_tokens=32):
    torch.manual_seed(0)
    inputs = tokenizer(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                       # greedy
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output[0][inputs['input_ids'].shape[1]:]   # drop the prompt
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# Step 30 - rag_answer
def rag_answer(query, chunks, embeddings, embed_model, generator, tokenizer, k=3):
    query_vec = embed_text(embed_model, query)
    scores = cosine_similarity_search(query_vec, embeddings)
    retrieved = top_k_chunks(scores, chunks, k)

    context = format_context(retrieved)
    prompt = build_prompt_template().format(context=context, question=query)
    answer = generate_answer(generator, tokenizer, prompt)

    sources = [c for c, s in retrieved]
    return {'answer': answer, 'sources': sources, 'query': query}

# Step 31 - track_source_chunk_ids
def track_source_chunk_ids(source_chunks):
    # TODO: return the list of chunk ids from the retrieved source chunks, preserving order
    return [d['id'] for d in source_chunks if 'id' in d]

# Step 32 - append_source_references
def append_source_references(answer_text, source_chunks):
    sources = track_source_chunk_ids(source_chunks)
    return f"{answer_text}\nSources: [{', '.join(sources)}]"

# Step 33 - query_rewrite
FILLERS = [
    'could you please', 'can you please', 'could you', 'can you',
    'please tell me', 'tell me', 'i want to know', 'i would like to know',
    "i'd like to know", 'please', 'kindly',
]

def query_rewrite(raw_query):
    q = normalize_text(raw_query).lower()
    changed = True
    while changed:
        changed = False
        for f in FILLERS:
            if q == f:
                q = ''
                break
            if q.startswith(f + ''):
                q = q[len(f) + 1:]
                changed = True
                break
    return q.rstrip('?.! ')

# Step 34 - hyde_retrieve
def hyde_retrieve(query, hypothetical_answer, chunks, embeddings, embed_model, k=5):
    # TODO: embed the hypothetical answer and return the top-k chunks by cosine similarity.
    q = np.asarray(embed_text(embed_model, hypothetical_answer), dtype=np.float32)
    scores = cosine_similarity_search(q, embeddings)
    topk_chunks = top_k_chunks(scores, chunks, k)
    return [id_ for id_, s in topk_chunks]

# Step 35 - reciprocal_rank_fusion
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Step 36 - bm25_search
import math
from collections import Counter
def bm25_search(query, chunks, k=5, k1=1.5, b=0.75):
    q_terms = query.lower().split()
    docs = [c['text'].lower().split() for c in chunks]
    N = len(docs)
    if N == 0 or not q_terms:
        return []
    avgdl = sum(len(d) for d in docs) / N
    df = Counter()
    for d in docs:
        df.update(set(d))
        
    results = []
    for i, d in enumerate(docs):
        tf = Counter(d)
        dl = len(d)
        score = 0.0
        matched = False
        for t in q_terms:
            f = tf.get(t, 0)
            if f == 0:
                continue
            matched = True
            idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
            score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        if matched:
            results.append((i, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]

# Step 37 - hybrid_search
def minmax_normalize(vec):
    vec = np.asarray(vec, dtype=float)
    small, large = vec.min(), vec.max()
    if large - small == 0:
        return np.zeros_like(vec)
    return (vec - small) / (large - small)

def hybrid_search(query, chunks, embeddings, embed_model, alpha=0.5, k=5):
    n = len(chunks)

    q_vec = np.asarray(embed_text(embed_model, query), dtype=float)
    dense = minmax_normalize(cosine_similarity_search(q_vec, embeddings))

    bm25_raw = np.zeros(n)
    for idx, s in bm25_search(query, chunks, k=n):
        bm25_raw[idx] = s
    sparse = minmax_normalize(bm25_raw)

    combined = alpha * dense + (1 - alpha) * sparse
    order = sorted(range(n), key=lambda i: -combined[i])   # stable → ties keep chunk order
    return [(i, float(combined[i])) for i in order[:k]]

# Step 38 - rerank_cross_encoder
def rerank_cross_encoder(query, candidate_chunks, cross_encoder):
    if not candidate_chunks:
        return []
    pairs = [(query, c['text']) for c in candidate_chunks]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, s in ranked]

# Step 39 - maximal_marginal_relevance
def maximal_marginal_relevance(query_embedding, candidate_embeddings, k=5, lambda_param=0.5):
    C = np.asarray(candidate_embeddings, dtype=float)
    q = np.asarray(query_embedding, dtype=float)
    N = len(C)
    if N == 0 or k <= 0:
        return []
    q = q / (np.linalg.norm(q) or 1.0)

    relevance = C @ q          # (N,) similarity to query
    sim = C @ C.T              # (N, N) candidate-to-candidate similarity

    selected = []
    remaining = list(range(N))
    while remaining and len(selected) < k:
        best_idx, best_score = None, -np.inf
        for i in remaining:                      # ascending order
            redundancy = max(sim[i, j] for j in selected) if selected else 0.0
            score = lambda_param * relevance[i] - (1 - lambda_param) * redundancy
            if score > best_score:               # strict > → ties keep smaller index
                best_idx, best_score = i, score
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected

