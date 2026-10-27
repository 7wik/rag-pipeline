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

