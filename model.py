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

