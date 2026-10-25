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

