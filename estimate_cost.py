import os
import random
import subprocess
import zipfile
import re

def get_text_docx(path):
    try:
        with zipfile.ZipFile(path) as z:
            xml_content = z.read('word/document.xml').decode('utf-8')
            # very crude XML tag removal
            text = re.sub('<[^>]+>', ' ', xml_content)
            return text
    except:
        return ""

def get_text_doc(path):
    try:
        # crude text extraction from binary .doc using strings
        result = subprocess.run(['strings', path], capture_output=True, text=True, errors='ignore')
        return result.stdout
    except:
        return ""

folder = '/Users/danielabueno/Downloads/Word'
docs = []
docxs = []

for root, _, files in os.walk(folder):
    for f in files:
        low = f.lower()
        path = os.path.join(root, f)
        if low.endswith('.doc'):
            docs.append(path)
        elif low.endswith('.docx'):
            docxs.append(path)

print(f"Found {len(docs)} .doc files and {len(docxs)} .docx files.")

# To get a good average without taking too long, sample 200 files
random.seed(42)
sample_docs = random.sample(docs, min(200, len(docs)))
sample_docxs = random.sample(docxs, min(200, len(docxs)))

print("Extracting text from samples to estimate size...")
total_chars_docs = sum(len(get_text_doc(p)) for p in sample_docs)
total_chars_docxs = sum(len(get_text_docx(p)) for p in sample_docxs)

avg_chars_doc = total_chars_docs / len(sample_docs) if sample_docs else 0
avg_chars_docx = total_chars_docxs / len(sample_docxs) if sample_docxs else 0

total_chars_estim = (avg_chars_doc * len(docs)) + (avg_chars_docx * len(docxs))

# roughly 4 characters per token
tokens_estim = total_chars_estim / 4

cost_large = (tokens_estim / 1_000_000) * 0.13
cost_small = (tokens_estim / 1_000_000) * 0.02

print("-" * 40)
print(f"Average characters per .doc  : {avg_chars_doc:,.0f} (sample: {len(sample_docs)})")
print(f"Average characters per .docx : {avg_chars_docx:,.0f} (sample: {len(sample_docxs)})")
print(f"Estimated total characters   : {total_chars_estim:,.0f}")
print(f"Estimated total tokens       : {tokens_estim:,.0f}")
print("-" * 40)
print(f"Estimated Cost (text-embedding-3-large): ${cost_large:,.4f}")
print(f"Estimated Cost (text-embedding-3-small): ${cost_small:,.4f}")
