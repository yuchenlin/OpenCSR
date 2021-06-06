# Lint as: python3
"""Preprocess the corpus with spacy."""

import spacy
import json
import os

from absl import app
from absl import flags
from tqdm import tqdm 

nlp = spacy.load('en_core_web_lg')
nlp.pipeline = [('tagger', nlp.tagger), ('parser', nlp.parser)]
# Disable other components for speed.

FLAGS = flags.FLAGS

flags.DEFINE_string("DATA_ROOT", None, "Path to root folder.")
flags.DEFINE_string("CORPUS_PATH", None, "Path to corpus file.")
flags.DEFINE_string("OUTPUT_JSON_PATH", None, "Path to output json file.")

def formatter_gkbcorpus(filepath):
  with open(filepath) as f:
    lines = f.read().split("\n")
  gkb_corpus = {}
  sentence_set = set()
  print("# sentence (original):", len(lines))
  for line in lines[1:-1]:
    index = len(gkb_corpus)
    sent_id = "gkb-best#%d"%index
    source, term, quantifier, sent, score = line.split("\t")
    sent = sent.replace(" isa ", " is a ")
    sent = sent.replace(" have (part) ", " have ")
    sent = sent.replace(" has-part ", " have ")
    if sent.lower() in sentence_set:
        continue
    sentence_set.add(sent.lower())
    remark = dict(source=source, title=term, quantifier=quantifier, score=float(score))
    gkb_corpus[sent_id]=dict(sent_id=sent_id, sentence=sent, remark=remark )
  return gkb_corpus



def main(_):
  gkb_corpus = formatter_gkbcorpus(os.path.join(FLAGS.DATA_ROOT, FLAGS.CORPUS_PATH))
  sentences = [item["sentence"] for sent_id, item in gkb_corpus.items()]
  print("# sentence (unique):", len(sentences))
  docs = nlp.pipe(sentences)  # multi-threading
  results = []
  for index, doc in tqdm(enumerate(docs), total=len(sentences)):
    sent_id = "gkb-best#%d"%index
    tokens = [t.text for t in doc]
    pos_tags = [t.pos_ for t in doc]
    lemmas = [t.lemma_ for t in doc]
    noun_chunks = []
    for chunk in doc.noun_chunks:
      chunk_text = chunk.text.split()
      start = chunk.start
      end = chunk.end
      while start < end and pos_tags[start] not in ["NOUN", "ADJ", "PROPN"]:
          start += 1
      if start == end:
          continue
      chunk_text = " ".join(lemmas[start:end])
      noun_chunks.append((chunk_text, start, end))
    gkb_corpus[sent_id]["tokens"] = tokens
    gkb_corpus[sent_id]["pos_tags"] = pos_tags
    gkb_corpus[sent_id]["lemmas"] = lemmas
    gkb_corpus[sent_id]["noun_chunks"] = noun_chunks


  json_lines = [json.dumps(item) for _, item in gkb_corpus.items()]
  with open(os.path.join(FLAGS.DATA_ROOT, FLAGS.OUTPUT_JSON_PATH), "w") as f:
      f.write("\n".join(json_lines))


if __name__ == "__main__":
  app.run(main)
