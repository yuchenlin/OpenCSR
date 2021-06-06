"""Indexing the corpus with Elasticsearch."""
import json
import sys

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm


def simple_index(drfact_format_corpus_path):
  """Loads corpus data."""
  with open(drfact_format_corpus_path) as f:
    print("Reading the corpus from %s ..." % f.name)
    jsonlines = f.read().split("\n")
  corpus_data = [json.loads(jsonline) for jsonline in jsonlines if jsonline]

  es = Elasticsearch(port=9200, timeout=30)
  for ind, fact in tqdm(
          enumerate(corpus_data),
          desc="Indexing the corpus.", total=len(corpus_data)):
    es.index(index='gkb_best_new', doc_type='fact', id=ind, body=fact)


def bulk_index(filename_prefix):
  """Indexing with bulks."""
  es = Elasticsearch(port=9200, timeout=30)
  print(es.indices.create(index='gkb_best_facts', ignore=400))

  def bulk_gen(filename):
    with open(filename, encoding='utf-8') as f:
      for line in tqdm(f.read().split('\n')[:], desc=filename):
        if line:
          fact = json.loads(line)
          yield {
              "_index": "gkb_best_facts_new",
              "_type": "document",
              "doc": fact,
          }
  for i in range(0, 21):
    print("Number:", i)
    filename_template = filename_prefix + "{:0>2d}"
    bulk(es, bulk_gen(filename_template.format(i)))


# simple_index(drfact_format_corpus_path=sys.argv[1])
bulk_index(filename_prefix=sys.argv[1])
