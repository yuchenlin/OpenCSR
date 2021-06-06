"""BM25 search for answering."""
import json

from absl import app, flags
from elasticsearch import Elasticsearch
from tqdm import tqdm
import six.moves.urllib as urllib

import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True
})

FLAGS = flags.FLAGS
flags.DEFINE_string("linked_qa_file", "",
                    "The path to the linked qa files.")

es = Elasticsearch()
es = Elasticsearch(port=9200, timeout=30)


def get_top_k(query, k):
  """Get the top-k results from the inedx."""
  results = es.search(index='gkb_best_facts_new', params={
                      "q": query, "size": k})['hits']['hits']
  facts = []
  scores = []
  for res in results:
    scores.append(float(res['_score']))
    facts.append(res['_source']['doc'])
  return facts, scores


def save_BM25_hits():
  """Save the retrieved results."""
  with open(FLAGS.linked_qa_file) as f:
    instances = [json.loads(line) for line in f.read().split("\n") if line]

  num_questions = len(instances)
  thresholds = [
      10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700,
      800, 900, 1000]
  # num_first_position = [0] * len(thresholds)
  # num_last_position = [0] * len(thresholds)
  result_data = []
  # num_top_correct = 0
  for ins in tqdm(instances[:], desc=FLAGS.linked_qa_file):
    q = ins["question"]
    q = urllib.parse.quote_plus(q) 
    ret_facts, ret_scores = get_top_k(q, k=thresholds[-1])
    all_ret_facts = [(f["id"], s) for f, s in zip(ret_facts, ret_scores)]
    
    BM25_res = {}    
    BM25_res["all_ret_facts"] = all_ret_facts
    ins["results"] = BM25_res

    result_data.append(ins)

  with open(FLAGS.linked_qa_file.replace(".jsonl", "")+".BM25.jsonl", "w") as f:
    f.write("\n".join([json.dumps(r) for r in result_data]))
  return 


def main(_):
  """Run the BM25 search."""
  save_BM25_hits()
  
if __name__ == "__main__":
  app.run(main)
