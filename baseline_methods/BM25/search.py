"""BM25 search for answering."""
import json

from absl import app, flags
from elasticsearch import Elasticsearch
from tqdm import tqdm
import six.moves.urllib as urllib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import collections
import numpy as np

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


def evaluate_BM25_hits():
  """Evaluates the retrieved results."""
  with open(FLAGS.linked_qa_file) as f:
    instances = [json.loads(line) for line in f.read().split("\n") if line]

  num_questions = len(instances)
  thresholds = [
      10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700,
      800, 900, 1000]
  num_first_position = [0] * len(thresholds)
  num_last_position = [0] * len(thresholds)
  result_data = []
  # num_top_correct = 0
  for ins in tqdm(instances[:], desc=FLAGS.linked_qa_file):
    q = ins["question"]
    q = urllib.parse.quote_plus(q) 
    answer_concepts = set([a["name"] for a in ins["answer_concepts"]])
    ret_facts, ret_scores = get_top_k(q, k=thresholds[-1])
    all_ret_facts = [(f["id"], s) for f, s in zip(ret_facts, ret_scores)]
    # supporting_ret_facts = []
    covered_concepts = set()
    first_answer_conept_position = -1
    last_answer_concept_position = -1
    ret_concept_scores = collections.defaultdict(list)
    for ind, fact in enumerate(ret_facts):
      for m in fact["mentions"]:
        c = m["kb_id"]
        ret_concept_scores[c].append(ret_scores[ind])
        if c in answer_concepts:
          # Found an answer concept.
          # supporting_ret_facts.append(
          #     {"fact_id": fact["id"],
          #      "covered_answer_concept": c, "hit_position": ind})
          if len(covered_concepts) == 0:
            first_answer_conept_position = ind
          if c not in covered_concepts:
            covered_concepts.add(c)
          if covered_concepts == answer_concepts and last_answer_concept_position < 0:
            last_answer_concept_position = ind
            break
    BM25_res = {}
    BM25_res["covered_concepts"] = list(covered_concepts)
    # BM25_res["supporting_ret_facts"] = supporting_ret_facts
    BM25_res["first_answer_conept_position"] = first_answer_conept_position
    BM25_res["last_answer_concept_position"] = last_answer_concept_position
    # Rank ret_concepts by reduce_sum
    ret_concept_scores_sum = {c: sum(scores)
                              for c, scores in ret_concept_scores.items()}
    ret_concept_scores_mean = {
        c: np.mean(scores) for c, scores in ret_concept_scores.items()}
    ret_concept_scores_max = {
        c: max(scores) for c, scores in ret_concept_scores.items()}
    BM25_res["ret_concept_scores_sum"] = [(k, v) for k, v in sorted(
        ret_concept_scores_sum.items(), key=lambda x: x[1], reverse=True)]
    BM25_res["ret_concept_scores_mean"] = [(k, v) for k, v in sorted(
        ret_concept_scores_mean.items(), key=lambda x: x[1], reverse=True)]
    BM25_res["ret_concept_scores_max"] = [(k, v) for k, v in sorted(
        ret_concept_scores_max.items(), key=lambda x: x[1], reverse=True)]
    BM25_res["all_ret_facts"] = all_ret_facts
    ins["results"] = BM25_res

    result_data.append(ins)

  with open(FLAGS.linked_qa_file+".BM25.jsonl", "w") as f:
    f.write("\n".join([json.dumps(r) for r in result_data]))
  return 


def main(_):
  """Run the BM25 search."""
  evaluate_BM25_hits()
  
if __name__ == "__main__":
  app.run(main)
