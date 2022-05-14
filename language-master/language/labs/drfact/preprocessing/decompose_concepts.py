"""Converts the qas format json to csv."""
import json
from absl import app
from absl import flags
from language.labs.drfact import index_corpus

FLAGS = flags.FLAGS

flags.DEFINE_string("linked_qas_file", "drfact_data/datasets/ARC/linked_dev.jsonl", "Path to dataset file.")
flags.DEFINE_string("jsonl_file", "drfact_data/datasets/ARC/dev.jsonl", "Path to original file.")
flags.DEFINE_string("CONCEPT_VOCAB", "drfact_data/knowledge_corpus/gkb_best.vocab.txt", "Path to original file.")

def main(_):
  concept2id = index_corpus.load_concept_vocab(FLAGS.CONCEPT_VOCAB)
  with open(FLAGS.jsonl_file) as f:
    lines = f.read().splitlines()
    instances = [json.loads(l) for l in lines]

  with open(FLAGS.linked_qas_file) as f:
    lines = f.read().splitlines()
    linked_instances = [json.loads(l) for l in lines]

  assert len(instances) == len(linked_instances)

  all_answer_concepts_decomp = {}
  for item in instances:
    qid = item["id"]
    truth_choice = ""
    if item["answerKey"] in "ABCDEFGH":
      truth_choice = ord(item["answerKey"]) - ord("A")  # e.g., A-->0
    elif item["answerKey"] in "12345":
      truth_choice = int(item["answerKey"]) - 1
    choices = item["question"]["choices"]
    assert choices[truth_choice]["label"] == item["answerKey"]
    correct_answer_lemmas = choices[truth_choice]["lemmas"]
    matched_concepts = index_corpus.simple_match(
        correct_answer_lemmas, concept_vocab=concept2id, max_len=4,
        disable_overlap=False)
    all_answer_concepts_decomp[qid] = [{
        "kb_id": c["mention"],
        "name": c["mention"]
    } for c in matched_concepts]

  for item in linked_instances:
    qid = item["_id"]
    item["all_answer_concepts_decomp"] = all_answer_concepts_decomp[qid]

  with open(FLAGS.linked_qas_file, "w") as f:
    for item in linked_instances:
      f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
  app.run(main)
