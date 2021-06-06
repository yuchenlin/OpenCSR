# Lint as: python3
"""Adds pre-computed results as initial facts for OpenCSR data."""
import json

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
from language.labs.drfact import index_corpus

FLAGS = flags.FLAGS

# flags.DEFINE_string("concept_vocab_file", "drfact_data/knowledge_corpus/gkb_best.vocab.txt", "Path to dataset file.")
flags.DEFINE_string("linked_qas_file", None, "Path to dataset file.")
flags.DEFINE_string("drfact_format_gkb_file", None, "Path to gkb corpus.")
flags.DEFINE_string("ret_result_file", None, "Path to dataset file.")
flags.DEFINE_string("output_file", None, "Path to dataset file.")
flags.DEFINE_string("split", "train", "Path to dataset file.")
flags.DEFINE_integer("max_num_facts", 10, "Path to dataset file.")


COMMON_CONCEPTS = ["make", "cause", "factor",
                   "person", "need", "use", "people", "part", "system"]


def main(_):
  """Main funciton."""
  logging.set_verbosity(logging.INFO)

  concept2id = index_corpus.load_concept_vocab(FLAGS.concept_vocab_file)

  with open(FLAGS.drfact_format_gkb_file) as f:
    logging.info("Reading %s..." % f.name)
    gkb_id_to_id = {}
    facts_dict = {}
    cur_fact_ind = 0
    for line in f.read().split("\n"):
      if line:
        instance = json.loads(line)
        gkb_id_to_id[instance["id"]] = cur_fact_ind
        facts_dict[instance["id"]] = instance
        cur_fact_ind += 1

  with open(FLAGS.ret_result_file) as f:
    logging.info("Reading %s..." % f.name)
    ret_data = [json.loads(line) for line in f.read().split("\n") if line]

  logging.info("Reading QAS(-formatted) data...")
  with open(FLAGS.linked_qas_file) as f:
    jsonlines = f.read().split("\n")
  data = [json.loads(jsonline) for jsonline in jsonlines if jsonline]

  assert len(ret_data) == len(data)

  new_data = []
  num_covered = 0
  num_2nd = 0
  for ind, ins in tqdm(
          enumerate(data),
          desc=FLAGS.linked_qas_file, total=len(data)):
    all_ret_facts = ret_data[ind]["results"]["all_ret_facts"]
    ins["sup_facts"] = []
    question_concepts = set([c["kb_id"]
                             for c in ins["entities"]]) - set(COMMON_CONCEPTS)
    # TODO: decomp?
    answer_concepts = set([c["kb_id"] for c in ins["all_answer_concepts"]]
                          ) - set(COMMON_CONCEPTS) - question_concepts
    is_covered = False
    concept_set = set()
    original_rank = 0
    question_only_facts = []
    answer_only_facts = []
    # first round 
    for fid, s in all_ret_facts:
      original_rank += 1
      fact = facts_dict[fid]
      fact_concepts = set([m["kb_id"] for m in fact["mentions"]])
      contain_question = fact_concepts & question_concepts
      contain_answer = fact_concepts & answer_concepts
      if contain_question and not contain_answer:
        question_only_facts.append((gkb_id_to_id[fid], s, fact["context"]))
        continue
      if contain_answer and not contain_question:
        answer_only_facts.append((gkb_id_to_id[fid], s, fact["context"]))
        continue
      if contain_answer and contain_question:
        ins["sup_facts"].append((gkb_id_to_id[fid], s, fact["context"]))
      if len(ins["sup_facts"]) >= FLAGS.max_num_facts:
        break
      concept_set.update(fact_concepts)
    
    if len(answer_only_facts) + len(ins["sup_facts"]) <=10:
      # second round
      is_covered = False
      concept_set = set()
      original_rank = 0
      question_only_facts = []
      answer_only_facts = []
      num_2nd += 1
      answer_concepts = set([c["kb_id"] for c in ins["all_answer_concepts_decomp"]]
                          ) - set(COMMON_CONCEPTS) - question_concepts
      for fid, s in all_ret_facts:
        original_rank += 1
        fact = facts_dict[fid]
        fact_concepts = set([m["kb_id"] for m in fact["mentions"]])
        contain_question = fact_concepts & question_concepts
        contain_answer = fact_concepts & answer_concepts
        if contain_question and not contain_answer:
          question_only_facts.append((gkb_id_to_id[fid], s, fact["context"]))
          continue
        if contain_answer and not contain_question:
          answer_only_facts.append((gkb_id_to_id[fid], s, fact["context"]))
          continue
        if contain_answer and contain_question:
          ins["sup_facts"].append((gkb_id_to_id[fid], s, fact["context"]))
        if len(ins["sup_facts"]) >= FLAGS.max_num_facts:
          break
        if len(ins["sup_facts"]) > 1:
          is_covered = True
        concept_set.update(fact_concepts)


    if len(ins["sup_facts"]) > 1:
        is_covered = True 
    ins["answer_only_facts"] = answer_only_facts[:FLAGS.max_num_facts]
    ins["question_only_facts"] = question_only_facts[:FLAGS.max_num_facts]
    ins["sup_facts_source"] = FLAGS.ret_result_file
    # del ins["all_ret_facts"]
    if is_covered:
      num_covered += 1
    ins["num_mentioned_concepts"] = len(concept_set)
    new_data.append(ins)
  with open(FLAGS.output_file, "w") as f:
    logging.info("num_2nd: %d", num_2nd)
    logging.info("num_covered: %d", num_covered)
    logging.info("len(new_data): %d", len(new_data))
    logging.info("Coverage:%.2f", num_covered/len(new_data))
    logging.info("Writing to %s", f.name)
    f.write("\n".join([json.dumps(i) for i in new_data])+"\n")
  logging.info("Done.")


if __name__ == "__main__":
  app.run(main)
