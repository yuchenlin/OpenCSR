# Lint as: python3
"""Adds pre-computed results as initial facts for OpenCSR data."""
import json

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("linked_qas_file", None, "Path to dataset file.")
flags.DEFINE_string("drfact_format_gkb_file", None, "Path to gkb corpus.")
flags.DEFINE_string("ret_result_file", None, "Path to dataset file.")
flags.DEFINE_string("sup_facts_file", None, "Path to dataset file.")
flags.DEFINE_string("output_file", None, "Path to dataset file.")
flags.DEFINE_string("split", "train", "Path to dataset file.")
flags.DEFINE_integer("max_num_facts", 1000, "Path to dataset file.")

COMMON_CONCEPTS = ["make", "cause", "factor", "person", "need", "use", "people", "part", "system"]

def main(_):
  """Main funciton."""
  logging.set_verbosity(logging.INFO)

  with open(FLAGS.drfact_format_gkb_file) as f:
    logging.info("Reading %s..."%f.name)
    gkb_id_to_id = {}
    facts_dict = {}
    cur_fact_ind = 0
    for line in f.read().splitlines(): 
      instance = json.loads(line)
      gkb_id_to_id[instance["id"]] = cur_fact_ind
      facts_dict[instance["id"]] = instance
      cur_fact_ind += 1
  
  with open(FLAGS.ret_result_file) as f:
    logging.info("Reading %s..."%f.name)
    ret_data = [json.loads(line) for line in f.read().split("\n") if line]

  with open(FLAGS.linked_qas_file) as f:
    logging.info("Reading QAS(-formatted) data...%s"%f.name)
    jsonlines = f.read().splitlines()
  data = [json.loads(jsonline) for jsonline in jsonlines]

  assert len(ret_data) == len(data)

  sup_facts_data = []
  if FLAGS.sup_facts_file:
    with open(FLAGS.sup_facts_file) as f:
      jsonlines = f.read().splitlines()
    sup_facts_data = [json.loads(jsonline) for jsonline in jsonlines]
    assert len(data) == len(sup_facts_data)

  new_data = []
  num_covered = 0
  for ind, ins in tqdm(enumerate(data), desc=FLAGS.linked_qas_file, total=len(data)):
    all_ret_facts = ret_data[ind]["results"]["all_ret_facts"]
    ins["init_facts"] = []
    ins["sup_facts"] = [[], []]
    # question_concepts = set([c["kb_id"] for c in ins["entities"]])
    answer_concepts = set([c["kb_id"] for c in ins["all_answer_concepts"]])
    question_concepts = set([c["kb_id"] for c in ins["entities"]]) - set(COMMON_CONCEPTS)
    is_covered = False
    concept_set = set()
    for fid, s in all_ret_facts:
      fact = facts_dict[fid]
      fact_concepts = set([m["kb_id"] for m in fact["mentions"]])
      
      # TODO: this is equvilent to dense_first and then sparse
      # if FLAGS.split == "train":
      contain_answer = False
      if fact_concepts & question_concepts:
        # keep only question-mentioned facts as the first hop 
        ins["init_facts"].append((gkb_id_to_id[fid], s))
        contain_answer = fact_concepts & answer_concepts
        # elif fact_concepts & answer_concepts:
        #   # not mention question concept but mention answer 
        #   # answer only concepts
        #   ins["sup_facts"][0].append((gkb_id_to_id[fid], s))
        #   ins["sup_facts"][1].append((gkb_id_to_id[fid], s))
        # else:
        #   ins["init_facts"].append((gkb_id_to_id[fid], s))
          # continue
      # else:
      #   ins["init_facts"].append((gkb_id_to_id[fid], s))

      #   # if len(ins["init_facts"]) < FLAGS.max_num_facts or contain_answer:  # Cause some problems
      # else:
      #   if len(ins["init_facts"]) < FLAGS.max_num_facts:
      #     ins["init_facts"].append((gkb_id_to_id[fid], s)) 
      if len(ins["init_facts"]) >= FLAGS.max_num_facts:
          break
      if contain_answer:
        is_covered = True
      concept_set.update(fact_concepts)
    if is_covered:
      num_covered += 1
    ins["num_mentioned_concepts"] = len(concept_set) 
    init_fact_set = set([fid for fid, _ in ins["init_facts"]])
    if FLAGS.split == "train" and sup_facts_data:
      for sup_item in sup_facts_data[ind]["sup_facts"]:
        item_supfacts = sup_item[0] # a list of facts
        if len(item_supfacts) == 1: # One-hop quesiton
          fid = item_supfacts[0][0]
          score = item_supfacts[0][1]
          ins["sup_facts"][0].append((fid, score))
          # TODO: put the same fact to the second hop slot
          ins["sup_facts"][1].append((fid, score))
          if fid not in init_fact_set:
            ins["init_facts"].append((fid, score))
            init_fact_set.add(fid)
        elif len(item_supfacts) == 2: # Two-hop quesiton
          fid_1 = item_supfacts[0][0]
          score_1 = item_supfacts[0][1]
          fid_2 = item_supfacts[1][0]
          score_2 = item_supfacts[1][1]
          ins["sup_facts"][0].append((fid_1, score_1))
          ins["sup_facts"][1].append((fid_2, score_2))
          if fid_1 not in init_fact_set:
            # Only put the the first hop as the initial facts for training time.
            ins["init_facts"].append((fid_1, score_1))
      # Make them unique 
      ins["sup_facts"][0] = list(set(ins["sup_facts"][0]))
      ins["sup_facts"][1] = list(set(ins["sup_facts"][1]))
    elif FLAGS.split != "train" and len(ins["init_facts"]) < FLAGS.max_num_facts: 
      for fid, s in all_ret_facts:
        fact = facts_dict[fid]
        fact_concepts = set([m["kb_id"] for m in fact["mentions"]])  
        ins["init_facts"].append((gkb_id_to_id[fid], s)) 
        if len(ins["init_facts"]) >= FLAGS.max_num_facts:
          break  
    new_data.append(ins)
  with open(FLAGS.output_file, "w") as f:
    logging.info("num_covered: %d", num_covered)
    logging.info("len(new_data): %d", len(new_data))
    logging.info("Coverage:%.2f", num_covered/len(new_data))
    logging.info("Writing to %s", f.name)
    f.write("\n".join([json.dumps(i) for i in new_data])+"\n")
  logging.info("Done.")

if __name__ == "__main__":
  app.run(main)
