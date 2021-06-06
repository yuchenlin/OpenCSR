# Lint as: python3
"""Adds pre-computed results as initial facts for OpenCSR data."""
import json

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string("drfact_format_gkb_file", None, "Path to gkb corpus.")
flags.DEFINE_string("qid2facts", None, "Path to dataset file.")
flags.DEFINE_string("sup_facts_file", None, "Path to dataset file.")


def main(_):
  """Main funciton."""
  logging.set_verbosity(logging.INFO)

  with open(FLAGS.drfact_format_gkb_file) as f:
    logging.info("Reading %s..."%f.name)
    gkb_id_to_id = {}
    fact_to_id = {}
    facts_dict = {}
    cur_fact_ind = 0
    for line in f.read().splitlines(): 
      instance = json.loads(line)
      gkb_id_to_id[instance["id"]] = cur_fact_ind
      fact_to_id[instance["context"].replace(" ","")] = cur_fact_ind
      facts_dict[instance["id"]] = instance
      cur_fact_ind += 1
  with open(FLAGS.qid2facts, "r") as f:
    qid2facts = json.load(f)
 
  sup_facts_data = [] 
  with open(FLAGS.sup_facts_file) as f:
    jsonlines = f.read().splitlines()
  sup_facts_data = [json.loads(jsonline) for jsonline in jsonlines]

  for ind, ins in tqdm(enumerate(sup_facts_data), desc=FLAGS.sup_facts_file, total=len(sup_facts_data)):
    assert ins["_id"] in qid2facts
    truth_facts = qid2facts[ins["_id"]]
    assert truth_facts[0].replace(" ","") in fact_to_id, truth_facts[0] 

    # sup_item = ([], 100.0, truth_facts[0])
    # sup_facts_data[ind]["sup_facts"].append()
    # for sup_item in sup_facts_data[ind]["sup_facts"]:
    #   item_supfacts = sup_item[0] # a list of facts
    #   if len(item_supfacts) == 1: # One-hop quesiton
    #     fid = item_supfacts[0][0]
    #     score = item_supfacts[0][1]
    #   elif len(item_supfacts) == 2: # Two-hop quesiton
    #     fid_1 = item_supfacts[0][0]
    #     score_1 = item_supfacts[0][1]
    #     fid_2 = item_supfacts[1][0]
    #     score_2 = item_supfacts[1][1]
    #     if (fid_1, fid_2) not in fact_links:
    #       fact_links.add((fid_1, fid_2))

  logging.info("Done.")

if __name__ == "__main__":
  app.run(main)
