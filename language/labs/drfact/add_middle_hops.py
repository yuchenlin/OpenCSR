# Lint as: python3
"""Adds middle hops as distant supervision for OpenCSR data."""
import json

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
import networkx as nx 
import os
import numpy as np
import itertools
from scipy import sparse
import tensorflow.compat.v1 as tf
from language.labs.drkit import search_utils
import pickle
from collections import defaultdict


FLAGS = flags.FLAGS

flags.DEFINE_string("linked_qas_file", None, "Path to dataset file.")
flags.DEFINE_string("drfact_format_gkb_file", None, "Path to gkb corpus.")
flags.DEFINE_string("sup_fact_result_without_ans", None, "Path to dataset file.")
flags.DEFINE_string("sup_fact_result_with_ans", None, "Path to dataset file.")
flags.DEFINE_string("f2f_index_file", None, "Path to dataset file.")
flags.DEFINE_string("f2f_nxgraph_file", None, "Path to dataset file.")
flags.DEFINE_string("output_file", None, "Path to dataset file.")
flags.DEFINE_string("do", None, "Path to dataset file.")



      
def preprare_fact2fact_network():
  """Loads the f2f data."""
  f2f_checkpoint = os.path.join(FLAGS.f2f_index_file)
  with tf.device("/cpu:0"):
    with tf.Graph().as_default():
      logging.info("Reading %s", f2f_checkpoint) 
      with tf.Session() as sess: 
        new_saver = tf.train.import_meta_graph(f2f_checkpoint+'.meta')
        new_saver.restore(sess, f2f_checkpoint)
        fact2fact_data = sess.run('fact2fact_data:0')
        fact2fact_indices = sess.run('fact2fact_indices:0')
        fact2fact_rowsplits = sess.run('fact2fact_rowsplits:0')
        S = sparse.csr_matrix((fact2fact_data, fact2fact_indices, fact2fact_rowsplits))
  row, col = S.nonzero()
  f2f_nxgraph = nx.DiGraph()
  node_in_dict = defaultdict(set)
  node_out_dict = defaultdict(set)
  for f_i, f_j in tqdm(list(zip(row, col)), desc="adding edges"):    
    node_out_dict[int(f_i)].add(int(f_j))
    node_in_dict[int(f_j)].add(int(f_i))
    # f2f_nxgraph.add_edge(int(f_i), int(f_j))
  
  with open(FLAGS.f2f_nxgraph_file+".indict", "wb") as f:
    pickle.dump(dict(node_in_dict), f)
  with open(FLAGS.f2f_nxgraph_file+".outdict", "wb") as f:
    pickle.dump(dict(node_out_dict), f)

  # with open(FLAGS.f2f_nxgraph_file, "wb") as f:
  #   logging.info("Writing to %s", f.name)
  #   logging.info("Num of nodes %d", f2f_nxgraph.number_of_nodes())
  #   logging.info("Num of edges %d", f2f_nxgraph.number_of_edges())
  #   pickle.dump(f2f_nxgraph, f)


def bridge(f2f_nxgraph, start_facts, end_facts, gkb_id_to_id):
  for i, j in itertools.product(start_facts, end_facts):
    for path in nx.all_shortest_paths(f2f_nxgraph, source=i, target=j):
      print(path)

def find_gap(id_to_gkb_id, facts_dict, source, target):
  # judge if it's already connected 
  fact_s = facts_dict[id_to_gkb_id[source]]
  fact_t = facts_dict[id_to_gkb_id[target]]
  fact_s_concepts = set([m["kb_id"] for m in fact_s["mentions"]])
  fact_t_concepts = set([m["kb_id"] for m in fact_t["mentions"]])
  # print(fact_s["context"], fact_t["context"])
  intersection = fact_s_concepts & fact_t_concepts
  # print(fact_s_concepts & fact_t_concepts)
  return intersection
  

def main_find_bridge(id_to_gkb_id, facts_dict):
    with open(FLAGS.sup_fact_result_with_ans) as f:
      logging.info("Reading QAS(-formatted) data...%s", f.name)
      jsonlines = f.read().split("\n")
      instances_w_ans = [json.loads(jsonline) for jsonline in jsonlines if jsonline]

    with open(FLAGS.sup_fact_result_without_ans) as f:
      logging.info("Reading QAS(-formatted) data...%s", f.name)
      jsonlines = f.read().split("\n")
      instances_wo_ans = [json.loads(jsonline) for jsonline in jsonlines if jsonline]
    
    assert len(instances_wo_ans) == len(instances_w_ans)
    num_onehop = 0
    num_twohop = 0
    final_instances = []
    for iwa, iwoa in zip(instances_w_ans, instances_wo_ans):
      # Judge if it's a one-hop questions 
      hop_num = 1
      sup_facts = [([item], item[1], 0) for item in iwa["sup_facts"]]
      if len(iwa["sup_facts"]) >= 5 and sup_facts[0][1]>= 60:
        num_onehop += 1
      else:  
        source_questions = iwa["question_only_facts"]
        target_questions = iwa["answer_only_facts"]
        two_hop = False
        for source in source_questions:
          for target in target_questions:
            intersection = find_gap(id_to_gkb_id, facts_dict, source[0], target[0])
            if len(intersection) > 0:
              sup_facts.append(([source, target], (source[1]+target[1])/2, len(intersection)))
              if not two_hop: 
                two_hop = True            
        if two_hop:
          num_twohop += 1
          hop_num = 2
      ins = iwa 
      del ins["question_only_facts"]    
      del ins["answer_only_facts"]    
      del ins["sup_facts_source"]
      ins["sup_facts"] = sup_facts
      ins["hop_num"] = hop_num
      final_instances.append(ins)

    with open(FLAGS.output_file, "w") as f: 
      logging.info(f.name + " One-hop Coverage: %.2f", num_onehop/len(instances_w_ans))
      logging.info(f.name + " Two-hop Coverage: %.2f", (num_onehop+num_twohop)/len(instances_w_ans))
      f.write("\n".join([json.dumps(i) for i in final_instances])+"\n")
    logging.info("Done.")

    
  
  
def main(_):
  """Main funciton."""
  if not FLAGS.do:
    return

  if FLAGS.do == "prepro_f2f_net":
    preprare_fact2fact_network()
  elif FLAGS.do == "hopping":
    # with open(FLAGS.f2f_nxgraph_file, "rb") as f:
    #   f2f_nxgraph = pickle.load(f)  
    # bridge(f2f_nxgraph, [339607], [69224], gkb_id_to_id) 

    # with open(FLAGS.f2f_nxgraph_file+".indict", "rb") as f:
    #   node_in_dict = pickle.load(f)
    # with open(FLAGS.f2f_nxgraph_file+".outdict", "rb") as f:
    #   node_out_dict = pickle.load(f)

    with open(FLAGS.drfact_format_gkb_file) as f:
      logging.info("Reading %s..."%f.name)
      gkb_id_to_id = {}
      id_to_gkb_id = {}
      facts_dict = {}
      cur_fact_ind = 0
      for line in f.read().split("\n"):
        if line:
          instance = json.loads(line)
          gkb_id_to_id[instance["id"]] = cur_fact_ind
          id_to_gkb_id[cur_fact_ind] = instance["id"]
          facts_dict[instance["id"]] = instance
          cur_fact_ind += 1
      logging.info("Done")

    # find_gap(id_to_gkb_id, facts_dict, 339607, 69224)
    main_find_bridge(id_to_gkb_id, facts_dict)
  

if __name__ == "__main__":
  app.run(main)
