# Lint as: python3
"""Evaluates DrFact results on OpenCSR Dataset."""

import collections
import json

from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


# def scoring_choice(choice2found_concepts, method="top"):
#   """Computes the score of each choice and return the best one."""
#   scores = {}
#   for choice, found_concepts in choice2found_concepts.items():
#     if method == "top":
#       scores[choice] = float(found_concepts[0][1])  # 0 is the top tuple
#     elif method == "avg":
#       scores[choice] = float(np.mean([fc[1] for fc in found_concepts]))
#   return scores


def opencsr_eval_fn(dataset, results, name_map, output_prediction_file,
                    paragraphs, **kwargs):
  """Computes evaluation metrics for OpenCSRDataset.

  Args:
    dataset: An object of type OpenCSRDataset.
    results: A list of result dicts from running estimator.predict.
    name_map: A mapping from prediction indices to text strings.
    output_prediction_file: File to store predictions to.
    paragraphs: All facts in a dict.
    **kwargs: Variable keyword arguments.

  Returns:
    metrics: A dict mapping metric names to values.
  """
  del kwargs

  # def remove_nested(concept_set):
  #   concept_set_copy = list(concept_set)
  #   for a in concept_set_copy:
  #     for b in concept_set_copy:
  #       if a != b and a in b:
  #         if a in concept_set:
  #           concept_set.remove(a)
  #           break
  #   return concept_set

  #   # Collect ground truth answers.
  gt_correct = {ex.qas_id: ex.correct_choice for ex in dataset.examples}
  gt_choices = {ex.qas_id: ex.choice2concepts for ex in dataset.examples}
  gt_ques = {ex.qas_id: ex.question_text for ex in dataset.examples}

  keep_num_large = 5000
  keep_num_mini = 500
  num_hops = FLAGS.num_hops
  if FLAGS.model_type == "drkit":
    num_hops += 1 # for the qry entitiy layer 
  layer_weights = np.zeros_like(results[0]["layer_probs"])
  tf.logging.info("layer_weights.shape: %s", str(layer_weights.shape))
  all_predictions = {}
  # K = 300
  Ks = [20, 50, 100]
  top_k_count = {}
  layer_top_k_count = [{} for i in range(num_hops)]
  for K in Ks:
    top_k_count[K] = 0
    for i in range(num_hops):
      layer_top_k_count[i][K]=0

  for rid, result in enumerate(results):
    qas_id = result["qas_ids"].decode("utf-8")
    preds = result["top_idx"]
    scores = result["top_vals"]
    all_predictions[qas_id] = collections.OrderedDict()
    correct_choice = gt_correct[qas_id]
    choice2concepts = gt_choices[qas_id]  # choice2concepts dict
    answer_concepts = choice2concepts[correct_choice]
    # answer_concepts = remove_nested(answer_concepts)
    pred_list = []
    if rid % 100 == 0:
      tf.logging.info("Processed %d results in hotpot_eval_fn", rid)
      # tf.logging.info("found_choices: %d ", len(found_choices))
      # tf.logging.info("len(preds): %d ", len(preds))
    # found_oracle = False
    hit_flags = {}
    
    for K in Ks:
      hit_flags[K] = False
    for i, pred in enumerate(preds):
      pred_concept = name_map[str(pred)]
      if float(scores[i]) < 0:
        continue  # ignore the negative scored entities
      if pred_concept in answer_concepts:
        for K in Ks:
          if i <= K and not hit_flags[K]:
            top_k_count[K] += 1
            hit_flags[K] = True
      if len(pred_list) <= keep_num_large:
        if float(scores[i]) > 0:
          pred_list.append((pred_concept, float(scores[i])))
      else:
        break
    all_predictions[qas_id]["question"] = gt_ques[qas_id]
    all_predictions[qas_id]["correct_choice"] = correct_choice
    if FLAGS.model_type == "drfact":
      # Qry Ents
      qry_ents = [int(v) for v in result["qry_ents"] if v >= 0]
      qry_ents_scores = [float(v) for v in result["qry_ent_scores"] if v >= 0]
      all_predictions[qas_id]["qry_ents"] = [
          (eid, name_map[str(eid)], score)
          for eid, score in zip(qry_ents, qry_ents_scores)
      ]

      # Qry Init Facts
      qry_init_facts = [int(v) for v in result["qry_init_facts"] if v >= 0]
      qry_init_fact_scores = [float(v) for v in result["qry_init_fact_scores"] if v >= 0]
      all_predictions[qas_id]["qry_init_facts"] = [
          (fid, " ".join(paragraphs[fid]).replace(" ##", ""), score)
          for fid, score in zip(qry_init_facts, qry_init_fact_scores)
      ]


      # Facts
      for hop_id in range(4):
        hop_key = "layer_%d_fact_ids" % hop_id
        if hop_key not in result:
          continue
        cur_facts = [int(v) for v in result[hop_key]]
        cur_fact_scores = [
            float(v) for v in result[hop_key.replace("ids", "scs")]
        ]
        all_predictions[qas_id][hop_key] = [
            (fid, " ".join(paragraphs[fid]).replace(" ##", ""), score)
            for fid, score in zip(cur_facts, cur_fact_scores)
            if score > 0
        ]
    # Non-accuracy stats
    layer_weights += result["layer_probs"]
    layer_entities = {i: [] for i in range(num_hops)}
    layer_scores = {i: [] for i in range(num_hops)}
    all_predictions[qas_id]["layer_probs"] = str(result["layer_probs"])
    all_predictions[qas_id]["layer_ent_pred"] = collections.OrderedDict()
    
    for i in range(num_hops):
      layer_entities[i] = result["layer_%d_ent" % i][:keep_num_mini]
      layer_scores[i] = result["layer_%d_scs" % i][:keep_num_mini]
      for K in Ks:
        layer_pred_at_K = result["layer_%d_ent" % i][:K]
        layer_pred_at_K = [name_map[str(ee)] for ee in layer_pred_at_K if ee > 0]
        layer_top_k_count[i][K] += 1 if any([ans in layer_pred_at_K for ans in answer_concepts]) else 0
      

      all_predictions[qas_id]["layer_ent_pred"]["layer_%d" % i] = [
          (name_map[str(ee)], float(layer_scores[i][ee_id])) for ee_id, ee in enumerate(layer_entities[i]) if ee > 0
      ]
    all_predictions[qas_id]["top_%d_predictions" % keep_num_large] = pred_list

  metric = dict()
  metric["num_examples"] = len(results)
  for K in Ks:
    metric["top_%d_acc"%K] = top_k_count[K] / metric["num_examples"] 
    for i in range(num_hops):
      metric["hop%d_top_%d_acc"%(i, K)] =  layer_top_k_count[i][K] / metric["num_examples"] 
  metric["accuracy"] = metric["top_%d_acc"%20]
  # metric["layer_top_k_acc"] = layer_top_k_count
  
  # Non-accuracy analysis
  for i in range(num_hops):  # hop_id
    metric["analysis/layer_weight_%d" %
           i] = layer_weights[i] / len(all_predictions)

  results = dict(all_predictions=all_predictions, metric=metric)
  with tf.gfile.Open(output_prediction_file, "w") as gfo:
    tf.logging.info("Saving results to: %s ", output_prediction_file)
    gfo.write(json.dumps(metric) + "\n")
    gfo.write("\n".join([
        json.dumps(dict(qas_id=k, predictions=v))
        for k, v in all_predictions.items()
    ]))
    gfo.write("\n")
  return metric


# def opencsr_eval_fn(dataset, results, name_map, output_prediction_file,
#                     paragraphs, **kwargs):
#   """Computes evaluation metrics for OpenCSRDataset.

#   Args:
#     dataset: An object of type OpenCSRDataset.
#     results: A list of result dicts from running estimator.predict.
#     name_map: A mapping from prediction indices to text strings.
#     output_prediction_file: File to store predictions to.
#     paragraphs: All facts in a dict.
#     **kwargs: Variable keyword arguments.

#   Returns:
#     metrics: A dict mapping metric names to values.
#   """
#   del kwargs

#   #   # Collect ground truth answers.
#   gt_correct = {ex.qas_id: ex.correct_choice for ex in dataset.examples}
#   gt_choices = {ex.qas_id: ex.choice2concepts for ex in dataset.examples}
#   gt_ques = {ex.qas_id: ex.question_text for ex in dataset.examples}

#   keep_num_large = 5000
#   keep_num_mini = 50
#   num_correct_top = 0
#   num_correct_avg = 0
#   num_oracle = 0

#   num_hops = 2
#   layer_weights = np.zeros_like(results[0]["layer_probs"])
#   tf.logging.info("layer_weights.shape: %s", str(layer_weights.shape))
#   all_predictions = {}
#   for rid, result in enumerate(results):
#     qas_id = result["qas_ids"].decode("utf-8")
#     preds = result["top_idx"]
#     scores = result["top_vals"]
#     all_predictions[qas_id] = collections.OrderedDict()
#     correct_choice = gt_correct[qas_id]
#     choice2concepts = gt_choices[qas_id]  # choice2concepts dict
#     pred_list = []
#     concept2choice = {}  # backtrack
#     for choice, concepts in choice2concepts.items():
#       for concept in concepts:
#         assert concept not in concept2choice
#         concept2choice[concept] = choice
#     choice2found_concepts = collections.defaultdict(list)
#     found_concepts = set()
#     if rid % 100 == 0:
#       tf.logging.info("Processed %d results in hotpot_eval_fn", rid)
#       # tf.logging.info("found_choices: %d ", len(found_choices))
#       # tf.logging.info("len(preds): %d ", len(preds))
#     found_oracle = False
#     num_pred_concept = 0
#     for i, pred in enumerate(preds):
#       pred_concept = name_map[str(pred)]
#       if float(scores[i]) < 0:
#         continue  # ignore the negative scored entities
#       num_pred_concept += 1
#       if pred_concept in concept2choice:
#         found_concepts.add(pred_concept)
#         pred_choi = concept2choice[pred_concept]
#         choice2found_concepts[pred_choi].append(
#             (pred_concept, float(scores[i])))
#         if pred_choi == correct_choice and not found_oracle:
#           num_oracle += 1
#           found_oracle = True

#       if len(pred_list) <= keep_num_large:
#         if float(scores[i]) > 0:
#           pred_list.append((pred_concept, float(scores[i])))
#       if len(found_concepts) == len(concept2choice):
#         # Early stop when we found all choice-concepts.
#         break
#     choice2score_top = scoring_choice(choice2found_concepts, method="top")
#     choice2score_avg = scoring_choice(choice2found_concepts, method="avg")
#     choice2score_top_sorted = sorted(
#         [(k, float(v)) for k, v in choice2score_top.items()],
#         key=lambda x: x[1],
#         reverse=True)
#     choice2score_avg_sorted = sorted(
#         [(k, float(v)) for k, v in choice2score_avg.items()],
#         key=lambda x: x[1],
#         reverse=True)
#     if choice2score_top_sorted:
#       # Found at least one choice.
#       if choice2score_top_sorted[0][0] == correct_choice:
#         num_correct_top += 1
#       if choice2score_avg_sorted[0][0] == correct_choice:
#         num_correct_avg += 1
#         all_predictions[qas_id]["correct"] = True
#       else:
#         all_predictions[qas_id]["correct"] = False
#     all_predictions[qas_id]["question"] = gt_ques[qas_id]
#     all_predictions[qas_id]["num_pred_concept"] = num_pred_concept
#     all_predictions[qas_id]["correct_choice"] = correct_choice
#     all_predictions[qas_id]["choice2found_concepts"] = choice2found_concepts
#     all_predictions[qas_id]["choice2score"] = {
#         "top": choice2score_top,
#         "avg": choice2score_avg
#     }
#     if FLAGS.model_type == "drfact":
#       # Qry Ents
#       qry_ents = [int(v) for v in result["qry_ents"] if v >= 0]
#       qry_ents_scores = [float(v) for v in result["qry_ent_scores"] if v >= 0]
#       all_predictions[qas_id]["qry_ents"] = [
#           (eid, name_map[str(eid)], score)
#           for eid, score in zip(qry_ents, qry_ents_scores)
#       ]

#       # Qry Init Facts
#       qry_init_facts = [int(v) for v in result["qry_init_facts"] if v >= 0]
#       qry_init_fact_scores = [float(v) for v in result["qry_init_fact_scores"] if v >= 0]
#       all_predictions[qas_id]["qry_init_facts"] = [
#           (fid, " ".join(paragraphs[fid]).replace(" ##", ""), score)
#           for fid, score in zip(qry_init_facts, qry_init_fact_scores)
#       ]


#       # Facts
#       for hop_id in range(4):
#         hop_key = "layer_%d_fact_ids" % hop_id
#         if hop_key not in result:
#           continue
#         cur_facts = [int(v) for v in result[hop_key]]
#         cur_fact_scores = [
#             float(v) for v in result[hop_key.replace("ids", "scs")]
#         ]
#         all_predictions[qas_id][hop_key] = [
#             (fid, " ".join(paragraphs[fid]).replace(" ##", ""), score)
#             for fid, score in zip(cur_facts, cur_fact_scores)
#             if score > 0
#         ]
#     # Non-accuracy stats
#     layer_weights += result["layer_probs"]
#     layer_entities = {i: [] for i in range(num_hops)}
#     layer_scores = {i: [] for i in range(num_hops)}
#     all_predictions[qas_id]["layer_probs"] = str(result["layer_probs"])
#     all_predictions[qas_id]["layer_ent_pred"] = collections.OrderedDict()
#     for i in range(num_hops):
#       layer_entities[i] = result["layer_%d_ent" % i][:keep_num_mini]
#       layer_scores[i] = result["layer_%d_scs" % i][:keep_num_mini]
#       all_predictions[qas_id]["layer_ent_pred"]["layer_%d" % i] = [
#           (name_map[str(ee)], float(layer_scores[i][ee_id])) for ee_id, ee in enumerate(layer_entities[i]) if ee > 0
#       ]
#     all_predictions[qas_id]["top_%d_predictions" % keep_num_large] = pred_list

#   metric = dict()
#   metric["num_correct_top"] = num_correct_top
#   metric["num_correct_avg"] = num_correct_avg
#   metric["num_examples"] = len(results)
#   metric["accuracy_top"] = num_correct_top / len(results)
#   metric["accuracy_avg"] = num_correct_avg / len(results)
#   metric["accuracy"] = metric["accuracy_avg"]
#   metric["num_oracle"] = num_oracle
#   metric["oracle"] = num_oracle / len(results)
#   # Non-accuracy analysis
#   for i in range(num_hops):  # hop_id
#     metric["analysis/layer_weight_%d" %
#            i] = layer_weights[i] / len(all_predictions)
#     # metric["analysis/num_entities_%d" %
#     #        i] = num_layer_entities[i] / len(all_predictions)
#     # metric["analysis/num_new_entities_%d" %
#     #        i] = num_new_entities[i] / len(all_predictions)

#   results = dict(all_predictions=all_predictions, metric=metric)
#   with tf.gfile.Open(output_prediction_file, "w") as gfo:
#     tf.logging.info("Saving results to: %s ", output_prediction_file)
#     gfo.write(json.dumps(metric) + "\n")
#     gfo.write("\n".join([
#         json.dumps(dict(qas_id=k, predictions=v))
#         for k, v in all_predictions.items()
#     ]))
#     gfo.write("\n")
#   return metric
