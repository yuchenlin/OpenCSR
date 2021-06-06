"""Converts the DPR index to DrFact."""
from absl import app
from absl import flags
import pickle
import numpy as np
import os
from language.labs.drkit import search_utils
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS
flags.DEFINE_string("index_result_path", None, "Path to output files.")
flags.DEFINE_string("dpr_pkl_path", None, "Path to dpr pickle file.")
flags.DEFINE_string("embed_prefix", "dpr_bert_base",
                    "The prefix of the embedding files.")

def main(_):
  """Main fuction."""
  with open(FLAGS.dpr_pkl_path, "rb") as reader:
    doc_vectors = pickle.load(reader)

  dim = doc_vectors[0][1].shape[0]
  num_facts = len(doc_vectors)
  fact_emb = np.empty((num_facts, dim), dtype=np.float32)

  for ind, doc in enumerate(doc_vectors):
    _, doc_vector = doc
    fact_emb[ind, :] = doc_vector

  output_index_path = os.path.join(FLAGS.index_result_path, "%s_fact_feats" %FLAGS.embed_prefix)
  tf.logging.info("Saving %d fact features to tensorflow checkpoint.",
                        fact_emb.shape[0])
  with tf.device("/cpu:0"):
    search_utils.write_to_checkpoint("fact_db_emb", fact_emb, tf.float32, output_index_path)

if __name__ == "__main__":
  app.run(main)
