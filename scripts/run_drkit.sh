#!/bin/bash
BERT_PATH=~/uncased_L-12_H-768_A-12  # BERT-base
question_num_layers=5
ENTAGG=max
CORPUS_PATH=drfact_data/knowledge_corpus/
INDEX_PATH=drfact_data/local_index/
INDEX_NAME=drkit_mention_index
F2F_INDEX_NAME=fact2fact_index
DATASET_PATH=drfact_data/datasets/${DATASET}
NUM_HOPS=$2
MODEL_OUTPUT_DIR=${OUT_DIR}/hop_$2
PREDICT_PREFIX=dev
if [ "$1" = "train" ]; 
then
echo "training mode"
rm -r ${MODEL_OUTPUT_DIR}
DO="do_train "
mkdir -p ${MODEL_OUTPUT_DIR}
LOG_FILE=${MODEL_OUTPUT_DIR}/tf_log.train.txt
elif [ "$1" = "continual_eval" ];
then
echo "continual_eval mode"
DO="do_predict "
mkdir -p ${MODEL_OUTPUT_DIR}
LOG_FILE=${MODEL_OUTPUT_DIR}/tf_log.cont_eval.txt
elif [ "$1" = "predict" ];
then
echo "prediction mode"
PREDICT_PREFIX=$4 # train/dev/test
DO="do_predict --use_best_ckpt_for_predict --model_ckpt_toload $3 "
LOG_FILE=${MODEL_OUTPUT_DIR}/tf_log.$3-${PREDICT_PREFIX}-prediction.txt
fi

touch ${LOG_FILE}


CUDA_VISIBLE_DEVICES=${GPUS} python -m language.labs.drfact.run_drfact \
--vocab_file ${BERT_PATH}/vocab.txt \
--tokenizer_model_file None \
--bert_config_file ${BERT_PATH}/bert_config.json \
--tokenizer_type bert_tokenization \
--output_dir ${MODEL_OUTPUT_DIR} \
--train_file ${DATASET_PATH}/linked_train.jsonl \
--predict_file ${DATASET_PATH}/linked_${PREDICT_PREFIX}.jsonl \
--predict_prefix ${PREDICT_PREFIX} \
--init_checkpoint ${BERT_PATH}/bert_model.ckpt \
--train_data_dir ${INDEX_PATH}/${INDEX_NAME} \
--test_data_dir ${INDEX_PATH}/${INDEX_NAME} \
--learning_rate 3e-05 \
--warmup_proportion 0.1 \
--train_batch_size 2 \
--predict_batch_size 1 \
--save_checkpoints_steps 100 \
--iterations_per_loop 300 \
--num_train_epochs 10.0 \
--max_query_length 128 \
--max_entity_len 5 \
--qry_layers_to_use -1 \
--qry_aggregation_fn concat \
--question_dropout 0.3 \
--question_num_layers ${question_num_layers} \
--projection_dim 200 \
--train_with_sparse  \
--fix_sparse_to_one  \
--predict_with_sparse  \
--data_type opencsr \
--model_type drkit \
--supervision entity \
--num_mips_neighbors 100 \
--entity_score_aggregation_fn ${ENTAGG} \
--entity_score_threshold 5e-2 \
--softmax_temperature 5.0 \
--sparse_reduce_fn max \
--sparse_strategy sparse_first \
--num_hops ${NUM_HOPS} \
--embed_index_prefix bert_base \
--num_preds -1 \
--$DO 2> ${LOG_FILE} &

echo ${LOG_FILE}

# watch -n 1 tail -n 50 ${LOG_FILE}