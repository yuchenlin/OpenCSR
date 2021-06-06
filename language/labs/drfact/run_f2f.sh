CORPUS_PATH=drfact_data/knowledge_corpus/
INDEX_PATH=drfact_data/local_drfact_index/

for (( c=$1; c<=$2; c++ ))
do
   python -m language.labs.drfact.fact2fact_index \
    --do_preprocess \
    --corpus_file ${CORPUS_PATH}/gkb_best.drfact_format.jsonl \
    --fact2fact_index_dir ${INDEX_PATH}/fact2fact_index_150 \
    --num_shards 150 --my_shard $c \
    --max_sleep -1 \
    --alsologtostderr &
done
wait