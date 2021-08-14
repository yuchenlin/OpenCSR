"""
This script is used for evaluating the submissions to the OpenCSR leaderboard.

The submission file is a jsonl file and each line is a json dict as follows:

{
    "qid": "ACTAAP_2014_7_6",
    "predictions":[
        "apple",
        "banana",
        "car", 
        "...."
    ]
}

"""

import json
from collections import defaultdict

def hit_at_K(truth, pred, K):
    num_examples = len(truth)
    num_hit = 0
    for qid in truth:
        answers = set(truth[qid]["all_answers"])
        pred_answers = set(pred[qid]["predictions"][:K])
        if any([a in pred_answers for a in answers]):
            num_hit += 1
    hit_acc_at_K = num_hit / num_examples
    return hit_acc_at_K

def recall_at_K(truth, pred, K):
    num_examples = len(truth)
    ratios = []
    for qid in truth:
        answers = set(truth[qid]["all_answers"])
        pred_answers = set(pred[qid]["predictions"][:K])
        ratios.append(len(answers & pred_answers) / len(answers))
    recall_acc_at_K = sum(ratios)/len(ratios)
    return recall_acc_at_K


def evaluate(truth_data, prediction_data):
    if set(truth_data.keys()) != set(prediction_data.keys()):
        print("qids are not matched.")
        # print(truth_data.keys())
        # print(prediction_data.keys())
        return 
    Ks = list(range(10, 301, 10))
    result = {}
    result["hit_at_K"] = {}
    result["recall_at_K"] = {}
    for K in Ks:
        result["hit_at_K"][K] = hit_at_K(truth_data, prediction_data, K)
        result["recall_at_K"][K] = recall_at_K(truth_data, prediction_data, K)
    return result

    
def clean_data(item):
    clean_item = {}
    clean_item["qid"] = item["_id"]
    def get_concepts(ent_list):
        return [ent["name"] for ent in ent_list]
    clean_item["all_answers"] = get_concepts(item["all_answer_concepts"])
    return clean_item

def load_truth_data(truth_data_file):
    truth_data = {}
    with open(truth_data_file, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            if not line:
                continue
            item = json.loads(line)
            item = clean_data(item)
            truth_data[item["qid"]] = item
    return truth_data

def load_predictions(pred_result_file):
    prediction_data = {}
    with open(pred_result_file, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            if not line:
                continue
            item = json.loads(line)
            # item["_id"] = item["_id"] # for testing the truth
            # item = clean_data(item)
            # item["predictions"] = item["all_answers"]
            prediction_data[item["qid"]] = item
    return prediction_data

if __name__ == "__main__":
    truth_data = load_truth_data("/Users/yuchenlin/Documents/GitHub/drfact/public_data/ARC/linked_test.jsonl")
    pred_data = load_predictions("/Users/yuchenlin/Documents/GitHub/drfact/public_data/ARC/test.pred.jsonl")
    # print(pred_data)
    result = evaluate(truth_data, pred_data)
    print(result)