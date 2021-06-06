

# Download 

[Download the OpenCSR datasets](https://forms.gle/VC9xdLjsLSXkjJM86)

This zip file contains train/dev/test data in *jsonl* format (i.e., each line is a dictionary in json) for ARC, OBQA, and QASC. Note that the gold annotations (i.e.,`all_answer_concepts` items) for the test sets are hidden for hosting [a shared task](https://open-csr.github.io/leaderboard) to fairly compare the submissions. 


# Format Illustration

## Examples for training/development data.
 
```json
// Example 1 from OBQA
{
  "_id": "257",  
    // unique example id
  "question": "Global warming is lowering the world's amount of ( ___ ) ",  
    // input: question text
  "entities": [{"kb_id": "global warming", "name": "global warming"}, ...],  
    // supplementary info: simple concept matching over the question text. "kb_id" and "name" are the same.
  "all_answer_concepts": [{"kb_id": "ice", "name": "ice"}],
    // output: answer concepts that a model needs to retrieve. there is a duplicate key "answer_concepts"
  "all_answer_concepts_decomp": [{"kb_id": "ice", "name": "ice"}],
    // supplementary info: the decomposition of the "all_answer_concepts"
  "answer": "ice"
    // supplementary info: original text of the correct choice.
}

// Example 2 from ARC
{
  "_id": "Mercury_7267873", 
  "question": "What provide the best evidence that life could develop on Mars?", 
  "entities": [{"kb_id": "good", "name": "good"}, {"kb_id": "evidence", "name": "evidence"}, {"kb_id": "life", "name": "life"}, {"kb_id": "develop", "name": "develop"}, {"kb_id": "mars", "name": "mars"}],
  "all_answer_concepts": [
    {"kb_id": "organic molecule", "name": "organic molecule"}, 
    {"kb_id": "ice", "name": "ice"}
  ],
  "all_answer_concepts_decomp": [{"kb_id": "organic molecule", "name": "organic molecule"}, {"kb_id": "ice", "name": "ice"}, {"kb_id": "organic", "name": "organic"}, {"kb_id": "molecule", "name": "molecule"}],
  "answer": "its ice and organic molecules" // original text of the correct choice.
}
```
<!-- 
// Example 3
{
  "_id": "ARC-Mercury_400056", // unique example id
  "question": "What plant trait is inherited?", // task input: question text
  "entities": ["plant", "trait"], 
    // optional info: simple concept matching.
  "original_choice": "the shape of its leaves", 
    // optional info: correct choice of the original data  
  "all_answer_concepts": ["shape", "leaf"]
    // output: answer concepts that a model needs to retrieve (as many as possible)
}

 -->

## Example for the test data.
Here is an example of the *test* data (the answers are hidden for the shared task):

```json
{
  "_id": "Mercury_SC_400328", 
    // unique example id
  "question": "What item is used for protection from chemical splashing?", 
  "entities": ["chemical splashing", "item", "use", "protection", "chemical"], 
    // supplementary info: simple concept matching over the question text.
  "all_answer_concepts": [ // hidden info; used for evaluaiton. 
    {"kb_id": "safety goggle", "name": "safety goggle"},
    {"kb_id": "lab coat", "name": "lab coat"},
    {"kb_id": "rubber boot", "name": "rubber boot"},
    {"kb_id": "protective glove", "name": "protective glove"},
    {"kb_id": "face shield", "name": "face shield"},
    {"kb_id": "helmet", "name": "helmet"},
    {"kb_id": "shoe", "name": "shoe"},
    {"kb_id": "glass", "name": "glass"},
    {"kb_id": "goggle", "name": "goggle"},
    {"kb_id": "glove", "name": "glove"},
    {"kb_id": "coverall", "name": "coverall"}
  ],  // Note that the first one is from the original choice, and others are from our crowdsourcing.
  "answer": "safety goggle",  // original text of the correct choice.
    // hidden info.
}
``` 


# Citations

Please cite the following papers:

```bibtex
@article{clark2018think,
  title={Think you have solved question answering? try arc, the ai2 reasoning challenge},
  author={Clark, Peter and Cowhey, Isaac and Etzioni, Oren and Khot, Tushar and Sabharwal, Ashish and Schoenick, Carissa and Tafjord, Oyvind},
  journal={arXiv preprint arXiv:1803.05457},
  year={2018}
}

@inproceedings{mihaylov2018can,
    title = "Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering",
    author = "Mihaylov, Todor  and Clark, Peter  and Khot, Tushar  and Sabharwal, Ashish",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    year = "2018"
}

@inproceedings{khot2020qasc,
  author    = {Tushar Khot and  Peter Clark and Michal Guerquin and Peter Jansen and Ashish Sabharwal},
  title     = {QASC: A Dataset for Question Answering via Sentence Composition},
  booktitle = {The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI
               2020, New York, NY, USA, February 7-12, 2020},
  year      = {2020}
}

@inproceedings{lin-etal-2021-differentiable,
    title = "Differentiable Open-Ended Commonsense Reasoning",
    author = "Lin, Bill Yuchen and Sun, Haitian and Dhingra, Bhuwan and Zaheer, Manzil and Ren, Xiang and Cohen, William",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.366",
    pages = "4611--4625"
}

```