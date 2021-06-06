## The Commonsense Knowledge Corpus

[Download the Preprocessed Corpus](https://mega.nz/folder/9ToQWLQJ#PxGYM-wymiOI4YRCNyyafA)

We use the [GenericsKB](https://allenai.org/data/genericskb) as our knowledge corpus. We preprocess the corpus, extract the frequent concepts, and finally link the facts to their mentioned concepts. The link contains two files:

- `gkb_best.drfact_format.jsonl` consists of generics commonsense facts, each of which is a statement of common knowledge. (See an example below.)
- `gkb_best.vocab.txt` is the vocabulary of the concepts (i.e., noun chunks) sorted by their frequency.

Below is an example json line for the fact: "_Trees remove dust and pollution from the air._"

```json
{
    "id": "gkb-best#934338", "url": "gkb-best#934338",
    "context": "Trees remove dust and pollution from the air .",
    "mentions": [
        { "kb_id": "tree", "start": 0, "text": "Trees", "sent_id": 0 },
        { "kb_id": "dust", "start": 13, "text": "dust", "sent_id": 0 },
        { "kb_id": "pollution", "start": 22, "text": "pollution", "sent_id": 0 },
        { "kb_id": "air", "start": 41, "text": "air", "sent_id": 0 }
    ],
    "title": "Trees", "kb_id": "tree"
}
```


**Please also cite the GenericsKB paper if you use the data here.**

```bibtex
@article{Bhakthavatsalam2020GenericsKBAK,
  title={GenericsKB: A Knowledge Base of Generic Statements},
  author={Sumithra Bhakthavatsalam and Chloe Anastasiades and P. Clark},
  journal={ArXiv},
  year={2020},
  volume={abs/2005.00660}
}
```