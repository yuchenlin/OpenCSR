# Open-Ended Commonsense Reasoning

### **_Quick links:_**  [**[Paper]**](https://www.aclweb.org/anthology/2021.naacl-main.366/)    [**[Video]**](https://mega.nz/file/5SpQjJKS#J82pfZVDzy3r4aWdNF4R6O8EP5gsepbY20vYihANfgE)  [**[Slides]**](https://open-csr.github.io/opencsr_naacl_slides.pptx)   [**[Documentation]**](https://open-csr.github.io/)

---

![DrFact](https://open-csr.github.io/images/poaster.png)

--- 

This is the repository of the paper, [_Differentiable Open-Ended Commonsense Reasoning_](https://www.aclweb.org/anthology/2021.naacl-main.366/), by [_Bill Yuchen Lin_](https://yuchenlin.xyz/), [_Haitian Sun_](https://scholar.google.com/citations?user=opSHsTQAAAAJ&hl=en), [_Bhuwan Dhingra_](http://www.cs.cmu.edu/~bdhingra/), [_Manzil Zaheer_](https://scholar.google.com/citations?user=A33FhJMAAAAJ&hl=en), [_Xiang Ren_](http://ink-ron.usc.edu/xiangren/), and [_William W. Cohen_](https://wwcohen.github.io/), in Proc. of [*NAACL 2021*](https://2021.naacl.org/). 



## Abstract 
Current commonsense reasoning research focuses on developing models that use commonsense knowledge to answer multiple-choice questions. However, systems designed to answer multiple-choice questions may not be useful in applications that do not provide a small list of candidate answers to choose from. As a step towards making commonsense reasoning research more realistic, we propose to study open-ended commonsense reasoning (OpenCSR) — the task of answering a commonsense question without any pre-defined choices — using as a resource only a corpus of commonsense facts. OpenCSR is challenging due to a large decision space, and because many questions require implicit multi-hop reasoning. As an approach to OpenCSR, we propose DrFact, an efficient Differentiable model for multi-hop Reasoning over knowledge Facts. To evaluate OpenCSR methods, we adapt several popular commonsense reasoning benchmarks, and collect multiple new answers for each test question via crowd-sourcing. Experiments show that DrFact outperforms strong baseline methods by a large margin.

## Content 

### Please check the [***documentation***](https://open-csr.github.io/methods/) for running the code.

We show the instructions for running four retrieval approaches to the OpenCSR task — BM25 (off-the-shelf), DPR (EMNLP2020), DrKIT (ICLR 2020) and DrFact (ours, NAACL 2021), as well as a concept re-ranker to boost the performance by learning with cross-attention. Note that there is a relative dependency of these four methods:

- training the DPR model needs the results from BM25 (to create training data);
- DrFact needs to reuse DPR’s fact index and single-hop results (for creating distant supervision);
- DrFact and DrKIT share many utility functions (sparse matrix operation and indexing scripts). We detailed the detailed instructions in individual pages.

## Outline and Documentation

- *[drfact_data/](drfact_data)*
    - *[datasets/](drfact_data/datasets)* **_(download from [here](https://open-csr.github.io/data#the-opencsr-datasets))_**
    - *[knowledge_corpus/](drfact_data/knowledge_corpus/)* **_(download from [here](https://open-csr.github.io/data#the-commonsense-knowledge-corpus))_**
- *[baseline_methods/](baseline_methods/)*
    - *[BM25/](baseline_methods/BM25)*   --> **https://open-csr.github.io/methods/bm25**
    - *[DPR/](baseline_methods/DPR)*    --> **https://open-csr.github.io/methods/dpr**
    - *[MCQA/](baseline_methods/MCQA)*     **_(i.e., Concept Re-ranker)_**  --> **https://open-csr.github.io/methods/reranker**
- *[language-master/language/labs/](language-master/language/labs/)*  
    - *[drkit/](language-master/language/labs/drkit)*    **_(common modules for DrKIT and DrFact)_**
    - *[drfact/](language-master/language/labs/drfact)*   **_(for running DrFact)_**    
- *[scripts/](scripts/)*
    - *[run_drkit.sh](scripts/run_drkit.sh)*    --> **https://open-csr.github.io/methods/drkit**
    - *[run_drfact.sh](scripts/run_drfact.sh)*  --> **https://open-csr.github.io/methods/drfact**
- *[evaluation/](evaluation/)*  --> **https://open-csr.github.io/evaluation**


## Citation
```bib
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

**Note: this is not an official Google product.**

## Contact
This repo is now under active development, and there may be issues caused by refactoring code.
Please email ***yuchen.lin@usc.edu*** if you have any questions.
