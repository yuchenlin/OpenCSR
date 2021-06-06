# Open-Ended Commonsense Reasoning

**_Quick links:_** 
[**[Paper]**](https://www.aclweb.org/anthology/2021.naacl-main.366/)  | 
[**[Video]**](https://mega.nz/file/5SpQjJKS#J82pfZVDzy3r4aWdNF4R6O8EP5gsepbY20vYihANfgE) | 
[**[Slides]**](/opencsr_naacl_slides.pptx) | 
[**[Documentation]**](https://open-csr.github.io/)


**Please check the [***documentation***](https://open-csr.github.io/) for running the code.**

## Intro 
This is the repository of the paper, [_Differentiable Open-Ended Commonsense Reasoning_](https://www.aclweb.org/anthology/2021.naacl-main.366/), by [_Bill Yuchen Lin_](https://yuchenlin.xyz/), [_Haitian Sun_](https://scholar.google.com/citations?user=opSHsTQAAAAJ&hl=en), [_Bhuwan Dhingra_](http://www.cs.cmu.edu/~bdhingra/), [_Manzil Zaheer_](https://scholar.google.com/citations?user=A33FhJMAAAAJ&hl=en), [_Xiang Ren_](http://ink-ron.usc.edu/xiangren/), and [_William W. Cohen_](https://wwcohen.github.io/), in Proc. of [*NAACL 2021*](https://2021.naacl.org/). 
The work was mainly done during Bill Yuchen Lin’s
research internship at Google.

## Content 


- drfact_data/
    - datasets/ **_(download from [here](https://open-csr.github.io/data#the-opencsr-datasets))_**
    - knowledge_corpus/**_(download from [here](https://open-csr.github.io/data#the-commonsense-knowledge-corpus))_**
- baseline_methods/
    - BM25/
    - DPR/
    - MCQA/     **_(i.e., Concept Re-ranker)_**
- language-master/language/labs/  
    - drkit/    **_(common modules for DrKIT and DrFact)_**
    - drfact/   **_(for running DrFact)_**
- scripts/
    - run_drkit.sh
    - run_drfact.sh

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

