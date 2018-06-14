# Stacking Bagged Boosted Forests for Automated Classification

This code was developed for my Master's Degree in Computer Science at Federal University of Minas Gerais. 
The most used code can be found in the folder *python* where I implementated my proposed methods (BERT and OOB stacking) and some baselines in python following the scikit learn interface.

## Work abstract

 Random Forests (RF) are one of the most successful strategies for automated classification tasks. 
 Motivated by the RF success, recently proposed RF-based classification approaches leverage the central RF idea of aggregating a large number of low-correlated trees, which are inherently parallelizable and provide exceptional generalization capabilities. In this context, this work brings several new contributions to this line of research. First, we propose a new RF-based strategy (BERT) that applies the boosting technique in bags of extremely randomized trees. Second, we empirically demonstrate that this new strategy, as well as the recently proposed BROOF and LazyNN\_RF classifiers do complement each other, motivating us to stack them to produce an even more effective classifier. Up to our knowledge, this is the first strategy to effectively combine the three main ensemble strategies: stacking, bagging  (the cornerstone of RFs) and boosting. Finally, we exploit the efficient and unbiased stacking strategy based on out-of-bag (OOB) samples to considerably speedup the very costly training process of the stacking procedure. Our experiments in several datasets covering two high-dimensional and noisy domains of topic and sentiment classification provide strong evidence in favor of the benefits of our RF-based solutions. We show that BERT is among the top performers in the vast majority of analyzed cases, while retaining the unique benefits of RF classifiers (explainability, parallelization, easiness of parameterization, heterogeneous data and missing value handling).
 
**Keywords:** *Stacking, Random Forest, Extremely Randomized Trees, Boosting, Classification, Supervised Learning, Machine Learning*

## Main contributions

In summary, the main contributions of this work are:
- The proposal of a novel RF-based classifier, named BERT, that is able to outperform state-of-the-art classifiers;
- The proposal of a new stacking classifier that exploits the complementary characteristics of BROOF, LazyNN\_RF and BERT that is able to outperform all analyzed classification algorithms, including a stacking of traditional methods, often by large margins; 
- The proposal of a new estimation strategy based on the use of OOB for generating the input for the stacked meta-classifier that substantially reduces the computational effort/runtime of the stacking strategy while retaining its predictive power;
- A new measure of comentarity among classifier the so-called Normalized Degree of Disagreement.

## Please cite

```
@inproceedings{Campos:2017:SBB:3077136.3080815,
 author = {Campos, Raphael and Canuto, S{\'e}rgio and Salles, Thiago and de S\'{a}, Clebson C.A. and Gon\c{c}alves, Marcos Andr{\'e}},
 title = {Stacking Bagged and Boosted Forests for Effective Automated Classification},
 booktitle = {Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval},
 series = {SIGIR '17},
 year = {2017},
 isbn = {978-1-4503-5022-8},
 location = {Shinjuku, Tokyo, Japan},
 pages = {105--114},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3077136.3080815},
 doi = {10.1145/3077136.3080815},
 acmid = {3080815},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {bagging, boosting, classification, ensemble, random forests, stacking},
} 
```

## Dissertation Text

My dissertation text is available on the following [link](http://homepages.dcc.ufmg.br/~rcampos/Dissertation_Stacking_Bagged_Boosted_Forests.pdf).
