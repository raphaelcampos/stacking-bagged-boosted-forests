---
title: "Effectivess comparison report"
author: "Raphael Rodrigues Campos"
date: "January 17, 2016"
output:
  pdf_document:
    keep_tex: yes
  html_document: default
header-includes: \usepackage{multirow}
---

Eu implementei o BROOF usando Extremely Randomized Trees no lugar da RF, gerando o algoritmo que chamei de BERT (Boosted Extremely Randomized Trees).

A própria ERT se sai melhor em alguns datasets do que a RF. Portanto, era de se esperar
que a BERT se saísse um pouco melhor que o BROOF, como pode-se verificar no arquivo anexo.

O arquivo anexo possui uma tabela comparando todos os métodos rodados até agora.

Além da implementaćão do BERT, eu também implementei método de ensemble "Stacked Generalization" descrito em [1] David H. Wolpert, "Stacked Generalization", Neural Networks, 5, 241--259, 1992.

O método comb1 na tabela é o stacking de 2 níveis para combinaćão dos métodos LazyNN\_RF e BROOF. No nível do zero do stacking foram utilizados os classificadores LazyNN_RF e BROOF para gerar o conjunto de treino do nível 1. No nível 1 foi utilizado uma RF com 200 árvores.

Os resultados apresentados são promissores. Sobretudo quando se trata de métrica microf1, onde tivemos mais ganhos significativos.




# Resultados

% latex table generated in R 3.2.4 by xtable 1.8-0 package
% Thu Apr 28 09:32:46 2016
\begin{table}[ht]
\centering
\begin{tabular}{lllllll}
  \hline
V1 & V2 & 20NG & 4UNI & ACM & REUTERS90 & MEDLINE \\ 
  \cline{3-7} \hline
\multirow{2}{*}{SVM-L2} & microF1 & \bf{90.06 $\pm$  0.43} & \bf{83.48 $\pm$  1.08} & \bf{75.4 $\pm$  0.66} & \bf{68.19 $\pm$  1.15} & 0 $\pm$  0 \\ 
   & macroF1 & \bf{89.93 $\pm$  0.43} & \bf{73.39 $\pm$  2.17} & \bf{63.84 $\pm$  0.55} & \bf{31.95 $\pm$  2.59} & 0 $\pm$  0 \\ 
   \cline{3-7}\multirow{2}{*}{BERT} & microF1 & 88.93 $\pm$  0.39 & \bf{84.61 $\pm$  0.98} & \bf{74.8 $\pm$  0.59} & \bf{67.33 $\pm$  0.72} & 0 $\pm$  0 \\ 
   & macroF1 & 88.59 $\pm$  0.5 & \bf{73.61 $\pm$  1.85} & 62.1 $\pm$  0.99 & \bf{29.24 $\pm$  1.4} & 0 $\pm$  0 \\ 
   \cline{3-7}\multirow{2}{*}{SVM-L1} & microF1 & \bf{89.8 $\pm$  0.4} & 78.23 $\pm$  1.49 & \bf{75.31 $\pm$  0.74} & \bf{68.25 $\pm$  1.2} & 0 $\pm$  0 \\ 
   & macroF1 & \bf{89.59 $\pm$  0.43} & 67.47 $\pm$  3.01 & 62.33 $\pm$  1.76 & \bf{31.37 $\pm$  2.22} & 0 $\pm$  0 \\ 
   \cline{3-7}\multirow{2}{*}{SVM-MAX} & microF1 & 88.35 $\pm$  0.37 & 81.36 $\pm$  1.01 & 73.82 $\pm$  0.78 & \bf{67.6 $\pm$  1.1} & 0 $\pm$  0 \\ 
   & macroF1 & 88.3 $\pm$  0.38 & 68.01 $\pm$  2.39 & \bf{62.55 $\pm$  1.53} & \bf{31.73 $\pm$  3.13} & 0 $\pm$  0 \\ 
   \cline{3-7}\multirow{2}{*}{BROOF} & microF1 & 87.96 $\pm$  0.24 & \bf{84.41 $\pm$  1.07} & 73.35 $\pm$  0.79 & 66.79 $\pm$  0.97 & 0 $\pm$  0 \\ 
   & macroF1 & 87.44 $\pm$  0.28 & \bf{73.23 $\pm$  1.1} & 60.76 $\pm$  0.8 & 28.48 $\pm$  2.17 & 0 $\pm$  0 \\ 
   \cline{3-7}\multirow{2}{*}{KNN} & microF1 & 87.53 $\pm$  0.69 & 75.63 $\pm$  0.94 & 70.99 $\pm$  0.96 & \bf{68.07 $\pm$  1.07} & 0 $\pm$  0 \\ 
   & macroF1 & 87.22 $\pm$  0.66 & 60.34 $\pm$  1.36 & 55.85 $\pm$  0.97 & \bf{29.93 $\pm$  2.48} & 0 $\pm$  0 \\ 
   \cline{3-7}\multirow{2}{*}{SVM-NONE} & microF1 & 83.47 $\pm$  0.46 & 80.55 $\pm$  0.72 & 71.34 $\pm$  1.01 & 66.6 $\pm$  1.06 & 0 $\pm$  0 \\ 
   & macroF1 & 83.37 $\pm$  0.42 & \bf{71.04 $\pm$  2.06} & 61.08 $\pm$  0.67 & \bf{31.68 $\pm$  3.32} & 0 $\pm$  0 \\ 
   \cline{3-7}\multirow{2}{*}{NB} & microF1 & 88.99 $\pm$  0.54 & 62.63 $\pm$  1.7 & 73.54 $\pm$  0.71 & 65.32 $\pm$  1.13 & \bf{82.92 $\pm$  0.14} \\ 
   & macroF1 & 88.68 $\pm$  0.55 & 51.38 $\pm$  3.19 & 58.03 $\pm$  0.85 & 27.86 $\pm$  0.79 & 63.8 $\pm$  0.43 \\ 
   \cline{3-7}\multirow{2}{*}{RF} & microF1 & 83.64 $\pm$  0.29 & 81.52 $\pm$  1 & 71.05 $\pm$  0.31 & 63.92 $\pm$  0.81 & 81.54 $\pm$  0.08 \\ 
   & macroF1 & 83.08 $\pm$  0.35 & 65.44 $\pm$  1.91 & 56.56 $\pm$  0.45 & 24.36 $\pm$  1.98 & \bf{67.4 $\pm$  0.36} \\ 
   \cline{3-7}\multirow{2}{*}{XT} & microF1 & 85.94 $\pm$  0.23 & 81.66 $\pm$  1.03 & 71.94 $\pm$  0.66 & 64.33 $\pm$  0.86 & 81.48 $\pm$  0.11 \\ 
   & macroF1 & 85.57 $\pm$  0.22 & 65.44 $\pm$  2.41 & 57.4 $\pm$  1.13 & 24.47 $\pm$  2.22 & \bf{67.34 $\pm$  0.29} \\ 
   \cline{3-7}\multirow{2}{*}{LAZY} & microF1 & 87.96 $\pm$  0.37 & 82.34 $\pm$  0.61 & 74.02 $\pm$  0.79 & 66.3 $\pm$  1.07 & 0 $\pm$  0 \\ 
   & macroF1 & 87.39 $\pm$  0.37 & 68.33 $\pm$  1.6 & 59.46 $\pm$  1.35 & 26.61 $\pm$  2.12 & 0 $\pm$  0 \\ 
   \cline{3-7}\multirow{2}{*}{LXT} & microF1 & 88.39 $\pm$  0.51 & 81.24 $\pm$  0.71 & 69.63 $\pm$  0.91 & 65.92 $\pm$  0.82 & 0 $\pm$  0 \\ 
   & macroF1 & 88.05 $\pm$  0.44 & 66.89 $\pm$  1.23 & 57.33 $\pm$  1.48 & 26.71 $\pm$  2.53 & 0 $\pm$  0 \\ 
   \cline{3-7}\end{tabular}
\caption{Comparaćão entre todos os métodos} 
\end{table}
% latex table generated in R 3.2.4 by xtable 1.8-0 package
% Thu Apr 28 09:33:02 2016
\begin{table}[ht]
\centering
\begin{tabular}{llllll}
  \hline
V1 & V2 & 20NG & 4UNI & ACM & REUTERS90 \\ 
  \cline{3-6} \hline
\multirow{2}{*}{COMBALL} & microF1 & \bf{91.67 $\pm$  0.44} & \bf{86.74 $\pm$  1.17} & \bf{78.46 $\pm$  0.72} & \bf{80.02 $\pm$  1.24} \\ 
   & macroF1 & \bf{91.43 $\pm$  0.42} & \bf{79.45 $\pm$  2.23} & \bf{63.72 $\pm$  1.01} & \bf{37.84 $\pm$  3.14} \\ 
   \cline{3-6}\multirow{2}{*}{COMB3} & microF1 & 90.63 $\pm$  0.57 & \bf{86.79 $\pm$  0.86} & 77.34 $\pm$  0.6 & \bf{79 $\pm$  1.14} \\ 
   & macroF1 & 90.4 $\pm$  0.57 & \bf{79.63 $\pm$  1.91} & \bf{62.91 $\pm$  0.92} & 33.93 $\pm$  2.97 \\ 
   \cline{3-6}\multirow{2}{*}{COMB1} & microF1 & 89.32 $\pm$  0.42 & \bf{86.52 $\pm$  1.18} & 76.74 $\pm$  0.73 & 77.22 $\pm$  1.14 \\ 
   & macroF1 & 89.01 $\pm$  0.44 & \bf{78.66 $\pm$  1.9} & 62.2 $\pm$  1.01 & 31.71 $\pm$  2.7 \\ 
   \cline{3-6}\multirow{2}{*}{COMB2} & microF1 & 90.2 $\pm$  0.51 & \bf{86.54 $\pm$  1.06} & 76.88 $\pm$  0.55 & 78.25 $\pm$  1.17 \\ 
   & macroF1 & 89.95 $\pm$  0.52 & \bf{79.41 $\pm$  1.63} & 62.66 $\pm$  0.81 & 32.86 $\pm$  2.23 \\ 
   \cline{3-6}\multirow{2}{*}{COMBSOTA} & microF1 & 90.65 $\pm$  0.4 & 83.79 $\pm$  1.3 & \bf{77.9 $\pm$  0.73} & 74.41 $\pm$  1.21 \\ 
   & macroF1 & 90.41 $\pm$  0.4 & 74.19 $\pm$  2.13 & \bf{63.15 $\pm$  0.76} & 28.18 $\pm$  1.58 \\ 
   \cline{3-6}\multirow{2}{*}{SVM-L2} & microF1 & 90.06 $\pm$  0.43 & 83.48 $\pm$  1.08 & 75.4 $\pm$  0.66 & 68.19 $\pm$  1.15 \\ 
   & macroF1 & 89.93 $\pm$  0.43 & 73.39 $\pm$  2.17 & \bf{63.84 $\pm$  0.55} & 31.95 $\pm$  2.59 \\ 
   \cline{3-6}\multirow{2}{*}{BERT} & microF1 & 88.93 $\pm$  0.39 & 84.61 $\pm$  0.98 & 74.8 $\pm$  0.59 & 67.33 $\pm$  0.72 \\ 
   & macroF1 & 88.59 $\pm$  0.5 & 73.61 $\pm$  1.85 & 62.1 $\pm$  0.99 & 29.24 $\pm$  1.4 \\ 
   \cline{3-6}\end{tabular}
\caption{Comparaćão entre todos os métodos} 
\end{table}

Legenda para os métodos:

- BERT: Boosted Extremely Randomized Trees
- LXT: Lazy Extremely Randomized Trees
- RF: Random Forest com 200 árvores
- RF1000: Random Forest com 1000 árvores
- XT: Extremely Randomized Trees com 200 árvores
- XT1000: Extremely Randomized Trees com 1000 árvores
- COMB1: Stacking (Lazy + BROOF)
- COMB2: Stacking (LXT + BERT)
- COMB3: Stacking (Lazy + BROOF + LXT + BERT)
- COMBSOTA: Stacking (KNN + RF + SVM + NB)