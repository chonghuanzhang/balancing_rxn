# Completing and balancing database excerpted chemical reactions with a hybrid mechanism - machine learning approach

Paper(https://pubs.acs.org/doi/10.1021/acsomega.4c00262),

Repository(https://github.com/chonghuanzhang/balancing_rxn)

## Introduction
Computer Aided Synthesis Planning (CASP) development of reaction routes requires understanding of complete reaction structures. However, most reactions in the current databases are missing reaction co-participants. Although reaction prediction and atom mapping tools are able to predict major reaction participants and trace atom rearrangements in reactions, they fail to identify the missing molecules to complete reactions. This is because these approaches are data-driven models trained on the current reaction databases which comprise of incomplete reactions. In this work, a workflow was developed to tackle the reaction completion challenge. This includes a heuristic-based method to identify the balanced reactions from reaction databases and complete some imbalanced reactions by adding candidate molecules. A machine learning masked language model (MLM) was trained to learn from reaction SMILES sentences of these completed reactions. The model predicted missing molecules for the incomplete reactions; a workflow analogous to predicting missing words in sentences. The model is promising for prediction of small and middle size missing molecules in incomplete reaction records. The workflow combining both the heuristic and the machine learning methods completed more than half of the entire reaction space.

## Usage
Use the .ipynb notebook files under chem_balancer and chem_mlm folders to run the two models respectively.

## License

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/chonghuanzhang/balancing_rxn/blob/main/LICENSE) for additional details.


