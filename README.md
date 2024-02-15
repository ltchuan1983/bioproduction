# Prediction of bioprocess yield, titer and rate

![Version](https://img.shields.io/badge/Version-1.0-blue)

Predicting microbial bioproduction (yield, titer and rate) from bioprocess variables and genetic modifications.

[Reference: Machine learning framework for assessment of microbial factory performance] (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0210558)

[Original Dataset] (https://doi.org/10.1371/journal.pone.0210558.s001)

Several techniques (log transformation, scaling, onehot encoding, embedding) were employed to handle the mixture of data types. Of note, representation of the categorical features and the list of genes through embeddings helped model performance. A variety of models, including those handling hyperparameter tuning e.g. BayesSearchCV, H2O AutoML and keras_tuner, was used. Similar to the reference, ensemble models, particularly ensemble of neural network with CatBoostRegressor, performed the best in terms of R<sup>2</sup> values. Despite the lack of COBRA features (from metabolic models), overall R<sup>2</sup> reached around 0.8 through careful feature engineering and selection of models. 



# Installation

1. Clone repository

git clone https://github.com/ltchuan1983/bioproduction.git

2. Install dependencies

pip install -r requirements.txt

# Usage

