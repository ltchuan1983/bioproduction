"""
Modes module

This module provides the functions to run different modes as specified at command line

Functions:
    run_train(X_train, y_train, X_test, y_test):                    Predict with CatBoostRegressor
    run_train_multi(X_train, y_train, X_test, y_test):              Predict with a variety of regressors
    run_train_gridsearch(X_train, y_train, X_test, y_test):         Predict with GridSearchCV using CatBoostRegressor
    run_train_bayes(X_train, y_train, X_test, y_test):              Predict with BayesSearchCV using CatBoostRegressor
    run_train_nn(X_train, y_train, X_test, y_test):                 Predict with Keras Neural Network
    run_train_embed_nn(X_train, y_train, X_test, y_test):           Predict with Keras Neural Network (1 embedding for onehot_encoded features)
    run_train_embed_genotype_nn(X_train, y_train, X_test, y_test):  Predict with Keras Neural Network (1 embedding for onehot_encoded features, 1 embedding for tokenized genotype)
    run_train_tunable_nn(X_train, y_train, X_test, y_test):         Predict with tunable Keras Neural Network (using keras-tuner)
    run_train_automl(X_train, y_train, X_test, y_test):             Predict with H2O automl
    run_train_stack(X_train, y_train, X_test, y_test):              Predict with stack ensemble (CatBoostRegressor -> LinearRegression)
    run_train_stack_nn1embed_catboost(X_train, y_train, X_test, y_test):  Predict with stack ensemble (neural network (1 embedding) -> CatBoostRegressor)
    run_train_stack_nn2embed_catboost(X_train, y_train, X_test, y_test):  Predict with stack ensemble (neural network (2 embeddings) -> CatBoostRegressor)

Class:
    HyperRegressor(keras_tuner.HyperModel): Class to custom define HyperModel so as to pass in self-defined metric 'r_squared'

"""

import pandas as pd
import numpy as np
import random
import json
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import tensorflow as tf
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Input, Flatten, Concatenate, Embedding, Conv1D, MaxPooling1D

from scikeras.wrappers import KerasRegressor

import keras_tuner

import h2o
from h2o.automl import H2OAutoML

from helper import display_cv_score, r2_rmse_score, r_squared, rmse, apply_log1p
from helper import make_nn_model_1output, make_nn_model_3outputs, make_tunable_nn_model, make_nn_model_embed_cat, make_nn_model_embed_cat_genotype
from helper import scale_X_and_Y, perform_cross_validation, perform_train_test
from config import Config


########################################
##### Import variables from config
########################################

config = Config()

NUM_FOLDS = config.config_data['NUM_FOLDS']
LEARNING_RATE_PARAM = config.config_data['LEARNING_RATE_PARAM']
N_ESTIMATORS_PARAM = config.config_data['N_ESTIMATORS_PARAM']
TREES_DEPTH_PARAM = config.config_data['TREES_DEPTH_PARAM']
L2_LEAF_REG_PARAM = config.config_data['L2_LEAF_REG_PARAM']

BATCH_SIZE = config.config_data['BATCH_SIZE']
AUG_BATCH_SIZE = config.config_data['AUG_BATCH_SIZE']
EPOCHS = config.config_data['EPOCHS']

ONEHOT_FEATURES = config.config_data['ONEHOT_FEATURES']


SEED = config.config_data['SEED']

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# tf.config.run_functions_eagerly(True)

########################################
##### Functions to run selected modes
########################################


def run_train(X_train, y_train, X_test, y_test):
    """
    Use CatBoostRegressor to predict. 
    Perform both cross validation and train_test_split predictions

    Args:
        X_train (DataFrame or array): Features in train data
        y_train (DataFrame or array): Targets in train data
        X_test (DataFrame or array): Features in test data
        y_test (DataFrame or array): Targets in test data
    """

    # Create instance of CatBoostRegressor
    regressor = CatBoostRegressor(n_estimators=1000, loss_function='MultiRMSE', verbose=0)

    # Perform cross validation on train data
    perform_cross_validation(X_train, y_train, regressor)

    # Fit regressor on train data, then validate with test data
    perform_train_test(X_train, y_train, X_test, y_test, regressor, "CatBoostRegressor")

def run_train_multi(X_train, y_train, X_test, y_test):
    """
    Use a variety of estimators to predict. 
    Only perform train_test_split predictions

    Args:
        X_train (DataFrame or array): Features in train data
        y_train (DataFrame or array): Targets in train data
        X_test (DataFrame or array): Features in test data
        y_test (DataFrame or array): Targets in test data
    """

    battery = {
    "CatBoostRegressor": CatBoostRegressor(n_estimators=1000, loss_function='MultiRMSE', verbose=0),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=1000),
    "AdaBoostRegressor": MultiOutputRegressor(AdaBoostRegressor(n_estimators=1000)),
    "XGBRegressor": XGBRegressor(n_estimators=1000),
    "LinearRegression": LinearRegression(),
    "ElasticNet": ElasticNet(),
    "Support Vector Machine": MultiOutputRegressor(SVR(C=0.8, epsilon=0.1))
    }

    for key, regressor in battery.items():
        perform_train_test(X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy(), regressor, key)

def run_train_gridsearch(X_train, y_train, X_test, y_test):
    """
    Perform GridSearchCV with CatBoostRegressor.
    Hyperparameters from config

    Args:
        X_train (DataFrame or array): Features in train data
        y_train (DataFrame or array): Targets in train data
        X_test (DataFrame or array): Features in test data
        y_test (DataFrame or array): Targets in test data
    """

    # Create instance of CatBoostRegressor
    regressor = CatBoostRegressor(loss_function='MultiRMSE', verbose=0)

    param_grid = {
        'n_estimators' : N_ESTIMATORS_PARAM,
        # 'learning_rate': LEARNING_RATE_PARAM,
        'depth': TREES_DEPTH_PARAM,
        # 'l2_leaf_reg': L2_LEAF_REG_PARAM
    }

    grid_search = GridSearchCV(
                    estimator=regressor,
                    param_grid = param_grid,
                    cv=5,
                    scoring='r2'
                    )
    
    grid_search.fit(X_train, y_train)

    # Print best parameters and score
    print('Best parameters from GridSearchCV: ', grid_search.best_params_)
    print('Best score from GridSearchCV: ', grid_search.best_score_)

    # Evaluate the best model on X_test
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    # y_pred output from pipe is numpy.array, so need to convert
    y_pred = pd.DataFrame(y_pred, columns=['yield', 'titer', 'rate'])

    # Evaluate predictions with r2 score and root mean square error
    r2_rmse_score(y_test, y_pred, "Best GridSearchCV Model")

def run_train_bayes(X_train, y_train, X_test, y_test):
    """    
    Perform BayesSearchCV with CatBoostRegressor.

    Args:
        X_train (DataFrame or array): Features in train data
        y_train (DataFrame or array): Targets in train data
        X_test (DataFrame or array): Features in test data
        y_test (DataFrame or array): Targets in test data
    """

    # Create instance of CatBoostRegressor
    regressor = CatBoostRegressor(loss_function='MultiRMSE', verbose=0)

    search_space = {
        'n_estimators' : Integer(1000, 2000),
        'learning_rate': Real(0.05, 0.2),
        'depth': Integer(4, 10),
        'l2_leaf_reg': Integer(1, 10)
    }

    bayes_search = BayesSearchCV(regressor, search_space, n_iter=32, scoring='r2', n_jobs=-1, cv=5)

    def on_step(optim_result):
        """Callback to print score after each iterations"""
        print(f"Iteration {optim_result.func_vals.shape[0]}, Best R-Squared Score: {-optim_result.fun.max()}")
    
    np.int = int # Because np.int deprecated

    bayes_search.fit(X_train, y_train, callback=on_step) # on_step prints score after each iteration
    print(bayes_search.best_params_)
    print(bayes_search.best_score_)

    # Evaluate best model on X_test
    best_model = bayes_search.best_estimator_

    y_pred = best_model.predict(X_test)

    # y_pred output from pipe is numpy.array, so need to convert
    y_pred = pd.DataFrame(y_pred, columns=['yield', 'titer', 'rate'])

    # Evaluate predictions with r2 score and root mean square error
    r2_rmse_score(y_test, y_pred, "Best BayesSearch Model")

def run_train_nn(X_train, y_train, X_test, y_test):
    """
    Make predictions with Keras neural network comprising mainly of
    Dense, Dropout and BatchNormalization layers 

    Args:
        X_train (DataFrame or array): Features in train data
        y_train (DataFrame or array): Targets in train data
        X_test (DataFrame or array): Features in test data
        y_test (DataFrame or array): Targets in test data
    """

    # Makes a big difference to scale the target values
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, all_columns, ScalerY = scale_X_and_Y(X_train, y_train, X_test, y_test, ONEHOT_FEATURES)

    # pca = PCA(n_components=0.95)
    # X_train_pca = pca.fit_transform(X_train_scaled)
    # X_test_pca = pca.transform(X_test_scaled)

    # input_dim = X_train_pca.shape[1]

    model = make_nn_model_3outputs()

    # Fit model
    model.fit(X_train_scaled, y_train_scaled, validation_data=[X_test_scaled, y_test_scaled], batch_size=BATCH_SIZE, epochs=40)

    y_pred = model.predict(X_test_scaled)

    # Evaluate predictions with r2 score and root mean square error
    # y_pred and y_test_scaled converted to pd DataFrame in r2_rmse_score
    r2_rmse_score(y_test_scaled, y_pred, "Neural Network Model")

    y_pred_unscaled = ScalerY.inverse_transform(y_pred)
    y_pred_unscaled = pd.DataFrame(y_pred_unscaled, columns=['yield', 'titer', 'rate'])
    r2_rmse_score(y_test, y_pred_unscaled, "After inverse scaling")

def run_train_embed_nn(X_train, y_train, X_test, y_test):
    """    
    Make predictions with Keras neural network comprising mainly of
    Dense, Dropout and BatchNormalization layers as well as an embedding layer
    to handle the one_hot encoded features

    Args:
        X_train (DataFrame or array): Features in train data
        y_train (DataFrame or array): Targets in train data
        X_test (DataFrame or array): Features in test data
        y_test (DataFrame or array): Targets in test data
    """

    # Receive all_columns back to label the new dataframe according to the new order after ColumnTransformer
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, all_columns, ScalerY = scale_X_and_Y(X_train, y_train, X_test, y_test, ONEHOT_FEATURES)

    X_train_onehot = X_train_scaled[ONEHOT_FEATURES]
    X_train_num = X_train_scaled.drop(ONEHOT_FEATURES, axis=1)

    X_test_onehot = X_test_scaled[ONEHOT_FEATURES]
    X_test_num = X_test_scaled.drop(ONEHOT_FEATURES, axis=1)

    print(X_test_onehot.info())
    print(X_test_num.info())

    num_features_count = len(X_train_num.columns)
    print(num_features_count)
    onehot_features_count = len(X_train_onehot.columns)
    print(onehot_features_count)

    model = make_nn_model_embed_cat(num_features_count, onehot_features_count)

    # Fit model
    model.fit([X_train_num, X_train_onehot], y_train_scaled, validation_data=([X_test_num, X_test_onehot], y_test_scaled), batch_size=BATCH_SIZE, epochs=40)

    y_pred = model.predict([X_test_num, X_test_onehot])

    # Evaluate predictions with r2 score and root mean square error
    r2_rmse_score(y_test_scaled, y_pred, "Neural Network (Embed Onehot) Model")

    y_pred_unscaled = ScalerY.inverse_transform(y_pred)
    y_pred_unscaled = pd.DataFrame(y_pred_unscaled, columns=['yield', 'titer', 'rate'])
    r2_rmse_score(y_test, y_pred_unscaled, "After inverse scaling")

def run_train_embed_genotype_nn(X_train, y_train, X_test, y_test):
    """    Make predictions with Keras neural network comprising mainly of
    Dense, Dropout and BatchNormalization layers as well as 2 embedding layers,
    one to handle onehot_encoded features and another to handle the column of 
    tokenized strain_background_genotype

    Args:
        X_train (DataFrame or array): Features in train data
        y_train (DataFrame or array): Targets in train data
        X_test (DataFrame or array): Features in test data
        y_test (DataFrame or array): Targets in test data
    """
    
    noscale_features = ONEHOT_FEATURES + ["strain_background_genotype_tokenized"]

    # Receive all_columns back to label the new dataframe according to the new order after ColumnTransformer
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, all_columns, ScalerY = scale_X_and_Y(X_train, y_train, X_test, y_test, noscale_features)

    all_columns_except = all_columns.copy()
    all_columns_except.remove("strain_background_genotype_tokenized")
    # columns returned as dtype object. Likely because "strain_background_genotype_tokenized" contains list, thus dtype object
    # Need to change to float or int to feed to keras layers
    for column in all_columns_except:
        X_train_scaled[column] = X_train_scaled[column].astype('float64')
        X_test_scaled[column] = X_test_scaled[column].astype('float64')
    
    # Can expand the array into individual columns (1 column for each element)
    # expanded_train_genotype = X_train_scaled["strain_background_genotype_tokenized"].apply(pd.Series)
    # expanded_train_genotype.columns = [f'genotype_{i+1}' for i in range(expanded_train_genotype.shape[1])]
    # X_train_scaled = X_train_scaled.drop("strain_background_genotype_tokenized", axis=1)
    # X_train_scaled = pd.concat([X_train_scaled, expanded_train_genotype], axis=1)
   

    X_train_onehot = X_train_scaled[ONEHOT_FEATURES]
    # Important for the data to be accepted by Keras Input
    # Change from pd df of dtype object to array of dtype int64
    X_train_genotype = np.array(X_train_scaled["strain_background_genotype_tokenized"].tolist())
    X_train_num = X_train_scaled.drop(noscale_features, axis=1)

    X_test_onehot = X_test_scaled[ONEHOT_FEATURES]
    # Important for the data to be accepted by Keras Input
    # Change from pd df of dtype object to array of dtype int64
    X_test_genotype = np.array(X_test_scaled["strain_background_genotype_tokenized"].tolist())
    X_test_num = X_test_scaled.drop(noscale_features, axis=1)

    num_features_count = len(X_train_num.columns)
    onehot_features_count = len(X_train_onehot.columns)

    model = make_nn_model_embed_cat_genotype(num_features_count, onehot_features_count)

    # Fit the model with data
    model.fit({"num_input": X_train_num, "onehot_input": X_train_onehot, "genotype_input": X_train_genotype}, 
              y_train_scaled, 
              validation_data=({"num_input": X_test_num, "onehot_input": X_test_onehot, "genotype_input": X_test_genotype}, y_test_scaled), 
              batch_size=BATCH_SIZE, epochs=40)

    y_pred = model.predict([X_test_num, X_test_onehot, X_test_genotype])

    # Evaluate predictions with r2 score and root mean square error
    r2_rmse_score(y_test_scaled, y_pred, "Neural Network (Embedded OneHot + Genotype) Model")

    y_pred_unscaled = ScalerY.inverse_transform(y_pred)
    y_pred_unscaled = pd.DataFrame(y_pred_unscaled, columns=['yield', 'titer', 'rate'])
    r2_rmse_score(y_test, y_pred_unscaled, "After inverse scaling")

class HyperRegressor(keras_tuner.HyperModel):
    """
    Create custom HyperModel so that a custom objective can be passed in
    to customize the metric

    Args:
        keras_tuner.HyperModel (class): Superclass
    """
    def build(self, hp):
        """
        Define architecture of neural network and return the model

        Args:
            hp (Hyperparameter Instance): To specify hyperparameters. Automatically passed by keras-tuner

        Returns:
            model (Object): keras neural network
        """
        input_dim = 42
        output_dim = 3

        model = Sequential()

        # Input layer
        model.add(Dense(units=hp.Int('input_dense', min_value=64, max_value=512, step=16), input_dim=input_dim, kernel_initializer='glorot_uniform', activation='tanh'))

        # Tunable number of blocks containing 1 Dropout, 1 BatchNormalization and 1 Dense layer
        # for i in range(hp.Int('norm_dropout_dense', min_value=1, max_value=5, step=1)):
        #     model.add(BatchNormalization())
        #     model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.4, step=0.05)))
        #     model.add(Dense(units=hp.Int('hidden_dense', min_value=64, max_value=4096, step=64), kernel_initializer='glorot_uniform', activation='tanh'))

        # 3-block architecture that works well over various trials
        model.add(BatchNormalization())
        model.add(Dropout(0.15))
        model.add(Dense(units=hp.Int('hidden_dense_1', min_value=64, max_value=4096, step=16), kernel_initializer='glorot_uniform', activation='tanh'))
        model.add(BatchNormalization())
        model.add(Dropout(0.15))
        model.add(Dense(units=hp.Int('hidden_dense_2', min_value=64, max_value=4096, step=16), kernel_initializer='glorot_uniform', activation='tanh'))
        model.add(BatchNormalization())
        model.add(Dropout(0.15))
        model.add(Dense(units=hp.Int('hidden_dense_3', min_value=64, max_value=4096, step=16), kernel_initializer='glorot_uniform', activation='tanh'))
        model.add(Dense(output_dim, activation="tanh"))

        # Output layer
        model.add(Dense(output_dim, activation="tanh"))

        # # Setting up choices for different optimizers
        # hp_optimizer = hp.Choice("Optimizer", values=['Adam', 'SGD'])

        # if hp_optimizer == 'Adam':
        #     hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])
        # elif hp_optimizer == 'SGD':
        #     hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])
        #     nesterov = True
        #     momentum = 0.9

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3])

        model.compile(optimizer='adam', loss="mean_squared_error")
        
        return model

    def fit(self, hp, model, x, y, validation_data, **kwargs):
        model.fit(x, y, **kwargs)
        x_val, y_val = validation_data
        y_pred = model.predict(x_val)
        return {
            'mse': mean_squared_error(y_val, y_pred),
            'r_squared': r2_score(y_val, y_pred)
        }

def run_train_tunable_nn(X_train, y_train, X_test, y_test):
    """    
    Make predictions with tunable Keras Tuner HyperModel comprising mainly of
    Dense, Dropout and BatchNormalization layers 

    Args:
        X_train (DataFrame or array): Features in train data
        y_train (DataFrame or array): Targets in train data
        X_test (DataFrame or array): Features in test data
        y_test (DataFrame or array): Targets in test data
    """

    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled = scale_X_and_Y(X_train, y_train, X_test, y_test)

    # Set up tuner with BayesianOptimization. Can try RandomSearch or Hyberband too.
    tuner = keras_tuner.BayesianOptimization(
        hypermodel=HyperRegressor(),
        objective = keras_tuner.Objective('r_squared', 'max'),
        seed=808,
        max_trials=10,
        directory='../models',
        project_name='keras_tuner_Bayers'
    )

    tuner.search(X_train_scaled, y_train_scaled, epochs=30, batch_size=32, validation_data=(X_test_scaled, y_test_scaled))

    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    print('Keras Tuner Bayes Best hyperparameters: ', best_hyperparameters.values)

    best_model = tuner.get_best_models()[0]
    y_pred = best_model.predict(X_test_scaled)

    # Evaluate predictions with r2 score and root mean square error
    r2_rmse_score(y_test_scaled, y_pred, "Keras Tuner")

def run_train_automl(X_train, y_train, X_test, y_test):
    """    
    Make predictions with H2O automl. Save best model. 

    Args:
        X_train (DataFrame or array): Features in train data
        y_train (DataFrame or array): Targets in train data
        X_test (DataFrame or array): Features in test data
        y_test (DataFrame or array): Targets in test data
    """

    # Only support 1 target at a time

    h2o.init()

    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, all_columns, ScalerY = scale_X_and_Y(X_train, y_train, X_test, y_test)

    # AutoML can only predict 1 target value at a time
    train_df = pd.concat([X_train_scaled, y_train_scaled['yield']], axis=1)

    train_h2o_df = h2o.H2OFrame(train_df)
    test_h2o_df = h2o.H2OFrame(X_test_scaled)

    train_h2o_df.head()

    aml = H2OAutoML(max_models=10, seed=808)
    aml.train(training_frame=train_h2o_df, y='yield')

    lb = aml.leaderboard
    print(lb.head(rows=lb.nrows))

    best_model = aml.get_best_model()

    # Save model
    h2o.save_model(model=best_model, path="../models/automl/best_models", force=True)

def run_train_stack(X_train, y_train, X_test, y_test):
    """    
    Make predictions with stack ensembled where base estimator is CatBoostRegressor and 
    meta estimator is LinearRegression

    Args:
        X_train (DataFrame or array): Features in train data
        y_train (DataFrame or array): Targets in train data
        X_test (DataFrame or array): Features in test data
        y_test (DataFrame or array): Targets in test data
    """

    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, all_columns, ScalerY = scale_X_and_Y(X_train, y_train, X_test, y_test, ONEHOT_FEATURES)

    # Create KerasRegressor
    keras_regressor = KerasRegressor(build_fn=make_nn_model_1output, epochs=40, batch_size=32, verbose=0)

    base_learners = [
        ('cat', CatBoostRegressor(n_estimators=1000, loss_function='MultiRMSE', verbose=0)),
        # ('rf', RandomForestRegressor(n_estimators=1000)),
        # ('nn', keras_regressor)
    ]

    targets = ['yield', 'titer', 'rate']

    stacked_model = []
    predictions = []

    for i, target in enumerate(targets):
        print("Fitting ", target)
        stacked_model.append(StackingRegressor(estimators=base_learners, final_estimator=LinearRegression()))
        stacked_model[i].fit(X_train_scaled, y_train_scaled[target])
        predictions.append(stacked_model[i].predict(X_test_scaled))
    
    y_pred = np.column_stack(predictions)

    # y_pred output from pipe is numpy.array, so need to convert
    y_pred = pd.DataFrame(y_pred, columns=targets)

    # Evaluate predictions with r2 score and root mean square error
    r2_rmse_score(y_test_scaled, y_pred, "Stacked Ensemble")

    y_pred_unscaled = ScalerY.inverse_transform(y_pred)
    y_pred_unscaled = pd.DataFrame(y_pred_unscaled, columns=['yield', 'titer', 'rate'])
    r2_rmse_score(y_test, y_pred_unscaled, "After inverse scaling")

# def run_train_stack_nn_catboost(X_train, y_train, X_test, y_test):

#     embedding_dim = 5
    
#     noscale_features = ONEHOT_FEATURES + ["strain_background_genotype_tokenized"]

#     # Receive all_columns back to label the new dataframe according to the new order after ColumnTransformer
#     X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, all_columns = scale_X_and_Y(X_train, y_train, X_test, y_test, noscale_features)

#     train_num = len(X_train_scaled)
#     test_num = len(X_test_scaled)

#     all_columns_except = all_columns.copy()
#     all_columns_except.remove("strain_background_genotype_tokenized")
#     # columns returned as dtype object. Likely because "strain_background_genotype_tokenized" contains list, thus dtype object
#     # Need to change to float or int to feed to keras layers
#     for column in all_columns_except:
#         X_train_scaled[column] = X_train_scaled[column].astype('float64')
#         X_test_scaled[column] = X_test_scaled[column].astype('float64')
    
#     # Combine train and test data into one df and recover the original split by iloc, not train_test_split
#     # Otherwise simply train test split this combined data will produce a new test set where
#     # many of the rows are actually from train set and the augmented data rows in the train set
#     # will be a form of data leakage
    
#     X_scaled = pd.concat([X_train_scaled, X_test_scaled], axis=0)
#     X_scaled = X_scaled.reset_index(drop=True)
#     y_scaled = pd.concat([y_train_scaled, y_test_scaled], axis=0)
#     y_scaled = y_scaled.reset_index(drop=True)

#     y_rows = len(y_scaled)
#     y_cols = len(y_scaled.columns)

#     # Use K-fold for neural network to define a new feature column without data leakage

#     # Define and initialize the number of folds
#     n_folds = 5
#     kf = KFold(n_splits=n_folds, shuffle=True)

#     # Initialize assays to store nn predictions
#     nn_pred = np.zeros((y_rows, y_cols), dtype='float') 

#     # Loop for K folds
#     for i, (train_index, test_index) in enumerate(kf.split(X_scaled, y_scaled)):
#         print("Fold: ", i+1)
#         # Split the data into training and test sets
#         X_train_kfold, y_train_kfold = X_scaled.iloc[train_index], y_scaled.iloc[train_index]
#         X_test_kfold, y_test_kfold = X_scaled.iloc[test_index], y_scaled.iloc[test_index]

#         X_train_onehot = X_train_kfold[ONEHOT_FEATURES]
#         # Important for the data to be accepted by Keras Input
#         # Change from pd df of dtype object to array of dtype int64
#         X_train_genotype = np.array(X_train_kfold["strain_background_genotype_tokenized"].tolist())
#         X_train_num = X_train_kfold.drop(noscale_features, axis=1)

#         X_test_onehot = X_test_kfold[ONEHOT_FEATURES]
#         # Important for the data to be accepted by Keras Input
#         # Change from pd df of dtype object to array of dtype int64
#         X_test_genotype = np.array(X_test_kfold["strain_background_genotype_tokenized"].tolist())
#         X_test_num = X_test_kfold.drop(noscale_features, axis=1)

#         num_features_count = len(X_train_num.columns)
#         onehot_features_count = len(X_train_onehot.columns)

#         model = make_nn_model_embed_cat_genotype(num_features_count, onehot_features_count, embedding_dim)

#         # Train nn on train fold
#         model.fit({"num_input": X_train_num, "onehot_input": X_train_onehot, "genotype_input": X_train_genotype}, y_train_kfold, 
#             batch_size=160, epochs=20)
        
#         # Make predict on test fold and store predictions
#         nn_pred[test_index] = model.predict([X_test_num, X_test_onehot, X_test_genotype])
    
#     df_nn_pred = pd.DataFrame(nn_pred, columns=['nn_yield', 'nn_titer', 'nn_rate'])

#     # Need to reset index on X_scaled to merge properly
#     X_stacked = pd.concat([X_scaled, df_nn_pred], axis=1)

#     X_stacked_train = X_stacked.iloc[:train_num, :]
#     X_stacked_test = X_stacked.iloc[train_num:, :]

#     y_stacked_train = y_scaled.iloc[:train_num, :]
#     y_stacked_test = y_scaled.iloc[train_num:, :]
    
#     X_stacked_train = X_stacked_train.drop(["strain_background_genotype_tokenized"], axis=1)
#     X_stacked_test = X_stacked_test.drop(["strain_background_genotype_tokenized"], axis=1)

#     # X_stacked_train, X_stacked_test, y_stacked_train, y_stacked_test = train_test_split(X_stacked, y_scaled, test_size=0.3, random_state=33)
    
#     # Create instance of CatBoostRegressor
#     regressor_stack = CatBoostRegressor(n_estimators=1000, loss_function='MultiRMSE', verbose=0)
#     #regressor = CatBoostRegressor(n_estimators=1000, loss_function='MultiRMSE', verbose=0)

#     # Perform cross validation on train data
#     perform_cross_validation(X_stacked_train, y_stacked_train, regressor_stack)

#     # Fit regressor on train data, then validate with test data
#     y_pred_stack = perform_train_test(X_stacked_train, y_stacked_train, X_stacked_test, y_stacked_test, regressor_stack, "Stacked nn + CatBoostRegressor")

#     #X_train_scaled = X_train_scaled.drop(["strain_background_genotype_tokenized"], axis=1)
#     #X_test_scaled= X_test_scaled.drop(["strain_background_genotype_tokenized"], axis=1)

#     #y_pred_rgs = perform_train_test(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, regressor, "CatBoostRegressor")

#     #y_pred_mix_match = pd.concat([y_pred_rgs['yield'], y_pred_stack['titer'], y_pred_rgs['rate']])

#     #r2_rmse_score(y_test_scaled, y_pred_mix_match, "mix & match")

def run_train_stack_nn1embed_catboost(X_train, y_train, X_test, y_test):
    """    
    Make predictions with stack ensemble, manually built by adding predictions of
    Keras neural network as new features and feeding the augmented feature set to
    CatBoostRegressor. Keras neural network comprises mainly of
    Dense, Dropout and BatchNormalization layers as well as 1 embedding layer
    to handle onehot_encoded features.

    Args:
        X_train (DataFrame or array): Features in train data
        y_train (DataFrame or array): Targets in train data
        X_test (DataFrame or array): Features in test data
        y_test (DataFrame or array): Targets in test data
    """

    noscale_features = ONEHOT_FEATURES

    # Receive all_columns back to label the new dataframe according to the new order after ColumnTransformer
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, all_columns, scalerY = scale_X_and_Y(X_train, y_train, X_test, y_test, noscale_features)

 
    # Split X_train into respective clusters of features
    X_train_onehot = X_train_scaled[ONEHOT_FEATURES]
    X_train_num = X_train_scaled.drop(noscale_features, axis=1)

    # Split X_test into respective clusters of features
    X_test_onehot = X_test_scaled[ONEHOT_FEATURES]
    X_test_num = X_test_scaled.drop(noscale_features, axis=1)

    # Count features to define Input shape
    num_features_count = len(X_train_num.columns)
    onehot_features_count = len(X_train_onehot.columns)

    model = make_nn_model_embed_cat(num_features_count, onehot_features_count)

    # Train nn on train fold
    model.fit({"num_input": X_train_num, "onehot_input": X_train_onehot}, y_train_scaled, 
        batch_size=AUG_BATCH_SIZE, epochs=EPOCHS)
    
    # Make predict on test fold and store predictions
    train_pred = model.predict([X_train_num, X_train_onehot])
    test_pred = model.predict([X_test_num, X_test_onehot])
    
    df_train_pred = pd.DataFrame(train_pred, columns=['nn_yield', 'nn_titer', 'nn_rate'])
    df_test_pred = pd.DataFrame(test_pred, columns=['nn_yield', 'nn_titer', 'nn_rate'])

    # Need to reset index on X_scaled to merge properly
    X_stacked_train = pd.concat([X_train_scaled, df_train_pred], axis=1)
    X_stacked_test = pd.concat([X_test_scaled, df_test_pred], axis=1)

    y_stacked_train = y_train_scaled
    y_stacked_test = y_test_scaled
    
    # Create instance of CatBoostRegressor
    regressor_stack = CatBoostRegressor(n_estimators=1000, loss_function='MultiRMSE', verbose=0)
    #regressor = CatBoostRegressor(n_estimators=1000, loss_function='MultiRMSE', verbose=0)

    # Perform cross validation on train data
    perform_cross_validation(X_stacked_train, y_stacked_train, regressor_stack)

    # Fit regressor on train data, then validate with test data
    y_pred_stack = perform_train_test(X_stacked_train, y_stacked_train, X_stacked_test, y_stacked_test, regressor_stack, "Stacked nn + CatBoostRegressor")
    
    # Examine RMSE after inverse scaling
    y_pred_stack_unscaled = scalerY.inverse_transform(y_pred_stack)
    y_pred_stack_unscaled = pd.DataFrame(y_pred_stack_unscaled, columns=['yield', 'titer', 'rate'])
    r2_rmse_score(y_test, y_pred_stack_unscaled, "After inverse scaling")
    
def run_train_stack_nn2embed_catboost(X_train, y_train, X_test, y_test):
    """    
    Make predictions with stack ensemble, manually built by adding predictions of
    Keras neural network as new features and feeding the augmented feature set to
    CatBoostRegressor. Keras neural network comprises mainly of
    Dense, Dropout and BatchNormalization layers as well as 2 embedding layers,
    one to handle onehot_encoded features and another to handle the column of 
    tokenized strain_background_genotype. 

    Args:
        X_train (DataFrame or array): Features in train data
        y_train (DataFrame or array): Targets in train data
        X_test (DataFrame or array): Features in test data
        y_test (DataFrame or array): Targets in test data
    """

    noscale_features = ONEHOT_FEATURES + ["strain_background_genotype_tokenized"]

    # Receive all_columns back to label the new dataframe according to the new order after ColumnTransformer
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, all_columns, scalerY, column_transformer = scale_X_and_Y(X_train, y_train, X_test, y_test, noscale_features)

    # Change feature dtypes to float (except strain_background_genotype_tokenized)
    all_columns_except = all_columns.copy()
    all_columns_except.remove("strain_background_genotype_tokenized")
    # columns returned as dtype object. Likely because "strain_background_genotype_tokenized" contains list, thus dtype object
    # Need to change to float or int to feed to keras layers
    for column in all_columns_except:
        X_train_scaled[column] = X_train_scaled[column].astype('float64')
        X_test_scaled[column] = X_test_scaled[column].astype('float64')

    # Split X_train into respective clusters of features
    X_train_onehot = X_train_scaled[ONEHOT_FEATURES]
    # Important for the data to be accepted by Keras Input
    # Change from pd df of dtype object to array of dtype int64
    X_train_genotype = np.array(X_train_scaled["strain_background_genotype_tokenized"].tolist())
    X_train_num = X_train_scaled.drop(noscale_features, axis=1)

    # Split X_test into respective clusters of features
    X_test_onehot = X_test_scaled[ONEHOT_FEATURES]
    # Important for the data to be accepted by Keras Input
    # Change from pd df of dtype object to array of dtype int64
    X_test_genotype = np.array(X_test_scaled["strain_background_genotype_tokenized"].tolist())
    X_test_num = X_test_scaled.drop(noscale_features, axis=1)

    # Count features to define Input shape
    num_features_count = len(X_train_num.columns)
    onehot_features_count = len(X_train_onehot.columns)

    model = make_nn_model_embed_cat_genotype(num_features_count, onehot_features_count)

    # Train nn on train fold
    model.fit({"num_input": X_train_num, "onehot_input": X_train_onehot, "genotype_input": X_train_genotype}, y_train_scaled, 
        batch_size=AUG_BATCH_SIZE, epochs=EPOCHS)
    
    # Make predict on test fold and store predictions
    train_pred = model.predict([X_train_num, X_train_onehot, X_train_genotype])
    test_pred = model.predict([X_test_num, X_test_onehot, X_test_genotype])
    
    df_train_pred = pd.DataFrame(train_pred, columns=['nn_yield', 'nn_titer', 'nn_rate'])
    df_test_pred = pd.DataFrame(test_pred, columns=['nn_yield', 'nn_titer', 'nn_rate'])

    # Need to reset index on X_scaled to merge properly
    X_stacked_train = pd.concat([X_train_scaled, df_train_pred], axis=1)
    X_stacked_test = pd.concat([X_test_scaled, df_test_pred], axis=1)

    y_stacked_train = y_train_scaled
    y_stacked_test = y_test_scaled
    
    X_stacked_train = X_stacked_train.drop(["strain_background_genotype_tokenized"], axis=1)
    X_stacked_test = X_stacked_test.drop(["strain_background_genotype_tokenized"], axis=1)

    # X_stacked_train, X_stacked_test, y_stacked_train, y_stacked_test = train_test_split(X_stacked, y_scaled, test_size=0.3, random_state=33)
    
    # Create instance of CatBoostRegressor
    regressor_stack = CatBoostRegressor(n_estimators=1000, loss_function='MultiRMSE', verbose=0)
    #regressor = CatBoostRegressor(n_estimators=1000, loss_function='MultiRMSE', verbose=0)

    # Perform cross validation on train data
    perform_cross_validation(X_stacked_train, y_stacked_train, regressor_stack)

    # Fit regressor on train data, then validate with test data
    # perform_train_test also checks for features that can be filtered
    features_to_filter = perform_train_test(X_stacked_train, y_stacked_train, X_stacked_test, y_stacked_test, regressor_stack, "Stacked nn + CatBoostRegressor")

    ###########################################################################
    # Drop features of low importances
    X_train_reduced = X_stacked_train.drop(features_to_filter, axis=1)
    X_test_reduced = X_stacked_test.drop(features_to_filter, axis=1)

    # Create new CatBoostRegressor to train on reduced feature set 
    regressor_reduced = CatBoostRegressor(n_estimators=1000, loss_function='MultiRMSE', verbose=0)
    regressor_reduced.fit(X_train_reduced, y_stacked_train)

    # Predict with new CatBoostRegressor based on reduced feature set
    y_pred_reduced = regressor_reduced.predict(X_test_reduced)
    # y_pred output from pipe is numpy.array, so need to convert
    y_pred_reduced = pd.DataFrame(y_pred_reduced, columns=['yield', 'titer', 'rate'])
    # Evaluate predictions with r2 score and root mean square error
    r2_rmse_score(y_stacked_test, y_pred_reduced, "Stacked nn + CatBoostRegressor (filter features)")
    
    # Examine RMSE after inverse scaling
    y_pred_stack_unscaled = scalerY.inverse_transform(y_pred_reduced)
    y_pred_stack_unscaled = pd.DataFrame(y_pred_stack_unscaled, columns=['yield', 'titer', 'rate'])
    r2_val = r2_rmse_score(y_test, y_pred_stack_unscaled, "After inverse scaling")

    # Initialize or load best r_squared score
    best_r2_val_file = "../models/best_r2_val.json"

    if os.path.exists(best_r2_val_file):
        with open(best_r2_val_file, 'r') as f:
            best_r2_val = json.load(f)['best_r2_val']
    
    else:
        best_r2_val = float('-inf')

    # Check r2_val is the best r2 score so far. If yes, then save the following
    # 1. ColumnTransformer (from scale_X_and_y) that performs log1p and Standard Scaling
    # 2. ScalerY (from scale_X_and_y) to scale y before prediction and inverse-transform after prediction
    # 3. Set of low importance features (from perform_train_test) to drop
    # 4. Keras model (nn2emebed). Contain custom metric r2_squared which needs to be registered in keras scope
    # 5. CatBootRegressor (from regressor_reduced trained on reduced feature set)
    # 6. New r-squared score and associated model parameters
    if r2_val > best_r2_val:
        best_r2_val = r2_val
        # Save columnTransformer
        joblib.dump(column_transformer, '../models/best_column_transformer.joblib')
        # Save fitted ScalerY
        joblib.dump(scalerY, '../models/best_scalerY.joblib')
        # Save fitted neural network model
        model.save('../models/best_nn2embed.keras')
        # Save fitted CatBoostRegressor
        regressor_reduced.save_model('../models/best_regressor.bin')


        # Save features to filter
        with open('../models/features_to_filter.json', 'w') as f:
            json.dump(features_to_filter, f)

        # Save r2 score and model parameters
        with open(best_r2_val_file, 'w') as f:
            json.dump({'best_r2_val': best_r2_val,
                       'model_name': "stack_nn2embed_catboost",
                       'data_augmentation': config.config_data["AUGMENT_NUM"]
                       }, f)

def run_predict(X_test, y_test):

    noscale_features = ONEHOT_FEATURES + ["strain_background_genotype_tokenized"]
    numerical_features = [col for col in X_test.columns if col not in noscale_features]

    # Perform ColumnTransform and scaling on Y

    # Load column_transformer
    column_transformer = joblib.load('../models/best_column_transformer.joblib')
    all_columns = numerical_features + noscale_features
    
    X_test_reordered = X_test.reindex(columns=all_columns)
    X_test_scaled = column_transformer.transform(X_test_reordered)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=all_columns)

    # Load scalerY
    scalerY = joblib.load('../models/best_scalerY.joblib')
    y_test_scaled = scalerY.transform(y_test)
    y_test_scaled = pd.DataFrame(y_test_scaled, columns=y_test.columns)

    # Change feature dtypes to float (except strain_background_genotype_tokenized)
    all_columns_except = all_columns.copy()
    all_columns_except.remove("strain_background_genotype_tokenized")
    # columns returned as dtype object. Likely because "strain_background_genotype_tokenized" contains list, thus dtype object
    # Need to change to float or int to feed to keras layers
    for column in all_columns_except:
        X_test_scaled[column] = X_test_scaled[column].astype('float64')
    
    # Split X_test into respective clusters of features
    X_test_onehot = X_test_scaled[ONEHOT_FEATURES]
    # Important for the data to be accepted by Keras Input
    # Change from pd df of dtype object to array of dtype int64
    X_test_genotype = np.array(X_test_scaled["strain_background_genotype_tokenized"].tolist())
    X_test_num = X_test_scaled.drop(noscale_features, axis=1)

    # Load best nn2embed model
    #with keras.utils.custom_object_scope({'r_sqaured': r_squared}):
    loaded_model = load_model("../models/best_nn2embed.keras")
    test_pred = loaded_model.predict([X_test_num, X_test_onehot, X_test_genotype])
    df_test_pred = pd.DataFrame(test_pred, columns=['nn_yield', 'nn_titer', 'nn_rate'])

    # Concat nn2embed predictions to original features
    X_stacked_test = pd.concat([X_test_scaled, df_test_pred], axis=1)
    X_stacked_test = X_stacked_test.drop(["strain_background_genotype_tokenized"], axis=1)

    y_stacked_test = y_test_scaled

    # Load best CatBoostRegressor 
    loaded_regressor = CatBoostRegressor()
    loaded_regressor.load_model('../models/best_regressor.bin')

    # Load features_to_filter
    with open('../models/features_to_filter.json', 'r') as f:
        features_to_filter = json.load(f)
    
    X_test_reduced = X_stacked_test.drop(features_to_filter, axis=1)
    print(X_test_reduced.info())

    # Use loaded regressor to produce predictions and convert to pd DataFrame
    y_pred_reduced = loaded_regressor.predict(X_test_reduced)
    y_pred_reduced = pd.DataFrame(y_pred_reduced, columns=['yield', 'titer', 'rate'])

    # Evaluate predictions with r2 score and root mean square error
    r2_rmse_score(y_stacked_test, y_pred_reduced, 'Saved nn2embed + CatBoost (filter features)')

    # Examine RMSE after inverse scaling
    y_pred_unscaled = scalerY.inverse_transform(y_pred_reduced)
    y_pred_unscaled = pd.DataFrame(y_pred_unscaled, columns=['yield', 'titer', 'rate'])
    r2_rmse_score(y_test, y_pred_unscaled, "After inverse scaling")