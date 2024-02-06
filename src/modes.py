import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import r2_score

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization


from pipelines import create_preprocessor, create_pipe
from helper import display_cv_score, r2_rmse_score, r2_rmse_score2, r_squared, rmse

tf.config.run_functions_eagerly(True)

NUM_FOLDS = 8
LEARNING_RATE_PARAM = [0.05, 0.1, 0.2]
N_ESTIMATORS_PARAM = [1000, 2000]
TREES_DEPTH_PARAM = [4, 6, 8]
L2_LEAF_REG_PARAM = [1, 5, 10]

BATCH_SIZE = 32


def perform_cross_validation(X_train, y_train, regressor):

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=40)

    cross_val_r2 = cross_val_score(regressor, X_train, y_train, cv=kf, scoring='r2')
    cross_val_negrmse = cross_val_score(regressor, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')

    display_cv_score(cross_val_r2, "R-squared")
    display_cv_score(-cross_val_negrmse, "RMSE")

def perform_train_test(X_train, y_train, X_test, y_test, regressor, regressor_name):

    #pipe = create_pipe(regressor)
    regressor.fit(X_train, y_train) # should save model

    y_pred = regressor.predict(X_test)

    # y_pred output from pipe is numpy.array, so need to convert
    y_pred = pd.DataFrame(y_pred, columns=['yield', 'titer', 'rate'])

    print(y_pred.head(20))

    # Evaluate predictions with r2 score and root mean square error
    r2_rmse_score(y_test, y_pred, regressor_name)

def run_train(X_train, y_train, X_test, y_test):

    # Create instance of CatBoostRegressor
    regressor = CatBoostRegressor(n_estimators=1000, loss_function='MultiRMSE', verbose=0)

    # Perform cross validation on train data
    perform_cross_validation(X_train, y_train, regressor)

    # Fit regressor on train data, then validate with test data
    perform_train_test(X_train, y_train, X_test, y_test, regressor, "CatBoostRegressor")


def run_train_multi(X_train, y_train, X_test, y_test):

    battery = {
    "CatBoostRegressor": CatBoostRegressor(n_estimators=1000, loss_function='MultiRMSE', verbose=0),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=1000),
    "AdaBoostRegressor": MultiOutputRegressor(AdaBoostRegressor(n_estimators=1000)),
    "XGBRegressor": XGBRegressor(n_estimators=1000),
    "LinearRegression": LinearRegression(),
    "ElasticNet": ElasticNet()
    }

    for key, regressor in battery.items():
        perform_train_test(X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy(), regressor, key)

def run_train_gridsearch(X_train, y_train, X_test, y_test):

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

    np.random.seed(808)
    tf.random.set_seed(808)

    input_dim = len(X_train.columns)
    output_dim = len(y_train.columns)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Makes a big different to scale the target values
    scaler_Y = StandardScaler()
    y_train_scaled = scaler_Y.fit_transform(y_train)
    y_test_scaled = scaler_Y.transform(y_test)

    # tanh performs much better than ReLU and LeakyReLU

    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, kernel_initializer='glorot_uniform', activation='tanh'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.15))
    # model.add(Dense(256, kernel_initializer='glorot_uniform', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(2048, kernel_initializer='glorot_uniform', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(512, kernel_initializer='glorot_uniform', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(64, kernel_initializer='glorot_uniform', activation='tanh'))
    model.add(Dense(output_dim, activation="tanh"))

    # Compile the model
    # r_squared uses r2_score from sklearn.preprocessing, not custom defiend
    model.compile(optimizer='adam', loss="mean_squared_error", metrics=[r_squared])

    # Fit model
    model.fit(X_train_scaled, y_train_scaled, validation_data=[X_test_scaled, y_test_scaled], batch_size=BATCH_SIZE, epochs=40)

    y_pred = model.predict(X_test_scaled)

    # y_pred and y_test_scaled are numpy.array, so need to convert
    y_pred = pd.DataFrame(y_pred, columns=['yield', 'titer', 'rate'])
    y_test_scaled = pd.DataFrame(y_test_scaled, columns=['yield', 'titer', 'rate'])

    # Evaluate predictions with r2 score and root mean square error
    r2_rmse_score(y_test_scaled, y_pred, "Neural Network Model")





