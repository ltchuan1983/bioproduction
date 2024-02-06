import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input, Flatten, Concatenate, Embedding, Conv1D, MaxPooling1D

from scikeras.wrappers import KerasRegressor

import keras_tuner

import h2o
from h2o.automl import H2OAutoML

from pipelines import create_preprocessor, create_pipe
from helper import display_cv_score, r2_rmse_score, r_squared, rmse, make_tunable_nn_model

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
    "ElasticNet": ElasticNet(),
    "Support Vector Machine": MultiOutputRegressor(SVR(C=0.8, epsilon=0.1))
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

def scale_X_and_Y(X_train, y_train, X_test, y_test):

    scalerX = StandardScaler()
    X_train_scaled = scalerX.fit_transform(X_train)
    X_test_scaled = scalerX.transform(X_test)

    scalerY = StandardScaler()
    y_train_scaled = scalerY.fit_transform(y_train)
    y_test_scaled = scalerY.transform(y_test)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled

def make_nn_model_1output():

    input_dim = 42
    output_dim = 1

     # tanh performs much better than ReLU and LeakyReLU
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(2048, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(512, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(64, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(Dense(output_dim, activation="tanh"))

    # Compile the model
    # r_squared uses r2_score from sklearn.preprocessing, not custom defiend
    model.compile(optimizer='adam', loss="mean_squared_error", metrics=[r_squared])

    return model

def make_nn_model_3outputs():

    input_dim = 42
    output_dim = 3

     # tanh performs much better than ReLU and LeakyReLU
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(2048, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(512, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(64, kernel_initializer='glorot_normal', activation='tanh'))
    model.add(Dense(output_dim, activation="tanh"))

    # Compile the model
    # r_squared uses r2_score from sklearn.preprocessing, not custom defiend
    model.compile(optimizer='adam', loss="mean_squared_error", metrics=[r_squared])

    return model

def run_train_nn(X_train, y_train, X_test, y_test):

    np.random.seed(808)
    tf.random.set_seed(808)

    # Makes a big difference to scale the target values
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled = scale_X_and_Y(X_train, y_train, X_test, y_test)

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

def run_train_embed_nn(X_train, y_train, X_test, y_test):

    embedding_dim = 5

    onehot_features = ['dir_evo', 'Mod_path_opt', 'reactor_type_1.0', 'reactor_type_2.0', 'reactor_type_3.0', 
                       'media_LB', 'media_M9', 'media_MOPS', 'media_NBS', 'media_RICH',
                       'media_TB', 'media_YE', 'oxygen_1.0', 'oxygen_2.0', 'oxygen_3.0']
    

    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled = scale_X_and_Y(X_train, y_train, X_test, y_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    y_train_scaled = pd.DataFrame(y_train_scaled, columns=y_train.columns)
    y_test_scaled = pd.DataFrame(y_test_scaled, columns=y_test.columns)

    X_train_onehot = X_train_scaled[onehot_features]
    X_train_num = X_train_scaled.drop(onehot_features, axis=1)

    X_test_onehot = X_test_scaled[onehot_features]
    X_test_num = X_test_scaled.drop(onehot_features, axis=1)

    print(X_test_onehot.info())
    print(X_test_num.info())

    num_features_count = len(X_train_num.columns)
    onehot_features_count = len(X_train_onehot.columns)

    # Define input layers
    num_input = Input(shape=(num_features_count, ), name='num_input')
    onehot_input = Input(shape=(onehot_features_count, ), name='onehot_input')

    # Define embedding layer for one-hot features
    # First step will produce output with shape (batch size, input_length, output_dim) (each dense vector represents each column)
    embedding_output = Embedding(input_dim=onehot_features_count, output_dim=embedding_dim, input_length=onehot_features_count)(onehot_input)
    # Flatten into (batch size, batch size x input_length)
    embedding_output = Flatten()(embedding_output)

    # Layer to concatenate continous_input and embedding output
    concatenated = Concatenate()([num_input, embedding_output])

    # Additional blocks of Dense, Dropout and BatchNormalization Layers
    concatenated = Dense(64, kernel_initializer='glorot_normal', activation='tanh')(concatenated)
    concatenated = Dropout(0.15)(concatenated)
    concatenated = BatchNormalization()(concatenated)

    concatenated = Dense(1024, kernel_initializer='glorot_normal', activation='tanh')(concatenated)
    concatenated = Dropout(0.15)(concatenated)
    concatenated = BatchNormalization()(concatenated)

    concatenated = Dense(256, kernel_initializer='glorot_normal', activation='tanh')(concatenated)
    concatenated = Dropout(0.15)(concatenated)
    concatenated = BatchNormalization()(concatenated)

    concatenated = Dense(32, kernel_initializer='glorot_normal', activation='tanh')(concatenated)

    # Output layer
    output = Dense(3, activation='linear')(concatenated)

    # Deifine model
    model = Model(inputs=[num_input, onehot_input], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[r_squared])

    # Fit model
    model.fit([X_train_num, X_train_onehot], y_train_scaled, validation_data=([X_test_num, X_test_onehot], y_test_scaled), batch_size=BATCH_SIZE, epochs=40)

    y_pred = model.predict([X_test_num, X_test_onehot])

    # Evaluate predictions with r2 score and root mean square error
    r2_rmse_score(y_test_scaled, y_pred, "Neural Network Model")

class HyperRegressor(keras_tuner.HyperModel):
    def build(self, hp):
        input_dim = 42
        output_dim = 3

        np.random.seed(808)
        tf.random.set_seed(808)

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

    np.random.seed(808)
    tf.random.set_seed(808)

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

    # Only support 1 target at a time

    h2o.init()

    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled = scale_X_and_Y(X_train, y_train, X_test, y_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    y_train_scaled = pd.DataFrame(y_train_scaled, columns=y_train.columns)
    y_test_scaled = pd.DataFrame(y_test_scaled, columns=y_test.columns)

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

    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled = scale_X_and_Y(X_train, y_train, X_test, y_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    y_train_scaled = pd.DataFrame(y_train_scaled, columns=y_train.columns)
    y_test_scaled = pd.DataFrame(y_test_scaled, columns=y_test.columns)

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