import pandas as pd

from sklearn.model_selection import cross_val_score, KFold

from pipelines import create_preprocessor, create_pipe
from helper import display_cv_score, r2_rmse_score

NUM_FOLDS = 8

def perform_cross_validation(X_train, y_train, regressor):

    preprocessor = create_preprocessor()

    X_train = preprocessor.fit_transform(X_train)

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=40)

    cross_val_r2 = cross_val_score(regressor, X_train, y_train, cv=kf, scoring='r2')
    cross_val_negrmse = cross_val_score(regressor, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')

    display_cv_score(cross_val_r2, "R-squared")
    display_cv_score(-cross_val_negrmse, "RMSE")

def perform_train_test(X_train, y_train, X_test, y_test, regressor, regressor_name):

    pipe = create_pipe(regressor)
    pipe.fit(X_train, y_train) # should save model

    y_pred = pipe.predict(X_test)

    # y_pred output from pipe is numpy.array, so need to convert
    y_pred = pd.DataFrame(y_pred, columns=['yield', 'titer', 'rate'])

    # Evaluate predictions with r2 score and root mean square error
    r2_rmse_score(y_test, y_pred, regressor_name)