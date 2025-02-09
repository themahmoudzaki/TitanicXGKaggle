from imports import *

def xgboost_model(tt_files, save_folder):
    train_X, test_X, train_y, test_y = tt_files

    grid_params = {
        'n_estimators': 2,
        'max_depth':    5,
        'learning_rate': 1e-3,
        'objective': 'binary:logistic'
    }
    model = XGBClassifier(**grid_params)
    model.fit(train_X, train_y)
    return model