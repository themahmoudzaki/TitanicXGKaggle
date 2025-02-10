from imports import *

def xgboost_model(tt_files, save_folder):
    train_X, test_X, train_y, test_y = tt_files

    grid_params = {
        'n_estimators': 5,
        'max_depth':    10,
        'learning_rate': 1e-1,
        'objective': 'binary:logistic'
    }
    model = XGBClassifier(**grid_params)
    model.fit(train_X, train_y)
    y_hat = model.predict(test_X)
    classification = classification_report(test_y, y_hat)
    print(f'Test classification: \n{classification}')

    return model