from imports import *

def xgboost_model(tt_files, save_folder):
    train_X, test_X, train_y, test_y = tt_files

    train_X = cp.array(train_X)
    test_X  = cp.array(test_X)

    # param_grid = {
    #     'n_estimators':     [10, 50, 100, 200, 1000],
    #     'max_depth':        [5, 7, 10, 15, 35],
    #     'learning_rate':    [1.5e-1, 2e-1, 3e-1, 5e-1, 1],
    # }
    param_grid = {
        'n_estimators':     [10],
        'max_depth':        [5],
        'learning_rate':    [1.5e-1],
        
    }
    model = XGBClassifier(
        objective='binary:logistic',
        missing=-999,
        device='cuda',
    )
    grid_model = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=5,
        cv=5,
        verbose=2,
    )
    grid_model.fit(train_X, train_y, eval_set=[(test_X, test_y)], verbose=False)
    model = grid_model.best_estimator_

    print(grid_model.best_params_)
    y_hat = model.predict(test_X)
    print(classification_report(test_y, y_hat))

    model.save_model(f'{save_folder}XGBmodel.json')
    return model