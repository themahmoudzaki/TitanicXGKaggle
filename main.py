from imports import *

TRAIN_DATA_FILE     = 'Data/train.csv'
TEST_DATA_FILE      = 'Data/test.csv'
UPDATED_DATA_FOLDER = 'Updated_Data/'
MODEL_FOLDER        = 'Model/'

def main():
    print('\n\n------------------------------------------\n\n')
    args = sys.argv
    train_flag      = 0
    process_flag    = 0
    for arg in args:
        if      arg == '-T':    train_flag = 1
        elif    arg == '-PD':   process_flag = 1
    
    print('Preprocessing Training Data: ')
    if process_flag: train_X, test_X, train_y, test_y = preprocess_data(file=TRAIN_DATA_FILE, split=(0.8, 0.2), save_folder=UPDATED_DATA_FOLDER)
    else:
        train_X = np.load(f'{UPDATED_DATA_FOLDER}train_X')
        test_X  = np.load(f'{UPDATED_DATA_FOLDER}test_X')
        train_y = np.load(f'{UPDATED_DATA_FOLDER}train_y')
        test_y  = np.load(f'{UPDATED_DATA_FOLDER}test_y')
    model_files = [train_X, test_X, train_y, test_y]
    print('Training Model: ')
    if train_flag: model = xgboost_model(tt_files=model_files, save_folder=MODEL_FOLDER)
    else: model = XGBClassifier().load_model(f'{MODEL_FOLDER}XGBmodel.json')

    print('Preprocessing Test Data: ')
    X = preprocess_data(file=TEST_DATA_FILE, split=(1, 0.0), save_folder='')
    X = cp.asarray(X)

    print('Predicting Y_hat: ')
    y_hat = model.predict(X)
    csv_predictions = pd.DataFrame({
        'PassengerId':  pd.read_csv(TEST_DATA_FILE)['PassengerId'],
        'Survived':     y_hat
    })
    csv_predictions.to_csv('Predictions.csv', index=False)
    
    print('Saved prediction!')

    print('\n\n------------------------------------------\n\n')
    return 0

if __name__ == '__main__': main()
