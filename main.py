from imports import *

TRAIN_DATA_FILE     = 'Data/train.csv'
UPDATED_DATA_FOLDER = 'Updated_Data/'

def main():
    args = sys.argv
    train_flag      = 0
    process_flag    = 0
    for arg in args:
        if      arg == '-T':    train_flag = 1
        elif    arg == '-PD':   process_flag = 1
    

    if process_flag: train_X, test_X, train_y, test_y = preprocess_data(file=TRAIN_DATA_FILE, split=(0.8, 0.2), save_folder=UPDATED_DATA_FOLDER)
    else:
        train_X = np.load(f'{UPDATED_DATA_FOLDER}train_X')
        test_X  = np.load(f'{UPDATED_DATA_FOLDER}test_X')
        train_y = np.load(f'{UPDATED_DATA_FOLDER}train_y')
        test_y  = np.load(f'{UPDATED_DATA_FOLDER}test_y')
    model_files = [train_X, test_X, train_y, test_y]
    if train_flag: xgboost_model(files=model_files)
    
    return 0

if __name__ == '__main__': main()
