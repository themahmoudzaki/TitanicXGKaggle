from imports import *

COLUMNS_TO_DROP = ['PassengerId', 'Name', 'Ticket', 'Cabin']
SEX_MAP         = {'male': 0, 'female': 1}
EMBARKED_MAP    = {'C': 0, 'S': 1, 'Q': 2}
FILE_NAME       = ['train_X', 'test_X', 'train_y', 'test_y']

def preprocess_data(file, split, save_folder):
    return_list     = [0, 0, 0, 0]
    
    df = pd.read_csv(file)
    print('\n\n')

    print('Before:')
    print(df.head())
    df.info()
    
    print('\n\n')

    if split[1] != 0.0:
        train, test = train_test_split(df, train_size=split[0], test_size=split[1])
    else: return preprocess_data_helper(df)

    if 'Survived' in train.columns:

        train_y = np.array(train['Survived'].values, dtype=np.int32)
        test_y  = np.array(test['Survived'].values, dtype=np.int32)

        train = train.drop(columns=['Survived'])
        test = test.drop(columns=['Survived'])

        return_list[2] = train_y 
        return_list[3] = test_y  

    df = [train, test]

    for i, data in enumerate(df): return_list[i] = preprocess_data_helper(data)

    


    if save_folder != '':
        for i, np_list in enumerate(return_list):
            np.save(f'{save_folder}{FILE_NAME[i]}', np_list, allow_pickle=True)

    return return_list



def preprocess_data_helper(data):
    data = data.drop(columns=COLUMNS_TO_DROP)

    data['Sex'] = data['Sex'].map(SEX_MAP)
    data['Embarked'] = data['Embarked'].map(EMBARKED_MAP)

    print('\n\n')

    print(f'After: ')

    data.info()

    print('\n\n')

    return np.array(data.values, dtype=np.float32)

    

