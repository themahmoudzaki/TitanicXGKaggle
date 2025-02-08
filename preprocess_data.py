from imports import *

def preprocess_data(file, split, save_folder):
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    sex_map         = {'male': 0, 'female': 1}
    embarked_map    = {'C': 0, 'S': 1, 'Q': 2}
    file_name       = ['train_X', 'test_X', 'train_y', 'test_y']
    return_list     = [0, 0, 0, 0]
    
    df = pd.read_csv(file)
    print('\n\n')

    print('Before:')
    print(df.head())
    df.info()
    
    print('\n\n')


    train, test = train_test_split(df, train_size=split[0], test_size=split[1])
    if 'Survived' in train:
        train_y = train['Survived'].values
        test_y  = test['Survived'].values
        return_list[2] = train_y 
        return_list[3] = test_y  

        train = train.drop(columns=['Survived'])
        test = test.drop(columns=['Survived'])

    df = [train, test]
    for i, data in enumerate(df):
        data = data.drop(columns=columns_to_drop)

        data['Sex'] = data['Sex'].map(sex_map)
        data['Embarked'] = data['Embarked'].map(embarked_map)

        print('\n\n')

        print(f'After {i}: ')
        data.info()

        print('\n\n')

        data = np.array(data, dtype=np.float32)

    return_list[0] = train
    return_list[1] = test
    if file_name != '':
        for i, np_list in enumerate(return_list):
            np.save(f'{save_folder}{file_name[i]}', np_list, allow_pickle=True)

    return return_list

    
    

