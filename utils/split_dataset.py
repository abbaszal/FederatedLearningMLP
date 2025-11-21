import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=42)

    args = parser.parse_args()

    data_dir = 'HumanGaitDataBase/Data'
    test_size = args.test_size
    random_state = args.random_seed


    if not os.path.exists('metadata'):
        os.makedirs('metadata')


    activity_dict = {
        1: 'Walking', 2: 'Running', 3: 'Going up', 4: 'Going down', 
        5: 'Sitting', 6: 'Sitting down', 7: 'Standing up', 8: 'Standing',
        9: 'Bycicling', 10: 'Up by elevator', 11: 'Down by elevator', 
        12: 'Sitting in car'
    }


    common_activities = [1, 3, 4, 5, 6, 7, 8]

    for user_id in range(1, 18 + 1):


        user_id_files = []
        for filename in os.listdir(data_dir):
            if not filename.endswith('.txt'):
                continue

            parts = filename.split('_')

            if len(parts) < 4:
                continue


            try:
                file_user_id = int(parts[-2])
            except ValueError:
                continue

            if file_user_id == user_id:
                user_id_files.append(os.path.join(data_dir, filename))


        df = pd.DataFrame()
        for filename in user_id_files:
            df = pd.concat(
                [df, pd.read_csv(filename, sep='\t', skiprows=3)],
                axis=0
            )

        df = df[df['act'].isin(common_activities)].reset_index(drop=True)
        df['act'] = df['act'].map(activity_dict)

        train, test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['act']
        )

  
        assert len(train['act'].unique()) == len(common_activities)
        assert len(test['act'].unique()) == len(common_activities)

        train.to_csv(f'metadata/train_{user_id:02d}.csv', index=False)
        test.to_csv(f'metadata/test_{user_id:02d}.csv', index=False)

    print('Dataset split completed. Train and test files are saved in the metadata directory.')
