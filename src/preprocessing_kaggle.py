# -*- coding=utf-8 -*-

from preprocessing import *
import os
import numpy as np
import pickle
import random
import shutil
import pandas as pd

def make_subset(kaggle_data_path, users_num, force=False, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    site_freq_path = os.path.join(kaggle_data_path, 'site_freq.pkl')
    sessions_path = os.path.join(kaggle_data_path, 'train_sessions_u{}.csv'.format(users_num))
    users_path = os.path.join(kaggle_data_path, 'train', 'other_user_logs')

    if not os.path.exists(sessions_path) or force:
        tmp_src = os.path.join(kaggle_data_path, 'tmp_users')
        if not os.path.exists(tmp_src):
            os.mkdir(tmp_src)
        else:
            shutil.rmtree(tmp_src)
            os.mkdir(tmp_src)

        users = random.sample(os.listdir(users_path), users_num - 1)
        for filename in users:
            shutil.copyfile(os.path.join(users_path, filename),
                            os.path.join(tmp_src, filename))
        shutil.copyfile(os.path.join(kaggle_data_path, 'train', 'Alice_log.csv'),
                        os.path.join(tmp_src, 'user0000.csv'))

        train_data = prepare_train_set(tmp_src,
                                       site_freq_path,
                                       session_length=10, session_timespan=30*60)
        output = train_data.drop(columns=['user_id'])
        target = np.zeros((len(train_data), ), dtype=int)
        target[train_data['user_id'] == 0] = 1
        output['target'] = target

        output.to_csv(sessions_path, index_label='session_id')
        shutil.rmtree(tmp_src)


if __name__ == '__main__':
    import argparse

    src_folder = os.path.join('.', '..', 'data', 'kaggle')

    parser = argparse.ArgumentParser()
    parser.add_argument('subset_len', nargs='?', default=None, help='len of users subset')
    parser.add_argument('data_path', nargs='?', default=src_folder, help='directory with kaggle data')

    args = parser.parse_args()

    PATH_TO_DATA = args.data_path

    if args.subset_len is not None:
        un = int(args.subset_len)
        make_subset(PATH_TO_DATA, un, random_seed=17)
        sessions_path = os.path.join(PATH_TO_DATA, 'train_sessions_u{}.csv'.format(un))
        fext_path = os.path.join(PATH_TO_DATA, 'train_sessions_fext_u{}.csv'.format(un))
    else:
        sessions_path = os.path.join(PATH_TO_DATA, 'train_sessions.csv')
        fext_path = os.path.join(PATH_TO_DATA, 'train_sessions_fext.csv')


    features = ['session_timespan', '#unique_sites', 'start_hour',
                'day_of_week', 'timespan_median', 'timespan_mean',
                'daily_aсtivity', 'freq_facebook', 'timespan_youtube',
                'timespan_mail', 'freq_googlevideo', 'freq_google']

    time_cols = ['time%d' % i for i in range(1, 11)]
    sessions = pd.read_csv(sessions_path,
                           index_col='session_id',
                           parse_dates=time_cols).rename(columns={'target': 'user_id'})
    train_data = prepare_train_set_fe(sessions,
                                      site_freq_path=os.path.join(PATH_TO_DATA, 'site_freq.pkl'),
                                      feature_names=features)

    columns = ['session_timespan',
               '#unique_sites',
               'start_hour',
               'day_of_week',
               'timespan_mail',
               'freq_googlevideo',
               'target']
    new_features = train_data[columns]

    with open(fext_path, 'wb') as fo:
        pickle.dump(new_features, fo)
