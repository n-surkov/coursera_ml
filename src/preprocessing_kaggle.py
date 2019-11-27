# -*- coding=utf-8 -*-

from preprocessing import *
import os
import numpy as np
import pickle
import random
import shutil

def make_subset(kaggle_data_path, users_num, force=False, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    site_freq_path = os.path.join(kaggle_data_path, 'site_dic_mod.pkl')
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
    parser.add_argument('stages', nargs='?', default='all', help='preprocessing stages')
    parser.add_argument('subset_len', nargs='?', default=10, help='len of users subset')
    parser.add_argument('data_path', nargs='?', default=src_folder, help='directory with kaggle data')

    args = parser.parse_args()

    PATH_TO_DATA = args.data_path

    if args.stages == 'all':
        ppstages = [i for i in range(3)]
    else:
        ppstages = [int(args.stages)]

    subset_length = int(args.subset_len)
    if 1 in ppstages:
        make_subset(PATH_TO_DATA, subset_length, random_seed=17)
