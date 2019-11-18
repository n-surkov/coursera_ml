# -*- coding=utf-8 -*-

import pandas as pd
import os
from collections import Counter
from glob import glob
import scipy.sparse
import numpy as np
import pickle
import itertools


def read_sites_freq(path_to_csv_files):
    '''
    Чтение частот сайтов из файлов записи активности пользователей
    :param path_to_csv_files: путь к файлам с записями
    :return: словарь: {'site_name': [site_id, site_freq]}
    '''
    sites_counter = Counter()
    for path in glob(os.path.join(path_to_csv_files, '*.csv')):
        data = pd.read_csv(path)
        sites_counter += Counter(data['site'])

    return make_site_freq(sites_counter)


def make_site_freq(counter, sort_by_names=False):
    '''
    Создание словаря частот встречаемых сайтов
    :param counter: collections.Counter посещённых пользователями сайтов
    :return: словарь: {'site_string': [site_id, site_freq]}
    '''
    site_freq_dict = {}

    if sort_by_names:
        i = 0
        for _, group in itertools.groupby(counter.most_common(), key=lambda x: x[1]):
            for site, freq in sorted(group):
                site_freq_dict[site] = [i + 1, freq]
                i += 1
    else:
        site_freq_list = counter.most_common()
        for i, pair in enumerate(site_freq_list):
            site_freq_dict[pair[0]] = [i + 1, pair[1]]

    return site_freq_dict


def make_column_names(session_length):
    columns = ['site{}'.format(i) for i in range(1, session_length + 1)]
    columns += ['time{}'.format(i) for i in range(1, session_length + 1)]
    columns += ['user_id']
    return columns


def make_sessions_by_sites(user_info_path, site_freq_path, session_length, window_size=None):
    '''
    Чтение информации о пользователе с разбитием на сессии по количеству сайтов
    :param user_info_path: путь к .csv файлу с сайтами, посещёнными пользователем
    :param site_freq_path: путь к .pkl файлу с частотами посещаемости сайтов
    :param session_length: длина сессии в сайтах
    :param window_size: разница между началом соседних сессий в сайтах
    :return: pandas.DataFrame ['site1', ..., 'site{session_length}, 'time1', ..., 'time{session_length}', 'user_id']
    '''
    if window_size is None:
        window_size = session_length

    with open(site_freq_path, 'rb') as fo:
        site_freq = pickle.load(fo)

    username, _ = os.path.splitext(os.path.basename(user_info_path))
    user_id = int(username[-4:])

    data = pd.read_csv(user_info_path, parse_dates=['timestamp'])

    data_len = len(data)

    output_columns = make_column_names(session_length)

    output_list = []

    for i in range(0, data_len, window_size):
        last_index = min(data_len, i + session_length)
        empty_data_len = i + session_length - last_index
        sites = data.iloc[i:last_index].site.apply(lambda x: site_freq[x][0])
        timestamps = data.iloc[i:last_index].timestamp

        output_list.append(list(sites) + [np.nan for i in range(empty_data_len)] +\
                           list(timestamps) + [np.nan for i in range(empty_data_len)] +\
                           [user_id])

    output = pd.DataFrame(output_list)
    output.columns = output_columns

    return output


def make_sessions_by_time(user_info_path, site_freq_path, session_timespan, session_length, window_size=None):
    '''
    Чтение информации о пользователе с разбитием на сессии:
    длина сессии выбирается таким образом, чтобы она не превышала значения session_timespan в секундах и
    значения session_length по количеству сайтов.
    :param user_info_path: путь к .csv файлу с сайтами, посещёнными пользователем
    :param site_freq_path: путь к .pkl файлу с частотами посещаемости сайтов
    :param session_length: длина сессии в секундах
    :param session_length: длина сессии в сайтах
    :param window_size: разница между началом соседних сессий в секундах
    :return: pandas.DataFrame ['site1', ..., 'site{session_length}, 'time1', ..., 'time{session_length}', 'user_id']
    '''
    if window_size is None:
        window_size = session_timespan

    with open(site_freq_path, 'rb') as fo:
        site_freq = pickle.load(fo)

    username, _ = os.path.splitext(os.path.basename(user_info_path))
    user_id = int(username[-4:])

    data = pd.read_csv(user_info_path, parse_dates=['timestamp'])

    data_len = len(data)

    output_columns = make_column_names(session_length)

    sessions = []

    start_time = data.iloc[0].timestamp
    current_session_sites = []
    current_session_timestamps = []
    for i in range(data_len):
        current_inwin_site = site_freq[data.iloc[i].site][0]
        current_inwin_time = data.iloc[i].timestamp

        if (current_inwin_time - start_time).total_seconds() > window_size:
            for j in range(i, data_len):
                current_outwin_site = site_freq[data.iloc[j].site][0]
                current_outwin_time = data.iloc[j].timestamp
                if (current_outwin_time - start_time).total_seconds() > session_timespan or \
                        len(current_session_sites) == session_length:
                    empty_data_len = session_length - len(current_session_sites)
                    sessions.append(current_session_sites + [np.nan for i in range(empty_data_len)] +
                                    current_session_timestamps + [np.nan for i in range(empty_data_len)] +
                                    [user_id])

                    current_session_sites = [current_inwin_site]
                    current_session_timestamps = [current_inwin_time]
                    break
                else:
                    current_session_sites += [current_outwin_site]
                    current_session_timestamps += [current_outwin_time]
            else:
                empty_data_len = session_length - len(current_session_sites)
                sessions.append(current_session_sites + [np.nan for i in range(empty_data_len)] +
                                current_session_timestamps + [np.nan for i in range(empty_data_len)] +
                                [user_id])
                current_session_sites = []
                current_session_timestamps = []
                break
            start_time = current_inwin_time
        else:
            if len(current_session_sites) == session_length:
                empty_data_len = session_length - len(current_session_sites)
                sessions.append(current_session_sites + [np.nan for i in range(empty_data_len)] +
                                current_session_timestamps + [np.nan for i in range(empty_data_len)] +
                                [user_id])
                current_session_sites = []
                current_session_timestamps = []
            else:
                current_session_sites.append(current_inwin_site)
                current_session_timestamps.append(current_inwin_time)

    if current_session_sites != []:
        empty_data_len = session_length - len(current_session_sites)
        sessions.append(current_session_sites + [np.nan for i in range(empty_data_len)] +
                        current_session_timestamps + [np.nan for i in range(empty_data_len)] +
                        [user_id])

    output = pd.DataFrame(sessions)
    output.columns = output_columns

    return output


def prepare_train_set(path_to_csv_files, site_freq_path, session_length=10, session_timespan=None, window_size=None):
    '''
    Подготовка сессий по всем пользователям
    :param path_to_csv_files: путь к папке с .csv файлами о пользователях
    :param site_freq_path: путь к .pkl файлу с частотами посещений сайтов
    :param session_length: максимальная длина сессии в сайтах
    :param session_timespan: длина сессии в секундах. default=None
    В случае None используется разбиение по сессиям make_sessions_by_sites()
    Иначе используется разбиение по сессиям make_sessions_by_time()
    :param window_size: разница между началом соседних сессий в сайтах (если session_timespan is None)
    или в секундах (если session_timespan is not None)
    :return: pandas.DataFrame ['site1', ..., 'site{session_length}, 'time1', ..., 'time{session_length}', 'user_id']
    '''
    output = pd.DataFrame(np.empty((0, 2 * session_length + 1)))
    output.columns = make_column_names(session_length)
    if session_timespan is None:
        for path in glob(os.path.join(path_to_csv_files, '*.csv')):
            current_user_data = make_sessions_by_sites(path, site_freq_path, session_length, window_size)
            output = output.append(current_user_data, ignore_index=True)
    else:
        for path in glob(os.path.join(path_to_csv_files, '*.csv')):
            current_user_data = make_sessions_by_time(path, site_freq_path, session_timespan,
                                                      session_length, window_size)
            output = output.append(current_user_data, ignore_index=True)

    return output


def prepare_sparse_train_set(path_to_csv_files, site_freq_path, session_length=10,
                             session_timespan=None, window_size=None):
    '''
    Подготовка сессий по всем пользователям
    :param path_to_csv_files: путь к папке с .csv файлами о пользователях
    :param site_freq_path: путь к .pkl файлу с частотами посещений сайтов
    :param session_length: максимальная длина сессии в сайтах
    :param session_timespan: длина сессии в секундах. default=None
    В случае None используется разбиение по сессиям make_sessions_by_sites()
    Иначе используется разбиение по сессиям make_sessions_by_time()
    :param window_size: разница между началом соседних сессий в сайтах (если session_timespan is None)
    или в секундах (если session_timespan is not None)
    :return: разреженная матрица сессий, id пользователей по сессиям
    '''
    with open(site_freq_path, 'rb') as fo:
        site_freq = pickle.load(fo)

    train_set = prepare_train_set(path_to_csv_files, site_freq_path, session_length,
                                  session_timespan, window_size)

    X = train_set.fillna(0).iloc[:, :(-1 - session_length)].values
    y = train_set.iloc[:, -1].values.astype(int)
    X_sparse = make_csr_matrix(X, site_freq)

    return X_sparse, y


def make_csr_matrix(sessions, site_freq):
    '''
    Создание разряженной матрицы количества посещений сайтов за сессию.
    Элемент  (i, j) соответствует количеству посещений сайта j в сессии i
    :param sessions: сессии пользователей (без user_id)
    :param site_freq: посещаемость всех сайтов во всех сессиях
    :return: scipy.sparce.csr_matrix
    '''
    row = []
    col = []
    data = []
    for i in range(len(sessions)):
        c = Counter(sessions[i])
        row += [i for j in range(len(c))]
        col += list(c)
        data += list(c.values())

    output = scipy.sparse.csr_matrix((data, (row, col)),
                                     shape=(len(sessions), len(site_freq) + 1),
                                     dtype=np.int)

    return output[:, 1:]


def extend_session_features(sites, timestamps, user_id, site_freq, features):
    '''
    Формирование сессии с новыми фичами с столбцами [site1, ..., site{session_length},
    time_diff1, ..., time_diff{session_length - 1}, {...features...}, target]
    здесь:
    siteN -- индекс N-го сайта сессии
    time_diffN -- разница вежду посещением N-го и (N+1)-го сайтов [сек]
    Далее столбцы features следующие:
        session_timespan -- продолжительность сессии [сек]
        #unique_sites -- число уникальных сайтов в сессии
        start_hour -- час начала сессии
        day_of_week -- день недели начала сессии
        timespan_median -- медианное значение продолжительности посещения сайта
        timespan_mean -- среднее значение продолжительности посещения сайта
        daily_aсtivity -- время суток начала сессии (0 -- [7-11], 1 -- [13-16], 2 -- [18-21], 4 -- остальные часы)
        freq_facebook -- количество посещений www.facebook.com в сессии
        timespan_youtube -- суммарное время посещения сайта youtube в сессии
        timespan_mail -- суммарное время посещения сайта mail.google.com в сессии
        freq_googlevideo -- количество посещений сайтов "*googlevideo*"
        freq_google -- количество посещений "www.google.*"
    target -- id пользователя

    :param path_to_csv_files: путь к папке с .csv файлами пользователей
    :param site_freq_path: путь к .pickle файлу со словарём файлов
    :param feature_names: название столбцов формирующегося DataFrame'a
    :param session_length: длина сессии
    :param window_size: ширина окна непересекающейся части сессии
    :return: pandas.DataFrame
    '''
    facebook_idx = site_freq['www.facebook.com'][0]
    youtube_idx = site_freq['s.youtube.com'][0]
    mail_idxs = [site_freq[key][0] for key in site_freq.keys() if key.find('mail') > -1]
    gvideo_idxs = [site_freq[key][0] for key in site_freq.keys() if key.find('googlevideo') > -1]
    google_idxs = [site_freq[key][0] for key in site_freq.keys() if key.find('www.google.') > -1]

    output = list(sites)
    times = timestamps.diff().apply(lambda x: x.total_seconds()).fillna(0)
    output += list(times)
    features_columns = ['site{}'.format(i) for i in range(1, len(sites) + 1)]
    features_columns += ['time{}'.format(i) for i in range(1, len(sites))]
    start = timestamps.min().hour

    if 'session_timespan' in features:
        features_columns.append('session_timespan')
        output.append((timestamps.max() - timestamps.min()).total_seconds())

    if '#unique_sites' in features:
        features_columns.append('#unique_sites')
        output.append((np.unique(sites) > 0).sum())

    if 'start_hour' in features:
        features_columns.append('start_hour')
        output.append(start)

    if 'day_of_week' in features:
        features_columns.append('day_of_week')
        output.append(timestamps.min().dayofweek)

    if 'timespan_median' in features:
        features_columns.append('timespan_median')
        output.append(np.median(times))

    if 'timespan_mean' in features:
        features_columns.append('timespan_mean')
        output.append(np.mean(times))

    if 'daily_aсtivity' in features:
        features_columns.append('daily_aсtivity')
        if start >= 7 and start <= 11:
            output.append(0)
        elif start >= 13 and start <= 16:
            output.append(1)
        elif start >= 18 and start <= 21:
            output.append(3)
        else:
            output.append(4)

    freq_facebook = 0
    timespan_youtube = 0
    timespan_mail = 0
    freq_googlevideo = 0
    freq_google = 0
    for i, site in enumerate(sites):
        if site == facebook_idx:
            freq_facebook += 1
            continue
        if site in gvideo_idxs:
            freq_googlevideo += 1
            continue
        if site in google_idxs:
            freq_google += 1
        if i < session_length - 1:
            if site == youtube_idx:
                timespan_youtube += times[i]
                continue
            if site in mail_idxs:
                timespan_mail += times[i]
                continue

    if 'freq_facebook' in features:
        features_columns.append('freq_facebook')
        output.append(freq_facebook)

    if 'timespan_youtube' in features:
        features_columns.append('timespan_youtube')
        output.append(timespan_youtube)

    if 'timespan_mail' in features:
        features_columns.append('timespan_mail')
        output.append(timespan_mail)

    if 'freq_googlevideo' in features:
        features_columns.append('freq_googlevideo')
        output.append(freq_googlevideo)

    if 'freq_google' in features:
        features_columns.append('freq_google')
        output.append(freq_google)

    features_columns.append('user_id')
    output.append(user_id)

    output = pd.DataFrame(output)
    output.columns = features_columns
    return output


if __name__ == '__main__':
    import argparse

    src_folder = os.path.join('.', '..', 'data', 'capstone_user_identification')

    parser = argparse.ArgumentParser()
    parser.add_argument('stages', nargs='?', default='all', help='preprocessing stages')
    parser.add_argument('data_path', nargs='?', default=src_folder, help='directory with sessions')

    args = parser.parse_args()

    PATH_TO_DATA = args.data_path
    if args.stages == 'all':
        ppstages = [i for i in range(3)]
    else:
        ppstages = [int(args.stages)]

    if 1 in ppstages:
        # Предобработка первой недели.
        # Поиск частот посещений сайтов
        # Составление сессий по 10 сайтов
        # Составление разреженных матриц для сессий по 10 сайтов
        for i in ['3', '10', '150']:
            # Частоты сайтов
            site_freq = read_sites_freq(os.path.join(PATH_TO_DATA, '{}users'.format(i)))
            # сессии по 10 сайтов
            train_data = prepare_train_set(os.path.join(PATH_TO_DATA, '{}users'.format(i)),
                                           os.path.join(PATH_TO_DATA, 'site_freq_{}users.pkl'.format(i)),
                                           session_length=10)

            # Разреженные матрицы для сессий из 10 сайтов
            X = train_data.fillna(0).iloc[:, :(-1 - 10)].values
            y = train_data.iloc[:, -1].values.astype(int)
            X_sparse = make_csr_matrix(X, site_freq)

            with open(os.path.join(PATH_TO_DATA, 'sessions_{}users.pkl'.format(i)), 'wb') as fo:
                pickle.dump(train_data, fo)
            with open(os.path.join(PATH_TO_DATA, 'site_freq_{}users.pkl'.format(i)), 'wb') as fo:
                pickle.dump(site_freq, fo)
            with open(os.path.join(PATH_TO_DATA, 'X_sparse_{}users.pkl'.format(i)), 'wb') as fo:
                pickle.dump(X_sparse, fo, protocol=2)
            with open(os.path.join(PATH_TO_DATA, 'y_{}users.pkl'.format(i)), 'wb') as fo:
                pickle.dump(y, fo, protocol=2)

            print('week 1 sessions for {} users are prepared'.format(i))

    if 2 in ppstages:
        # Предобработка второй недели
        # Составление разреженных матриц сессий с различными параметрами (ширина окна и длина сессии)
        # для 10 и 150 пользователей
        for num_users in [10, 150]:
            for window_size, session_length in itertools.product([10, 7, 5], [15, 10, 7, 5]):
                if window_size <= session_length and (window_size, session_length) != (10, 10):
                    print('pricessing {} users with {} session length and {} window size'.format(num_users,
                                                                                                 session_length,
                                                                                                 window_size))
                    with open(os.path.join(PATH_TO_DATA, 'site_freq_{}users.pkl'.format(num_users)), 'rb') as fo:
                        site_freq = pickle.load(fo)

                    train_set = prepare_train_set(os.path.join(PATH_TO_DATA, '{}users'.format(num_users)),
                                                  os.path.join(PATH_TO_DATA,
                                                               'site_freq_{}users.pkl'.format(num_users)),
                                                  session_length=session_length, window_size=window_size)

                    X = train_set.fillna(0).iloc[:, :(-1 - session_length)].values
                    y = train_set.iloc[:, -1].values.astype(int)
                    X_sparse = make_csr_matrix(X, site_freq)

                    filename = 'sessions_{}users_s{}_w{}.pkl'.format(num_users, session_length, window_size)
                    with open(os.path.join(PATH_TO_DATA, filename), 'wb') as fo:
                        pickle.dump(train_set, fo)
                    filename = 'X_sparse_{}users_s{}_w{}.pkl'.format(num_users, session_length, window_size)
                    with open(os.path.join(PATH_TO_DATA, filename), 'wb') as fo:
                        pickle.dump(X_sparse, fo)
                    filename = 'y_{}users_s{}_w{}.pkl'.format(num_users, session_length, window_size)
                    with open(os.path.join(PATH_TO_DATA, filename), 'wb') as fo:
                        pickle.dump(y, fo)

