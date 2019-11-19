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


def make_fe_dtype(features, session_length):
    dtype = [('site{}'.format(i), int) for i in range(1, session_length + 1)]
    dtype += [('time_diff{}'.format(i), int) for i in range(1, session_length)]

    if 'session_timespan' in features:
        dtype.append(('session_timespan', int))

    if '#unique_sites' in features:
        dtype.append(('#unique_sites', int))

    if 'start_hour' in features:
        dtype.append(('start_hour', int))

    if 'day_of_week' in features:
        dtype.append(('day_of_week', int))

    if 'timespan_median' in features:
        dtype.append(('timespan_median', float))

    if 'timespan_mean' in features:
        dtype.append(('timespan_mean', float))

    if 'daily_aсtivity' in features:
        dtype.append(('daily_aсtivity', int))

    if 'freq_facebook' in features:
        dtype.append(('freq_facebook', int))

    if 'timespan_youtube' in features:
        dtype.append(('timespan_youtube', int))

    if 'timespan_mail' in features:
        dtype.append(('timespan_mail', int))

    if 'freq_googlevideo' in features:
        dtype.append(('freq_googlevideo', int))

    if 'freq_google' in features:
        dtype.append(('freq_google', int))

    dtype.append(('target', int))

    return dtype


def prepare_train_set_fe(path_to_sessions, site_freq_path, feature_names):
    '''
    Формирование сессий с новыми фичами.
    столбцы:
    [site1, ..., site{session_length},
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
    with open(path_to_sessions, 'rb') as fo:
        sessions = pickle.load(fo)

    with open(site_freq_path, 'rb') as fo:
        site_freq = pickle.load(fo)

    facebook_idx = site_freq['www.facebook.com'][0]
    youtube_idx = site_freq['s.youtube.com'][0]
    mail_idxs = [site_freq[key][0] for key in site_freq.keys() if key.find('mail') > -1]
    gvideo_idxs = [site_freq[key][0] for key in site_freq.keys() if key.find('googlevideo') > -1]
    google_idxs = [site_freq[key][0] for key in site_freq.keys() if key.find('www.google.') > -1]

    sites = sessions[[col for col in sessions.columns if col.find('site') == 0]].fillna(0)
    output = np.zeros((len(sessions),), dtype=make_fe_dtype(feature_names, sites.shape[1]))

    for i in range(sites.shape[1]):
        output['site{}'.format(i + 1)] = sites.iloc[:, i].to_numpy()

    timestamps = sessions[[col for col in sessions.columns if col.find('time') == 0]]
    targets = sessions.user_id

    time_deltas = np.zeros((timestamps.shape[0], timestamps.shape[1] - 1), dtype=int)
    for i in range(1, timestamps.shape[1]):
        time_deltas[:, i - 1] = (timestamps.iloc[:, i] - timestamps.iloc[:, i - 1]).apply(lambda x: x.total_seconds()).fillna(0).to_numpy()
        output['time_diff{}'.format(i)] = time_deltas[:, i - 1]

    start = timestamps.min(axis=1).apply(lambda x: x.hour)

    if 'session_timespan' in feature_names:
        output['session_timespan'] = (timestamps.max(axis=1) - timestamps.min(axis=1)).apply(lambda x: x.total_seconds()).to_numpy()

    if '#unique_sites' in feature_names:
        output['#unique_sites'] = sessions[[col for col in sessions.columns if col.find('site') == 0]].nunique(axis=1)

    if 'start_hour' in feature_names:
        output['start_hour'] = start.to_numpy()

    if 'day_of_week' in feature_names:
        output['day_of_week'] = timestamps.min(axis=1).apply(lambda x: x.dayofweek).to_numpy()

    if 'timespan_median' in feature_names:
        output['timespan_median'] = np.median(time_deltas, axis=1)

    if 'timespan_mean' in feature_names:
        output['timespan_mean'] = np.mean(time_deltas, axis=1)

    if 'daily_aсtivity' in feature_names:
        output['daily_aсtivity'] = np.full((len(sessions), ), 4)
        output['daily_aсtivity'][np.logical_and(start >= 7, start <= 11)] = 0
        output['daily_aсtivity'][np.logical_and(start >= 13, start <= 16)] = 1
        output['daily_aсtivity'][np.logical_and(start >= 18, start <= 21)] = 3

    if 'freq_facebook' in feature_names:
        output['freq_facebook'] = (sites == facebook_idx).sum(axis=1).to_numpy()

    if 'freq_googlevideo' in feature_names:
        for idx in gvideo_idxs:
            output['freq_googlevideo'] += (sites == idx).sum(axis=1).to_numpy()

    if 'freq_google' in feature_names:
        for idx in google_idxs:
            output['freq_google'] += (sites == idx).sum(axis=1).to_numpy()

    if 'timespan_youtube' in feature_names:
        for i in range(time_deltas.shape[1]):
            indices = sites.iloc[:, i] == youtube_idx
            output['timespan_youtube'][indices] += time_deltas[indices, i]

    if 'timespan_mail' in feature_names:
        for idx in mail_idxs:
            for i in range(time_deltas.shape[1]):
                indices = sites.iloc[:, i] == idx
                output['timespan_mail'][indices] += time_deltas[indices, i]

    output['target'] = targets.to_numpy()

    return pd.DataFrame(output)


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
                pickle.dump(train_data, fo, protocol=2)
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

    if 3 in ppstages:
        for num_users in [10, 150]:
            features = ['session_timespan', '#unique_sites', 'start_hour',
                        'day_of_week', 'timespan_median', 'timespan_mean',
                        'daily_aсtivity', 'freq_facebook', 'timespan_youtube',
                        'timespan_mail', 'freq_googlevideo', 'freq_google']
            train_data = prepare_train_set_fe(os.path.join(PATH_TO_DATA,
                                                           'sessions_{}users.pkl'.format(num_users)),
                                              site_freq_path=os.path.join(PATH_TO_DATA,
                                                                          'site_freq_{}users.pkl'.format(num_users)),
                                              feature_names=features)

            columns = ['session_timespan',
                       '#unique_sites',
                       'start_hour',
                       'day_of_week',
                       'target']
            new_features = train_data[columns]

            with open(os.path.join(PATH_TO_DATA, 'new_features_{}users.pkl'.format(num_users)), 'wb') as fo:
                pickle.dump(new_features, fo)

            columns = ['#unique_sites', 'start_hour', 'daily_aсtivity', 'day_of_week',
                       'timespan_youtube', 'timespan_mail', 'freq_googlevideo', 'target']
            new_features = train_data[columns]

            with open(os.path.join(PATH_TO_DATA, 'selected_features_{}users.pkl'.format(num_users)), 'wb') as fo:
                pickle.dump(new_features, fo)
