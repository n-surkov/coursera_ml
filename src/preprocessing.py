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


def prepare_train_set(path_to_csv_files, site_freq_path, session_length, session_timespan=None, window_size=None):
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


if __name__ == '__main__':
    PATH_TO_DATA = 'path'
    for i in ['3', '10', '150']:
        site_freq = read_sites_freq(os.path.join(PATH_TO_DATA, '{}users'.format(i)))
        with open(os.path.join(PATH_TO_DATA, 'site_freq_{}users.pkl'.format(i)), 'wb') as fo:
            pickle.dump(site_freq, fo)
        train_data = prepare_train_set(os.path.join(PATH_TO_DATA, '{}users'.format(i)),
                                       os.path.join(PATH_TO_DATA, 'site_freq_{}users.pkl'.format(i)),
                                       session_length=10)
        train_data.to_csv(os.path.join(PATH_TO_DATA,
                                       'train_data_{}users.csv'.format(i)),
                          index_label='session_id', float_format='%d')

        print('sessions for {} users are prepared'.format(i))
