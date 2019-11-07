# -*- coding=utf-8 -*-

import pandas as pd
import os
from collections import Counter
from glob import glob
import scipy.sparse
import numpy as np
import pickle


def make_sessions_info(file_path, session_length):
    '''
    Функция создания датафрейма пользовательских сессий
    :param file_path: путь к файлу user????.csv с данными (timestamp, site)
    :param session_length: длина сессии
    :return: pd.DataFrame (site1, ..., siteN, user_id)
    '''
    data = pd.read_csv(file_path)
    username, _ = os.path.splitext(os.path.basename(file_path))
    user_id = int(username[-4:])

    user_freq = Counter(data['site'])

    session_list = []
    data_len = len(data)
    for i in range(0, data_len, session_length):
        last_index = min(data_len, i + session_length)
        empty_data_len = i + session_length - last_index
        df = data.iloc[i:last_index]
        session_list.append(list(df['site']) + [0 for i in range(empty_data_len)] + [user_id])

    return user_freq, session_list


def make_site_freq(counter):
    '''
    Создание словаря частот встречаемых сайтов
    :param counter: collections.Counter посещённых пользователями сайтов
    :return: словарь: {'site_string': [site_id, site_freq]}
    '''
    site_freq_list = counter.most_common()
    site_freq_dict = {}

    for i, pair in enumerate(site_freq_list):
        site_freq_dict[pair[0]] = [i + 1, pair[1]]

    return site_freq_dict


def prepare_train_set(path_to_csv_files, session_length=10):
    '''
    Создание датафрейма сессий
    :param path_to_csv_files: путь к файлам с информацией о пользователях
    :param session_length: длина сессии
    :return: DataFrame, в котором строки соответствуют уникальным (**не знаю к чему тут слово "уникальными",
    просто все подряд сессии**) сессиям из *session_length* сайтов, *session_length* столбцов – индексам этих
    *session_length* сайтов и последний столбец – ID пользователя
    '''
    site_freq = Counter()
    sessions = []
    for path in glob(os.path.join(path_to_csv_files, '*.csv')):
        current_freq, current_session = make_sessions_info(path, session_length)
        site_freq += current_freq
        sessions += current_session

    sessions = pd.DataFrame(sessions)
    sessions.columns = ['site{}'.format(i) for i in range(1, session_length + 1)] + ['user_id']

    site_freq = make_site_freq(site_freq)
    sessions = sessions.applymap(lambda x: site_freq[x][0] if isinstance(x, str) else x)

    return sessions, site_freq


def make_csr_matrix(sessions, site_freq):
    '''
    Создание разряженной матрицы количества посещений сайтов за сессию.
    Элемент  (i, j) соответствует количеству посещений сайта j в сессии i
    :param sessions: сессии пользователей (без user_id)
    :param site_freq: посещаемость всех сайтов во всех сессиях
    :return: scipy.sparce.csr_matrix
    '''
    output = scipy.sparce.csr_matrix((len(sessions), len(site_freq) + 1), dtype=np.int)
    for i in range(len(sessions)):
        c = Counter(sessions[i, :-1])
        output[i, list(c)] += np.array(list(c.values()))

    return output[:, 1:]


def prepare_sparse_train_set_window(path_to_csv_files, site_freq_path, session_length, window_size):
    '''
    Формирование массива сессий с пересечениями = длина сессии - ширина окна.
    :param path_to_csv_files: путь к папке с .csv файлами пользователей
    :param site_freq_path: путь к .pickle файлу со словарём файлов
    :param session_length: длина сессии
    :param window_size: ширина окна непересекающейся части сессии
    :return: scipy.sparse.csr_matrix -- сессии,
             numpy.array -- id пользователя для каждой сессии
    '''
    with open(site_freq_path, 'rb') as fo:
        site_freq = pickle.load(fo)
    row = []
    col = []
    values = []
    user_ids = []
    current_row = 0
    for path in glob(os.path.join(path_to_csv_files, '*.csv')):
        data = pd.read_csv(path)
        username, _ = os.path.splitext(os.path.basename(path))
        user_id = int(username[-4:])

        data_len = len(data)
        for i in range(0, data_len, window_size):
            last_index = min(data_len, i + session_length)
            empty_data_len = i + session_length - last_index
            df = data.iloc[i:last_index].site.apply(lambda x: site_freq[x][0])

            cnt = Counter(list(df) + [0 for i in range(empty_data_len)])
            sites = list(cnt)
            user_ids.append(user_id)
            row += [current_row for i in range(len(sites))]
            col += sites
            values += [cnt[s] for s in sites]

            current_row += 1

    matrix_shape = (current_row, len(site_freq) + 1)

    X = scipy.sparse.csr_matrix((values, (row, col)), shape=matrix_shape)[:, 1:]
    y = np.array(user_ids)

    return X, y


def prepare_train_set_with_fe(path_to_csv_files, site_freq_path, feature_names,
                              session_length=10, window_size=10):
    '''
    Формирование DataFrame'a сессий с столбцами [site1, ..., site{session_length},
    time_diff1, ..., time_diff{session_length - 1}, timespan, unique, start, day_of_week, target]
    здесь:
    siteN -- индекс N-го сайта сессии
    time_diffN -- разница вежду посещением N-го и (N+1)-го сайтов [сек]
    timespan -- продолжительность сессии [сек]
    unique -- число уникальных сайтов в сессии
    start -- час начала сессии
    day_of_week -- день недели
    target -- id пользователя
    :param path_to_csv_files: путь к папке с .csv файлами пользователей
    :param site_freq_path: путь к .pickle файлу со словарём файлов
    :param feature_names: название столбцов формирующегося DataFrame'a
    :param session_length: длина сессии
    :param window_size: ширина окна непересекающейся части сессии
    :return: pandas.DataFrame
    '''
    if len(feature_names) != (2 * session_length - 1 + 5):
        raise ValueError('length of feature_names must be equal to 2 * session_length - 1 + 5')

    with open(site_freq_path, 'rb') as fo:
        site_freq = pickle.load(fo)

    output = []
    for path in glob(os.path.join(path_to_csv_files, '*.csv')):
        data = pd.read_csv(path)
        username, _ = os.path.splitext(os.path.basename(path))
        user_id = int(username[-4:])

        data_len = len(data)
        for i in range(0, data_len, window_size):
            last_index = min(data_len, i + session_length)
            empty_data_len = i + session_length - last_index
            sub_df = data.iloc[i:last_index]
            sites = list(sub_df.site.apply(lambda x: site_freq[x][0])) + [0 for i in range(empty_data_len)]
            timestamps = sub_df.timestamp.astype(np.datetime64)
            times = [(timestamps.iloc[i+1] - timestamps.iloc[i]).total_seconds() for i in range(len(timestamps)-1)] + \
                    [0 for i in range(empty_data_len)]
            timespan = (timestamps.max() - timestamps.min()).total_seconds()
            unique = (np.unique(sites) > 0).sum()
            start = timestamps.min().hour
            day_of_week = timestamps.min().dayofweek
            target = user_id

            output.append(sites + times + [timespan, unique, start, day_of_week, target])

    output = pd.DataFrame(output, dtype=int)
    output.columns = feature_names

    return output


def prepare_train_set_with_fe_ext(path_to_csv_files, site_freq_path, feature_names=None,
                                  session_length=10, window_size=10):
    '''
    Формирование DataFrame'a сессий с столбцами [site1, ..., site{session_length},
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
    if feature_names is None:
        feature_names = ['site' + str(i) for i in range(1, session_length + 1)] + \
                        ['time_diff' + str(j) for j in range(1, session_length)] + \
                        ['session_timespan', '#unique_sites', 'start_hour',
                         'day_of_week', 'timespan_median', 'timespan_mean',
                         'daily_aсtivity', 'freq_facebook', 'timespan_youtube',
                         'timespan_mail', 'freq_googlevideo', 'freq_google',
                         'target']

    with open(site_freq_path, 'rb') as fo:
        site_freq = pickle.load(fo)

    facebook_idx = site_freq['www.facebook.com'][0]
    youtube_idx = site_freq['s.youtube.com'][0]
    mail_idxs = [site_freq[key][0] for key in site_freq.keys() if key.find('mail') > -1]
    gvideo_idxs = [site_freq[key][0] for key in site_freq.keys() if key.find('googlevideo') > -1]
    google_idxs = [site_freq[key][0] for key in site_freq.keys() if key.find('www.google.') > -1]

    output = []
    for path in glob(os.path.join(path_to_csv_files, '*.csv')):
        data = pd.read_csv(path)
        username, _ = os.path.splitext(os.path.basename(path))
        user_id = int(username[-4:])

        data_len = len(data)
        for i in range(0, data_len, window_size):
            last_index = min(data_len, i + session_length)
            empty_data_len = i + session_length - last_index
            sub_df = data.iloc[i:last_index]
            sites = list(sub_df.site.apply(lambda x: site_freq[x][0])) + [0 for i in range(empty_data_len)]
            timestamps = sub_df.timestamp.astype(np.datetime64)
            times = [(timestamps.iloc[i+1] - timestamps.iloc[i]).total_seconds() for i in range(len(timestamps)-1)] + \
                    [0 for i in range(empty_data_len)]
            timespan = (timestamps.max() - timestamps.min()).total_seconds()
            unique = (np.unique(sites) > 0).sum()
            start = timestamps.min().hour
            day_of_week = timestamps.min().dayofweek
            timespan_median = np.median(times)
            timespan_mean = np.mean(times)
            if start >= 7 and start <= 11:
                daily_activity = 0
            elif start >= 13 and start <= 16:
                daily_activity = 1
            elif start >= 18 and start <= 21:
                daily_activity = 3
            else:
                daily_activity = 4

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
            target = user_id

            output.append(sites + times + [timespan, unique, start, day_of_week,
                                           timespan_median, timespan_mean, daily_activity,
                                           freq_facebook, timespan_youtube, timespan_mail,
                                           freq_googlevideo, freq_google,
                                           target])

    output = pd.DataFrame(output, dtype=int)
    output.columns = feature_names

    return output