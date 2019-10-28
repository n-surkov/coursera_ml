# -*- coding=utf-8 -*-

import pandas as pd
import os
from collections import Counter
from glob import glob
import scipy.sparse
import numpy as np


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
