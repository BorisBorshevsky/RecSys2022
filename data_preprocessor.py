from collections import namedtuple

import numpy as np

from constants import TRAIN_RATIO, PATH_1M, NUM_USERS_1M, NUM_ITEMS_1M, NUM_TOTAL_RATINGS_1M, PATH_100K, NUM_USERS_100K, \
    NUM_ITEMS_100K, NUM_TOTAL_RATINGS_100K

Rating = namedtuple('Rating', [
    'R',
    'mask_R',
    'C',
    'train_R',
    'train_mask_R',
    'test_R',
    'test_mask_R',
    'num_train_ratings',
    'num_test_ratings',
    'user_train_set',
    'item_train_set',
    'user_test_set',
    'item_test_set',
    'num_users',
    'num_items',
    'data_set',
])


def _read_rating(path, filename, delim, num_users, num_items, num_total_ratings, data_set_name) -> Rating:
    one = 1  # a
    zero = 0  # b

    fp = open(path + filename)

    user_train_set = set()
    user_test_set = set()
    item_train_set = set()
    item_test_set = set()

    R = np.zeros((num_users, num_items))
    mask_R = np.zeros((num_users, num_items))
    # C = np.ones((num_users, num_items)) * zero
    C = np.zeros((num_users, num_items))

    train_R = np.zeros((num_users, num_items))
    test_R = np.zeros((num_users, num_items))

    train_mask_R = np.zeros((num_users, num_items))
    test_mask_R = np.zeros((num_users, num_items))

    ratings_shuffled = np.random.permutation(num_total_ratings)
    train_idx = ratings_shuffled[0:int(num_total_ratings * TRAIN_RATIO)]  # first 90% are train
    test_idx = ratings_shuffled[int(num_total_ratings * TRAIN_RATIO):]  # last 10% are test

    train_idx_set = set(train_idx)
    test_idx_set = set(test_idx)
    num_train_ratings = len(train_idx)
    num_test_ratings = len(test_idx)

    lines = fp.readlines()
    for idx, line in enumerate(lines):
        idx % 50000 == 0 and print("pre processing line num: {}".format(idx))
        user, item, rating, ts = line.split(delim)
        user_idx = int(user) - 1
        item_idx = int(item) - 1
        R[user_idx, item_idx] = int(rating)
        mask_R[user_idx, item_idx] = 1
        C[user_idx, item_idx] = 1

        if idx in train_idx_set:
            train_R[user_idx, item_idx] = int(rating)
            train_mask_R[user_idx, item_idx] = 1

            user_train_set.add(user_idx)
            item_train_set.add(item_idx)

        if idx in test_idx_set:
            test_R[user_idx, item_idx] = int(rating)
            test_mask_R[user_idx, item_idx] = 1

            user_test_set.add(user_idx)
            item_test_set.add(item_idx)

    return Rating(
        R,
        mask_R,
        C,
        train_R,
        train_mask_R,
        test_R,
        test_mask_R,
        num_train_ratings,
        num_test_ratings,
        user_train_set,
        item_train_set,
        user_test_set,
        item_test_set,
        num_users,
        num_items,
        data_set_name
    )


def _ratings_1m():
    return _read_rating(PATH_1M, "ratings.dat", "::", NUM_USERS_1M, NUM_ITEMS_1M, NUM_TOTAL_RATINGS_1M, '1m')


def _ratings_100k():
    return _read_rating(PATH_100K, 'u.data', "\t", NUM_USERS_100K, NUM_ITEMS_100K, NUM_TOTAL_RATINGS_100K, '100k')


def read_ratings(dataset='1m'):
    if dataset == '1m':
        return _ratings_1m()
    if dataset == '100k':
        return _ratings_100k()
