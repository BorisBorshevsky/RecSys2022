import numpy as np

from collections import namedtuple

from constants import TRAIN_RATIO

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
    'item_test_set'])


def read_rating(path, num_users, num_items, num_total_ratings) -> Rating:
    one = 1  # a
    zero = 0  # b

    fp = open(path + "ratings.dat")

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
        idx % 1000 == 0 and print("line {}".format(idx))
        user, item, rating, ts = line.split("::")
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

    # ''' Create Ratings matrix '''
    # lines = fp.readlines()
    # for line in lines:
    #     user, item, rating, ts = line.split("::")
    #     user_idx = int(user) - 1
    #     item_idx = int(item) - 1
    #     R[user_idx, item_idx] = int(rating)
    #     mask_R[user_idx, item_idx] = 1
    #     C[user_idx, item_idx] = 1
    #
    # ''' Train '''
    # for itr in train_idx:
    #     line = lines[itr]
    #     user, item, rating, _ = line.split("::")
    #     user_idx = int(user) - 1
    #     item_idx = int(item) - 1
    #     train_R[user_idx, item_idx] = int(rating)
    #     train_mask_R[user_idx, item_idx] = 1
    #
    #     user_train_set.add(user_idx)
    #     item_train_set.add(item_idx)
    #
    # ''' Test '''
    # for itr in test_idx:
    #     line = lines[itr]
    #     user, item, rating, _ = line.split("::")
    #     user_idx = int(user) - 1
    #     item_idx = int(item) - 1
    #     test_R[user_idx, item_idx] = int(rating)
    #     test_mask_R[user_idx, item_idx] = 1
    #
    #     user_test_set.add(user_idx)
    #     item_test_set.add(item_idx)

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
        item_test_set
    )
