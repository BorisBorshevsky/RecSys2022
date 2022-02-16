import os

import math
import numpy as np
import tensorflow.compat.v1 as tf
import time

tf.disable_v2_behavior()


class AutoRec:
    def __init__(self,
                 sess,
                 lamda: float,
                 k: int,
                 args,
                 num_users,
                 num_items,
                 rating,
                 result_path):

        self.sess = sess
        self.args = args

        self.num_users = num_users
        self.num_items = num_items

        self.R = rating.R  # Ratings user X item
        self.mask_R = rating.mask_R  # mask for items with data on R
        self.C = rating.C  # mask for items with data on R (with factor One)
        self.train_R = rating.train_R  # Ratings user X item (only train)
        self.train_mask_R = rating.train_mask_R  # mask for Ratings user X item (only train)
        self.test_R = rating.test_R  # Ratings user X item (only test)
        self.test_mask_R = rating.test_mask_R  # mask for Ratings user X item (only test)
        self.num_train_ratings = rating.num_train_ratings  # amount of ratings in train set
        self.num_test_ratings = rating.num_test_ratings  # amount of ratings in test set

        self.user_train_set = rating.user_train_set
        self.item_train_set = rating.item_train_set
        self.user_test_set = rating.user_test_set
        self.item_test_set = rating.item_test_set

        # self.hidden_neuron = args.hidden_neuron  # 500
        self.hidden_neuron = k  # 500
        self.train_epoch = args.train_epoch  # 2000
        self.batch_size = args.batch_size  # 100
        self.num_batch = int(math.ceil(self.num_users / float(self.batch_size)))  # how many batches

        self.base_lr = args.base_lr  # leaning rate
        self.optimizer_method = args.optimizer_method
        self.display_step = args.display_step

        self.global_step = tf.Variable(0, trainable=False)
        self.decay_epoch_step = args.decay_epoch_step
        self.decay_step = self.decay_epoch_step * self.num_batch
        self.lr = tf.train.exponential_decay(self.base_lr, self.global_step,
                                             self.decay_step, 0.96, staircase=True)

        # self.lambda_value = args.lambda_value
        self.lambda_value = lamda

        self.train_cost_list = []
        self.test_cost_list = []
        self.test_rmse_list = []

        self.result_path = result_path
        self.grad_clip = args.grad_clip

        self.iter_done = 0

        # to be filled in prepare step
        self.input_R = None
        self.input_mask_R = None

        self.Encoder = None
        self.Dropout = None
        self.Decoder = None

        self.cost = None
        self.optimizer = None

    def before_run(self, f='sigmoid', g='identity'):
        self.prepare_model(f, g)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def run(self, iters):
        for epoch_itr in range(iters):
            print("running step num: {}".format(self.iter_done))
            self.train_model(self.iter_done)
            self.test_model(self.iter_done)
            self.iter_done += 1

        self.make_records()

    def prepare_model(self, f, g):
        self.input_R = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_R")
        self.input_mask_R = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_mask_R")

        V = tf.get_variable(name="V", initializer=tf.truncated_normal(
            shape=[self.num_items, self.hidden_neuron],  # [num_items X 500] (K in paper)
            mean=0,
            stddev=0.03
        ), dtype=tf.float32)

        W = tf.get_variable(
            name="W",
            initializer=tf.truncated_normal(
                shape=[self.hidden_neuron, self.num_items],  # [500 X num_items] (K in paper)
                mean=0,
                stddev=0.03
            ), dtype=tf.float32)

        mu = tf.get_variable(
            name="mu",
            initializer=tf.zeros(shape=self.hidden_neuron),
            dtype=tf.float32)

        b = tf.get_variable(
            name="b",
            initializer=tf.zeros(shape=self.num_items),
            dtype=tf.float32)

        pre_Encoder = tf.matmul(self.input_R, V) + mu

        if f == 'sigmoid':
            self.Encoder = tf.nn.sigmoid(pre_Encoder)
        elif f == 'selu':
            self.Encoder = tf.nn.selu(pre_Encoder)
        elif f == 'softmax':
            self.Encoder = tf.nn.softmax(pre_Encoder)
        else:
            raise

        self.Dropout = tf.nn.dropout(self.Encoder, rate=2 / 3, seed=1)

        pre_Decoder = tf.matmul(self.Dropout, W) + b

        if g == 'identity':
            self.Decoder = tf.identity(pre_Decoder)  # g(.)
        elif g == 'softmax':
            self.Decoder = tf.nn.softmax(pre_Decoder)  # g(.)
        elif g == 'selu':
            self.Decoder = tf.nn.selu(pre_Decoder)  # g(.)
        else:
            raise

        pre_rec_cost = tf.multiply((self.input_R - self.Decoder), self.input_mask_R)
        rec_cost = tf.square(self.l2_norm(pre_rec_cost))

        pre_reg_cost = tf.square(self.l2_norm(W)) + tf.square(self.l2_norm(V))
        reg_cost = self.lambda_value * 0.5 * pre_reg_cost

        self.cost = rec_cost + reg_cost

        if self.optimizer_method == "Adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer_method == "RMSProp":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif self.optimizer_method == "Adagrad":
            optimizer = tf.train.AdagradOptimizer(self.lr)
        else:
            raise ValueError("Optimizer Key ERROR")

        if self.grad_clip:
            gvs = optimizer.compute_gradients(self.cost)
            capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
            self.optimizer = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
        else:
            self.optimizer = optimizer.minimize(self.cost, global_step=self.global_step)

    def train_model(self, itr):
        start_time = time.time()
        random_perm_doc_idx = np.random.permutation(self.num_users)

        batch_cost = 0
        for i in range(self.num_batch):
            if i == self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size:]
            elif i < self.num_batch - 1:
                batch_set_idx = random_perm_doc_idx[i * self.batch_size: (i + 1) * self.batch_size]

            _, Cost = self.sess.run(
                [self.optimizer, self.cost, ],
                feed_dict={self.input_R: self.train_R[batch_set_idx, :],
                           self.input_mask_R: self.train_mask_R[batch_set_idx, :]})

            batch_cost = batch_cost + Cost
        self.train_cost_list.append(batch_cost)

        if (itr + 1) % self.display_step == 0:
            print("Training //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(batch_cost),
                  "Elapsed time : %d sec" % (time.time() - start_time))

    def test_model(self, itr):
        start_time = time.time()
        Cost, Decoder = self.sess.run(
            [self.cost, self.Decoder],
            feed_dict={self.input_R: self.test_R,
                       self.input_mask_R: self.test_mask_R})

        self.test_cost_list.append(Cost)

        if (itr + 1) % self.display_step == 0:
            Estimated_R = Decoder.clip(min=1, max=5)
            unseen_user_test_list = list(self.user_test_set - self.user_train_set)
            unseen_item_test_list = list(self.item_test_set - self.item_train_set)

            for user in unseen_user_test_list:
                for item in unseen_item_test_list:
                    if self.test_mask_R[user, item] == 1:  # exist in test set
                        Estimated_R[user, item] = 3

            pre_numerator = np.multiply((Estimated_R - self.test_R), self.test_mask_R)
            numerator = np.sum(np.square(pre_numerator))
            denominator = self.num_test_ratings
            RMSE = np.sqrt(numerator / float(denominator))

            self.test_rmse_list.append(RMSE)

            print("Testing //", "Epoch %d //" % (itr), " Total cost = {:.2f}".format(Cost),
                  " RMSE = {:.5f}".format(RMSE),
                  "Elapsed time : %d sec" % (time.time() - start_time))
            print("=" * 100)

    def make_records(self):
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        basic_info = self.result_path + "basic_info.txt"
        train_record = self.result_path + "train_record.txt"
        test_record = self.result_path + "test_record.txt"

        with open(train_record, 'w') as f:
            f.write("{}\t\n".format("cost:"))
            for cost in self.train_cost_list:
                f.write("{}\n".format(cost))

        with open(test_record, 'w') as g:
            g.write("{}\t{}\t{}\n".format('idx:', 'cost:', 'rmse:'))
            for idx, (cost, rmse) in enumerate(zip(self.test_cost_list, self.test_rmse_list)):
                g.write("{}\t{}\t{}\n".format(idx, cost, rmse))

        with open(basic_info, 'w') as h:
            h.write(str(self.args))

    def l2_norm(self, tensor):
        return tf.sqrt(tf.reduce_sum(tf.square(tensor)))

    def get_rmse_results(self):
        return list(enumerate(self.test_rmse_list))
