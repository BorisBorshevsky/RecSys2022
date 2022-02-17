import tensorflow.compat.v1 as tf

from AutoRec import AutoRec
from constants import get_results_path, SEED, RunParams, pkl_name
from data_preprocessor import *
from parser import setup_parser
from serializer import dump


def tf_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    return config


def build_params(args):
    L = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    K = [10, 20, 40, 80, 100, 200, 300, 400, 500]

    all_params = []
    for l in L:
        for k in K:
            all_params.append(RunParams(k=k, epoch=args.train_epoch, lr=args.base_lr, reg=l))
    return all_params


bestParams = [RunParams(lr=0.001, k=500, reg=1.0, epoch=400)]

# g_options = ['identity', 'selu', 'softmax', 'sigmoid']
g_options = ['identity', 'selu', 'sigmoid']
# g_options = ['selu' ]

# f_options = ['sigmoid', 'selu', 'softmax', 'identity']
f_options = ['sigmoid', 'selu', 'identity']
# f_options = ['sigmoid']


def run_on_params(p: RunParams, args, rating: Rating, f='sigmoid', g='identity', dropout=False):
    try:
        tf.reset_default_graph()
        result_path = get_results_path(args.optimizer_method, args.base_lr)
        with tf.Session(config=tf_config()) as tf_sessions:
            model = AutoRec(tf_sessions,
                            p.reg,
                            p.k,
                            args,
                            rating.num_users,
                            rating.num_items,
                            rating,
                            result_path)

            train_epoch = p.epoch
            step = min(train_epoch, 50)
            model.before_run(f=f, g=g, dropout=dropout)

            pick, data_file_name = pkl_name('Updated-AutoRec', rating.data_set, p,
                                            extra="{}-f{}-g{}".format("-ydropout" if dropout else "-ndropout", f, g))
            for i in range(0, train_epoch, step):
                model.run(step)
                results = model.get_rmse_results()
                print("start: dumping stats to {}", data_file_name)
                dump(results, data_file_name)
                print(results)
                print("end  : dumping stats to {}", data_file_name)

            return model
    except Exception as e:
        print("failed on params: {}, error: {}".format(p, e))


def main(data_set):
    print("TensorFlow version:", tf.__version__)

    # Setup seed
    tf.set_random_seed(SEED)
    np.random.seed(SEED)

    # support TF v1
    tf.disable_v2_behavior()

    parser = setup_parser()
    args = parser.parse_args()

    rating = read_ratings(data_set)

    # all_params = build_params(args)
    all_params = bestParams
    for p in all_params:
        for f in f_options:
            for g in g_options:
                run_on_params(p, args, rating, f=f, g=g, dropout=True)
                run_on_params(p, args, rating, f=f, g=g, dropout=False)


if __name__ == '__main__':
    main(data_set="1m")
