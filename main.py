import tensorflow.compat.v1 as tf

from AutoRec import AutoRec
from constants import NUM_USERS, NUM_ITEMS, NUM_TOTAL_RATINGS, PATH, get_results_path, SEED, RunParams, pkl_name
from data_preprocessor import *
from parser import setup_parser
from serializer import dump

print("TensorFlow version:", tf.__version__)

# Setup seed
tf.set_random_seed(SEED)
np.random.seed(SEED)

# support TF v1
tf.disable_v2_behavior()

parser = setup_parser()
args = parser.parse_args()

result_path = get_results_path(args.optimizer_method, args.base_lr)

rating = read_rating(PATH, NUM_USERS, NUM_ITEMS, NUM_TOTAL_RATINGS)


def tf_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    return config


with tf.Session(config=tf_config()) as tf_sessions:
    params = RunParams(k=args.hidden_neuron, epoch=args.train_epoch, lr=args.base_lr, reg=1)

    model = AutoRec(tf_sessions,
                      args,
                      NUM_USERS,
                      NUM_ITEMS,
                      rating,
                      result_path)

    train_epoch = args.train_epoch
    step = 5
    model.before_run()

    pick, data_file_name = pkl_name('Adam-AutoRec', params)
    for i in range(0, train_epoch, step):
        model.run(step)
        results = model.get_rmse_results()
        print("start: dumping stats to {}", data_file_name)
        dump(results, data_file_name)
        print("end  : dumping stats to {}", data_file_name)






