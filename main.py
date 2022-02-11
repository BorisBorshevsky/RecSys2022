from constants import NUM_USERS, NUM_ITEMS, DATA_FILE_NAME, NUM_TOTAL_RATINGS, TRAIN_RATIO, PATH, get_results_path, SEED
from data_preprocessor import *
from AutoRec import AutoRec
import tensorflow.compat.v1 as tf

from parser import setup_parser

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
    AutoRec = AutoRec(tf_sessions,
                      args,
                      NUM_USERS,
                      NUM_ITEMS,
                      rating,
                      result_path)
    AutoRec.run()
