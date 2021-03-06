import numpy as np
from sklearn.metrics import mean_squared_error


class MF(object):

    def __init__(self, R, K, alpha, beta):
        """
        Perform matrix factorization to predict empty
        entries in a matrix.

        Arguments
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iter_done = 0
        self.last_rmse = None
        self.training_process = []
        self.iter = 0

        # Initialize user and item latent feature matrices
        self.P = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1. / self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

    def train(self, iterations):
        # Perform stochastic gradient descent for number of iterations
        for i in range(iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            self.last_rmse = self.rmse()

            self.iter_done += 1
            self.training_process.append((self.iter_done, self.last_rmse))

        return self.training_process

    def rmse(self):
        """
        A function to compute the total mean square error
        """
        predicted = self.full_matrix()
        return mean_squared_error(self.R, predicted) ** 0.5

    def sgd(self):
        """
        Perform stochastic gradient descent
        """
        for i, j, r in self.samples:
            # Compute prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Compute the full matrix using the resultant biases, P and Q
        """
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.P.dot(self.Q.T)


#
# def mf(R, K, alpha, beta):
#     P, Q = init_latent_feature_matrices(K, num_users, num_items)
#     b_i, b_u = init_biases(num_users, num_items)
#
#     for iter in iters:
#
#         # we want to run only on a sample of the matrix
#         samples = get_samples()
#
#         for sample in samples:
#             prediction = get_prediction(sample)
#             error = get_error(prediction, sample)
#             b_i, b_u = update_biases(alpha, beta, error)
#             P, Q = update_latent_feature_matrices(alpha, beta, error)
#
#
