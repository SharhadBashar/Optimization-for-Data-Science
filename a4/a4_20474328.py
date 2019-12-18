import numpy as np
from scipy.sparse import coo_matrix
from random import randrange as rnd

class MyMethod():
	def __init__(self):
		self.features = 1355191
		self.alpha0 = 0.1
		self.beta0 = -0.19353561519374984

	def toSparse(self, A, cols):
		row = []
		col = []
		data = []
		for i in range(len(A)):
			for key, val in A[i].items():
				row.append(i)
				col.append(key - 1)
				data.append(val)
		return coo_matrix((data, (row, col)), shape = (len(A), cols)).tocsr()

	def stocasticSubGradHinge(self, i, A, b, beta, x):
		A_i = A[i]
		b_i = b[i]
		s = 1 - b_i * (A_i * x + beta)
		if (s > 0):
			g = -A_i.T * b_i
			sigma = -b_i
		else:
			g = np.zeros((x.shape[0], 1))
			sigma = 0
		return g, sigma

	def hingeStocasticSubGrad(self, x0, A, b):
		xCurrent = x0
		alpha0 = self.alpha0
		alpha = alpha0
		beta = self.beta0

		for i in range (200):
			ind = rnd(A.shape[0])
			g, sigma = self.stocasticSubGradHinge(ind, A, b, beta, xCurrent)
				
			xCurrent = xCurrent - alpha * g
			beta = beta - alpha * sigma
			alpha = alpha0 / (i + 1)
		self.xCurrent = xCurrent
		self.beta = beta
		print('Done computing x and beta')

	def bHinge(self, A):
		bModel = np.zeros((A.shape[0]))
		x = self.xCurrent
		beta = self.beta
		s = A * x
		for i in range(A.shape[0]):
			f = s[i] + beta
			if (f < 0):
				bModel[i] = -1
			elif (f > 0):
				bModel[i] = 1
		return bModel

	def fit(self, train_data, train_label):
		A = self.toSparse(train_data, self.features)
		b = np.array(train_label)
		x0 = np.random.uniform(-1, 1, size = (A.shape[1], 1))
		self.hingeStocasticSubGrad(x0, A, b)

	def predict(self, test_data):
		A = self.toSparse(test_data, self.features)
		b = self.bHinge(A)
		return b