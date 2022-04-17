#Used Liberaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read Data File
path = 'D:\\Machine Learning\\Regression\\ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
# print('Data= \n',data.head(10))
# print('Data= \n',data.describe())

# Data Rescalling
data = (data-data.mean())/data.std()
# print('Data= \n',data.head(10))
# print('Data= \n',data.describe())

#Add ones column
data.insert(0, 'ones', 1)

# separate X (training data) from y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:,cols-1:cols]
# print('X data = \n' ,X.head(10) )
# print('y data = \n' ,y.head(10) )

# convert to matrices and initialize theta
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0,0]))
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X*theta.T)-y
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
        theta = temp
        cost[i] = computeCost(X, y, theta)
    return theta, cost

alpha = 0.01
iters = 100
g, cost = gradientDescent(X, y, theta, alpha, iters)
print('New Theta= ',g)
print('The Cost= ',computeCost(X, y, g))

# get best fit line for Size vs. Price
x = np.linspace(data.Size.min(), data.Size.max(), 100)
# print('x \n',x)
f = g[0, 0] + (g[0, 1] * x)
# print('f \n',f)
# draw the line for Size vs. Price
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Size, data.Price, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')

# get best fit line for Bedrooms vs. Price
x = np.linspace(data.Bedrooms.min(), data.Bedrooms.max(), 100)
# print('x \n',x)
f = g[0, 0] + (g[0, 1] * x)
# print('f \n',f)
# draw the line for Bedrooms vs. Price
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Bedrooms, data.Price, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')

# draw error graph
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
