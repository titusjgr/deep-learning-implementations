import numpy as np
from matplotlib import pyplot as plt

# seed
np.random.seed(0)

# dataset
X = np.random.ranf((250, 2)) * 100
y = np.int32(0.4 * X[:, 0] + 0.6 * X[:, 1] >= 60)

# initialize weights
w = np.zeros(2)
b = 0

# resize X
X = np.array((X[:, 0], X[:, 1]))

# activation func
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# loss func
def cross_entropy(a, y):
    return -(y * np.log(a + 1e-10) + (1 - y) * np.log(1 - a + 1e-10))


# plot decision boundary # plt.ion()
# cannot work SAD
def plot_decision_boundary():
    # w[0]x1 + w[1] x2 + b = 0
    x1 = np.linspace(0, 101, 1000)
    x2 = (-w[0] * x1 - b) / (w[1] + 1e-10)
    plt.clf()
    ind = y == True
    plt.scatter(X[0, ind], X[1, ind])
    ind = y == False
    plt.scatter(X[0, ind], X[1, ind])
    plt.plot(x1, x2)
    plt.draw()


# init
m = X.shape[0]
eta = 0.008
loss = []
incorrect = []

# training
for i in range(5001):
    a = sigmoid(np.dot(w, X) + b)
    w -= np.dot(X, a - y) * eta / m
    b -= np.sum(a - y) / m # because not using feature scaling
    if i % 100 == 0:
        loss.append(np.sum(cross_entropy(a, y)) / m)
        predicted_classes = np.int32((a >= 0.5))
        incorrect.append(np.sum(np.abs(predicted_classes - y)))

print('w =', w)
print('b =', b)


# plotting
fig, axs = plt.subplots(2, 1)
axs[0].plot(incorrect)
axs[0].set_title('Misclassified')
axs[1].plot(loss)
axs[1].set_title('Loss')
x1 = np.linspace(0, 101, 1000)
x2 = (-w[0] * x1 - b) / (w[1] + 1e-10) # w[0]x1 + w[1] x2 + b = 0
ind = y == True # indices for all y == True
fig, ax = plt.subplots(1, 1)
ax.scatter(X[0, ind], X[1, ind], label='Pass')
ind = y == False # indices for all y == False
ax.scatter(X[0, ind], X[1, ind], label='Fail')
ax.legend() # display the label
ax.plot(x1, x2)
plt.show()
