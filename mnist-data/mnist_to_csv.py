from mnist import MNIST
import pandas as pd
data_path = '.'
mndata = MNIST(data_path)
mndata.gz = True
train_X, train_Y = mndata.load_training()

train_X = pd.DataFrame(train_X)
train_Y = pd.Series(train_Y)

train_X.to_csv('mnist_image.csv', index=False, header=False)
train_Y.to_csv('mnist_label.csv', index=False, header=False)
