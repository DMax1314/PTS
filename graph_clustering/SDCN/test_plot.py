from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=500,
                           n_features=10,
                           n_classes=5,
                           n_informative=4,
                           random_state=0)
print(X.shape)
print(len(y))

plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, cmap='rainbow')
plt.show()

# for i in range(len(y)):
#     plt.scatter(X[i,0], X[i,1], c=y[i], cmap='rainbow')
# plt.show()