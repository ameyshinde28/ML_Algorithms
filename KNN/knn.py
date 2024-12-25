import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KNN:
    """
    KNN class
    """
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        # Compute the distance between x and all the points in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort the distances and return the indices of the first k nearest neighbour
        k_idx = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbour training samples
        k_neighbour_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbour_labels).most_common(1)
        return most_common[0][0]

if __name__ == "__main__":
    from matplotlib.colors import ListedColormap
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test,y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
        )
    
    k = 3
    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy(y_true=y_test, y_pred=predictions) * 100
    print(f"KNN classification accuracy is {np.around(accuracy, decimals=2)}%")