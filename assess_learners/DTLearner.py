import numpy as np

class DTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None
        self.root = None

    def author(self):
        return 'kdrexinger3'  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        default = np.array([[-1, data_y.mean(), np.nan, np.nan]])

        if len(np.unique(data_y)) == 1:
            return default
        if data_x.shape[0] <= self.leaf_size:
            return default

        split_feat, split_val = self.get_split(data_x, data_y)

        is_left = data_x[:, split_feat] <= split_val

        if len(np.unique(is_left)) == 1:
            return np.array([[-1, data_y[is_left == np.unique(is_left)[0]].mean(), np.nan, np.nan]])

        lTree = self.build_tree(data_x[is_left], data_y[is_left])
        rTree = self.build_tree(data_x[is_left != True], data_y[is_left != True])

        root = np.array([[split_feat, split_val, 1, len(lTree) + 1]])
        tree = np.concatenate((root, lTree, rTree), axis=0)
        return tree

    def get_split(self, X, Y):
        cList = []
        for y in range(X.shape[1]):
            cList.append(np.abs(np.corrcoef(X[:, y], Y)[0][1]))
        split_feat = cList.index(max(cList))
        split_val = np.median(X[:, split_feat])
        return split_feat, split_val

    def query(self, points):
        return np.array([self.predict(point, 0) for point in points])

    def predict(self, point, row):
        if int(self.tree[int(row), 0]) == -1:
            return self.tree[int(row), 1]
        if point[int(self.tree[int(row), 0])] <= self.tree[int(row), 1]:
            return self.predict(point, row + self.tree[int(row), 2])
        else:
            return self.predict(point, row + self.tree[int(row), 3])


if __name__ == "__main__":
    print("DT Learner")