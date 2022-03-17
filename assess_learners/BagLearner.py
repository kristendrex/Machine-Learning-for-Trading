import numpy as np

class BagLearner(object):

    def __init__(self, learner, kwargs={}, bags=20, boost=False, verbose=False):
        learners = []
        for i in range(bags):
            learners.append(learner(**kwargs))
        self.learners = learners
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.trees = []

    def author(self):
        return ('kdrexinger3')

    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)

    def query(self, points):
        queries = np.array([learner.query(points) for learner in self.learners])
        return (np.mean(queries, axis=0))

if __name__=="__main__":
    print ("Bag Learner")