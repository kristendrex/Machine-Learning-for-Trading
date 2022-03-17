import numpy as np

import LinRegLearner, DTLearner, RTLearner, BagLearner

class InsaneLearner(object):

    def __init__(self, bag_learner=BagLearner.BagLearner, learner=LinRegLearner.LinRegLearner,
                 num_bag_learners=20, verbose=False, **kwargs):
        self.verbose = verbose
        bag_learners = []
        for i in range(num_bag_learners):
            bag_learners.append(bag_learner(learner=learner, **kwargs))
        self.bag_learners = bag_learners

    def author(self):
        return "kdrexinger3"

    def add_evidence(self, dataX, dataY):
        for bag_learner in self.bag_learners:
            bag_learner.add_evidence(dataX, dataY)

    def query(self, points):
        preds = np.array([learner.query(points) for learner in self.bag_learners])
        return np.mean(preds, axis=0)

if __name__ == "__main__":
    print("Insane Learner:")