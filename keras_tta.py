import numpy as np


class TTA_ModelWrapper():
    """A simple TTA wrapper for keras computer vision models.
    Args:
        model (keras model): A fitted keras model with a predict method.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        """Wraps the predict method of the provided model.
        Augments the testdata with horizontal and vertical flips and
        averages the results.
        Args:
            X (numpy array of dim 4): The data to get predictions for.
        """
        pred = []
        for x_i in X:
            p0 = self.model.predict(self._expand(x_i))
            p1 = self.model.predict(self._expand(np.fliplr(x_i)))
            p2 = self.model.predict(self._expand(np.flipud(x_i)))
            p3 = self.model.predict(self._expand(np.fliplr(np.flipud(x_i))))
        
            p = (p0 + p1 + p2 + p3) / 4
            pred.append(p)
        return np.array(pred)

    def _expand(self, x):
        return np.expand_dims(x, axis=0)
