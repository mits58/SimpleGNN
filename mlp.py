import chainer
import chainer.functions as F
import chainer.links as L


class MLP(chainer.ChainList):
    """
    A class for Multi Layer Perceptron (MLP) which learns
    Aggregation and Combinine funtion.
    This MLP infer input dimension, so you don't have to set input dimension.
    And this MLP has BatchNormalization layer.

    Attributes
    ----------
    n_layers : int
        number of layers in MLP
    n_hidden : int
        number of hidden units in each layer
    n_out : int
        number of output vector dimension
        i.e. number of classes for classification task
    """
    def __init__(self, n_layers, n_hidden, n_out):
        """
        Parameters
        ----------
        n_layers : int
            number of layers in MLP
        n_hidden : int
            number of hidden units in each layer
        n_out : int
            number of output vector dimensions
            i.e. number of classes for classification task
        """
        super(MLP, self).__init__()

        # define the layers
        for layer in range(n_layers):
            if layer == 0:
                fc = L.Linear(None, n_hidden)
            elif layer != n_layers - 1:
                fc = L.Linear(n_hidden, n_hidden)
            elif layer == n_layers - 1:
                fc = L.Linear(n_hidden, n_out)

            self.add_link(fc)
            fc.name = "fc{}".format(layer)

            # add normalization layer
            if layer != n_layers - 1:
                norm = L.BatchNormalization(n_hidden)
                self.add_link(norm)
                norm.name = "norm{}".format(layer)

    # define forward calculation
    def __call__(self, x):
        for link in self.children():
            pre_activate = link(x)
            if 'fc' in link.name:
                x = pre_activate
            elif 'norm' in link.name:
                x = F.relu(pre_activate)
        return pre_activate
