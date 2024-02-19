import torch
import torch.nn as nn
import inspect
from classes.Logger import Logger as Logger
import sys


class Network(nn.Module):
    def __init__(self, input_shape, output_shape, n_features,  logger=Logger(), **kwargs):
        super().__init__()
        self.logger = logger
        seed = kwargs["seed"]
        fun_layers = kwargs["fun_layers"]
        self.dropout = kwargs["dropout"]
        self.dropout_list = kwargs["dropout_list"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        self.n_input = input_shape[-1]
        self.n_output = output_shape[0]
        self.n_features = n_features
        #print("instaciating DNN")

        if len(n_features) > 0 and len(n_features) == len(fun_layers):
            self.hidden_layers = []
            idx = 0
            for (n_feature, fun_layer) in zip(n_features, fun_layers):
                layers = [item for item in inspect.getmembers(torch.nn, inspect.isclass) if fun_layer in item]
                if len(layers) > 0:
                    layer = layers[0][1]
                    self.hidden_layers.append(layer())
                else:
                    self.logger.error("function {} is not defined in torch.nn".format(fun_layer))
                    sys.exit(1)
                if self.dropout and self.dropout_list[idx] > 0:
                    self.hidden_layers.append(nn.Dropout(self.dropout_list[idx]))
                if idx < len(n_features)-1 and n_features[idx] != n_features[idx+1]:
                    #print("setting linear", n_features[idx], n_features[idx+1])
                    #self.hidden_layers.append(nn.ReLU())
                    self.hidden_layers.append(nn.Linear(n_features[idx], n_features[idx+1]))
                    


                idx += 1
        else:
            self.logger.error("n_features and fun_layers should be lists with the same length.")
            sys.exit(1)

        self._L1 = nn.Linear(self.n_input, n_features[0])

        nn.init.xavier_uniform_(self._L1.weight, gain=nn.init.calculate_gain('relu'))
        self._L2 = nn.Linear(n_features[-1], self.n_output)

        nn.init.xavier_uniform_(self._L2.weight, gain=nn.init.calculate_gain('relu'))


    def forward(self, state, action=None):
        feature = self._L1(torch.squeeze(state, 1).float())
        for layer in self.hidden_layers:
            feature = layer(feature)
        q = self._L2(torch.squeeze(feature, 1).float())

        if action is None:
            return q
        else:
            action = action.long()
            q_acted = torch.squeeze(q.gather(1, action))

            return q_acted
    # Method to return the state of the seed in Torch
    def get_state(self):

        return torch.get_rng_state(), torch.cuda.get_rng_state_all()

    # Method to set the saved state of the seed in Torch
    def set_state(self, torch_seed_state):

        torch.set_rng_state(torch_seed_state[0])
        torch.cuda.set_rng_state_all(torch_seed_state[1])
