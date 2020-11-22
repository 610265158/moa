import torch.nn as nn

import torch.nn.functional as F
Dropout_Model=0.25


class MLP(nn.Module):
    def __init__(self, num_features, num_targets=206,extra_targets=402, hidden_size=1500):
        super(MLP, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(Dropout_Model)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(Dropout_Model)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(Dropout_Model)
        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size, extra_targets))
    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        xx = self.batch_norm3(x)
        xx = self.dropout3(xx)
        xx = self.dense3(xx)

        yy = self.batch_norm4(x)
        yy = self.dropout4(yy)
        yy = self.dense4(yy)
        return xx,yy
