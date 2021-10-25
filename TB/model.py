import torch
import torch.nn as nn


def linear_block(in_channels, out_channels, activation='ReLU'):
    if activation == 'ReLU':
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            # nn.BatchNorm1d(out_channels),
            # torch.nn.Dropout(p=0.5),
            nn.ReLU()
        )
    elif activation == 'Sigmoid':
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            # nn.BatchNorm1d(out_channels),
            # torch.nn.Dropout(p=0.5),
            nn.Sigmoid()
        )
    elif activation == 'Tanh':
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            # nn.BatchNorm1d(out_channels),
            # torch.nn.Dropout(p=0.5),
            nn.Tanh()
        )
    elif activation == 'Linear':
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            # torch.nn.Dropout(p=0.5),
        )


class MLP(nn.Module):
    def __init__(self, channel_size_list, activation='ReLU', last_activation='Sigmoid'):
        super(MLP, self).__init__()
        self.channel_size_list = channel_size_list
        layers = []
        for i in range(len(channel_size_list) - 1):
            if i != len(channel_size_list) - 2:
                layers.append(linear_block(
                    channel_size_list[i], channel_size_list[i + 1], activation=activation))
            else:
                # layers.append(linear_block(channel_size_list[i], channel_size_list[i+1], activation='Linear'))
                layers.append(linear_block(
                    channel_size_list[i], channel_size_list[i + 1], activation=last_activation))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        y = self.model(x)
        return y
