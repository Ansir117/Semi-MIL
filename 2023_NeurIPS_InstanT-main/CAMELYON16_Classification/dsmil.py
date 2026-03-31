import torch
import torch.nn as nn
import torch.nn.functional as F


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=2):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, out_size)
        )

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0):
        super(BClassifier, self).__init__()
        self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_v),
            nn.Linear(512, output_class)
        )

    def forward(self, feats, c):
        device = feats.device
        V = self.v(feats)  # N x V
        Q = self.q(feats)  # N x Q

        # compute attention scores
        A = torch.mm(Q, Q.transpose(0, 1))  # N x N
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)

        # compute bag representation
        M = torch.mm(A.transpose(0, 1), V)  # N x V

        # predict bag class
        Y_prob = self.classifier(M[0].unsqueeze(0))  # 1 x C

        return Y_prob, A, M


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)

        return classes, prediction_bag, A, B