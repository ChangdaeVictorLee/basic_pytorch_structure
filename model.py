import torch
import torch.nn as nn

class ImageClassifier(nn.Module):

    # init 함수에서 layer 정의
    def __init__(self,
                 input_size, # 784
                 output_size): # 10
        self.input_size = input_size
        self.output_size = output_size

        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.LeakyReLU(),
            nn.BatchNorm1d(500),

            nn.Linear(500, 400),
            nn.LeakyReLU(),
            nn.BatchNorm1d(400),

            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.BatchNorm1d(300),

            nn.Linear(300, 200),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200),

            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100),

            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50),

            nn.Linear(50, output_size),
            # (batch size, hidden size)로 나오는데 (batch_size, 784=28*28)
            # 이 때, 각 배치별로 softmax가 들어가야 하므로 -1로 해줌
            nn.LogSoftmax(dim=-1),
        )

    # forward에서는 계산 수행
    def forward(self, x):
        # |x| = (batch_size, input_size)

        y = self.layers(x)
        # |y| = (batch_size, output_size)

        return y
