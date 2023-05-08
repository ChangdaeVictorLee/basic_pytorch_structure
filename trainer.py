from copy import deepcopy

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

class Trainer():

    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

        '''
        index_select 설명
        x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        indices = torch.tensor([0, 2])
        selected_rows = torch.index_select(x, dim=0, index=indices)

        결과
        tensor([[1, 2, 3],
                [7, 8, 9]])

        selected_cols = torch.index_select(x, dim=1, index=indices)
        결과
        tensor([[1, 3],
                [4, 6],
                [7, 9]])
        '''


    def _train(self, x, y, config):
        # train 모드로 호출하기, nn모듈에서 상속받았으므로 .train()으로 불러올 수 있단
        self.model.train()

        # Shuffle before begin.
        # x의 크기는 (batch_size,28*28)
        indices = torch.randperm(x.size(0), device=x.device)
        x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
        y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)

        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            # squeeze를 해주는 이유는 (batch_size,1)일 경우를 방지해서 (batch_size,)로 만들기 위해
            loss_i = self.crit(y_hat_i, y_i.squeeze())

            # Initialize the gradients of the model.
            # 기존에 있는 기울기 지우고 backward를 통해 기울기 계산하고
            # step을 통해 기울기 갱신
            self.optimizer.zero_grad()
            loss_i.backward()

            self.optimizer.step()

            if config.verbose >= 2:
                print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

            # Don't forget to detach to prevent memory leak.
            total_loss += float(loss_i)

        return total_loss / len(x)

    def _validate(self, x, y, config):
        # Turn evaluation mode on.
        self.model.eval()

        # Turn on the no_grad mode to make more efficintly.
        with torch.no_grad():
            # Shuffle before begin.
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
            y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)

            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                if config.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

                total_loss += float(loss_i)

            return total_loss / len(x)

    # train_data, valid_data는 리스트 형태로 [(batch_size, 784)]로 입력
    # config는 train.py에서 학습 관련 파라미터 저장하는 곳
    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(config.n_epochs):
            # 이 때 train_data[0]은 input, train_data[1]은 정답을 의미
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)

            # You must use deep copy to take a snapshot of current best weights.
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                epoch_index + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))

        # Restore to best model.
        self.model.load_state_dict(best_model)
