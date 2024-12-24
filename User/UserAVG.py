import torch
import torch.nn as nn
from User.User import Userbase

# Implementation for FedAvg clients
class UserAVG(Userbase):
    def __init__(self, device, numeric_id, train_data_len, train_loader, model, batch_size, learning_rate,
                 local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data_len, train_loader, model, batch_size, learning_rate,
                         local_epochs)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def train(self):
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for _, (inputs,targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss=self.loss(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    # def set_grads(self, new_grads):
    #     if isinstance(new_grads, nn.Parameter):
    #         for model_grad, new_grad in zip(self.model.parameters(), new_grads):
    #             model_grad.data = new_grad.data
    #     elif isinstance(new_grads, list):
    #         for idx, model_grad in enumerate(self.model.parameters()):
    #             model_grad.data = new_grads[idx]
