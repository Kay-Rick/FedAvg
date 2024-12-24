import torch
import torch.nn as nn
from User.UserAVG import UserAVG
from Server.Server import Server
from utils.model_utils import read_data, generate_server_testloader

# Implementation for FedAvg Server
class FedAvg(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, evaluate_batch_size, learning_rate, num_glob_iters,
                 local_epochs, optimizer, num_users, times, iid):
        super().__init__(device, dataset, algorithm, model, batch_size, learning_rate, num_glob_iters,
                         local_epochs, optimizer, num_users, times)
        # Initialize data for all users
        users_train_dataloaders = read_data(dataset, batch_size, iid, num_users)
        total_users = num_users
        self.users = []
        self.iid = iid
        self.criterion = nn.CrossEntropyLoss()
        # testloader test each round global model
        self.testloader = generate_server_testloader(dataset, evaluate_batch_size)
        self.training_time = []
        self.aggregate_time = []

        for i in range(0, total_users):
            user_trainset_len = users_train_dataloaders[i][0]
            user_train_loader = users_train_dataloaders[i][1]
            user = UserAVG(device, i, user_trainset_len, user_train_loader, model, batch_size, learning_rate, local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:", num_users, " / ", total_users)
        print("Finished creating FedAvg server.")

    def Evaluate(self, glob_iter):
        self.model.eval()
        device = self.device
        net = self.model
        criterion = self.criterion
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            global_acc = 100.*correct / total
            global_loss = test_loss / 100
            self.rs_glob_acc.append([glob_iter, global_acc])
            self.rs_glob_loss.append([glob_iter, global_loss])
            print("Average Global Accurancy: ", global_acc)
            print("Average Global Trainning Loss: ", global_loss)

    def aggregate_grads_AVG(self):
        total_data_nums = 0
        for user in self.users:
            total_data_nums += user.train_samples
        # 初始化为0
        global_state = self.model.state_dict()
        for key in global_state.keys():
            global_state[key] = torch.zeros_like(global_state[key], device=self.device).float()

        # 对user加权求和
        for user in self.users:
            user_state = user.model.state_dict()
            weight = user.train_samples / total_data_nums
            for key in global_state.keys():
                global_state[key] += weight * user_state[key]
        
        self.model.load_state_dict(global_state)

    def train(self):
        torch.cuda.empty_cache()
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            # send model parameters to each client
            self.send_parameters()
            for i in range(len(self.users)):
                self.users[i].train()  # user.train_sample
            self.aggregate_grads_AVG()
            self.Evaluate(glob_iter)
        print("Highest accuracy")
        print(max(self.rs_glob_acc))
        self.save_results(self.rs_glob_acc, self.rs_glob_loss)
        self.save_model()


    # TODO 注意不收敛
    # def aggregate_parameters(self):
    #     total_data_nums = 0
    #     for user in self.users:
    #         total_data_nums += user.train_samples
    #     # 初始化为0
    #     global_parameters = list(self.model.parameters())
    #     for param in global_parameters:
    #         param.data.zero_()

    #     for user in self.users:
    #         user_parameters = list(user.model.parameters())
    #         weight = user.train_samples / total_data_nums
    #         for global_param, user_param in zip(global_parameters, user_parameters):
    #             global_param.data += weight * user_param.data
