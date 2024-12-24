import torch
import os
import copy
import pandas as pd

class Server:
    def __init__(
        self,
        device,
        dataset,
        algorithm,
        model,
        batch_size,
        learning_rate,
        num_glob_iters,
        local_epochs,
        optimizer,
        num_users,
        times,
    ):

        # Set up the main attributes
        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.optimizer = optimizer
        self.num_users = num_users
        self.algorithm = algorithm
        self.rs_glob_loss = []
        self.rs_glob_acc = []
        self.times = times

    def send_parameters(self):
        assert self.users is not None and len(self.users) > 0
        for user in self.users:
            user.set_parameters(self.model)

    # modify
    def save_model(self):
        model_path = os.path.join("TrainedModels", self.dataset)
        a = ""
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        torch.save(
            self.model,
            os.path.join(
                model_path, self.algorithm + a + ".pt"
            ),
        )

    def save_results(self, glob_acc, glob_loss):
        glob_acc_filename = "./result/glob_acc_" + str(self.dataset) + "_" + str(self.num_users) + "_"  + str(self.iid) + ".csv"
        glob_loss_filename = "./result/glob_loss_" + str(self.dataset) + "_" + str(self.num_users) + "_"  + str(self.iid) + ".csv"
        dataframe = pd.DataFrame(glob_acc)
        dataframe.to_csv(glob_acc_filename, index=False, sep=',')
        dataframe = pd.DataFrame(glob_loss)
        dataframe.to_csv(glob_loss_filename, index=False, sep=",")

    # def load_model(self, model_path):
    #     assert os.path.exists(model_path)
    #     print("model loaded!")
    #     self.model = torch.load(model_path)

    # def model_exists(self):
    #     return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

    # def select_users(self, num_users):
    #     if num_users == len(self.users):
    #         print("All users are selected")
    #         return self.users

    #     num_users = min(num_users, len(self.users))
    #     return np.random.choice(self.users, num_users, replace=False)