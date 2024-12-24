import torch
import os
from torch.utils.data import DataLoader
import copy

class Userbase:
    """
    Base class for users in federated learning.
    """
    def __init__(self, device, id, train_data_len, train_loader, model, batch_size = 0, learning_rate = 0, local_epochs = 0):

        self.device = device
        self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_samples = train_data_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.trainloader = train_loader
        self.local_parameters = copy.deepcopy(list(self.model.parameters()))
    
    def set_parameters(self, model):
        self.model.load_state_dict(model.state_dict())

    # def set_parameters(self, model):
    #     for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_parameters):
    #         old_param.data = new_param.data.clone()
    #         local_param.data = new_param.data.clone()

    # def get_parameters(self):
    #     for param in self.model.parameters():
    #         param.detach()
    #     return self.model.parameters()
    
    # def clone_model_paramenter(self, param, clone_param):
    #     for param, clone_param in zip(param, clone_param):
    #         clone_param.data = param.data.clone()
    #     return clone_param
    
    # def get_updated_parameters(self):
    #     return self.local_weight_updated
    
    # def update_parameters(self, new_params):
    #     for param , new_param in zip(self.model.parameters(), new_params):
    #         param.data = new_param.data.clone()

    # def get_grads(self):
    #     grad=[]
    #     for param_1,param_0 in zip(self.model.parameters(),self.local_parameters):
    #         param=param_0.data-param_1.data
    #         #param=param_0.data-param_0.data
    #         grad=param.data.view(-1) if not len(grad) else torch.cat((grad,param.view(-1)))
    #     return grad

    # def test(self):
    #     self.model.eval()
    #     test_acc = 0
    #     for x, y in self.testloaderfull:
    #         x, y = x.to(self.device), y.to(self.device)
    #         output = self.model(x)
    #         test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
    #         #@loss += self.loss(output, y)
    #         #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
    #         #print(self.id + ", Test Loss:", loss)
    #     return test_acc, y.shape[0]

    # def save_model(self):
    #     model_path = os.path.join("models", self.dataset)
    #     if not os.path.exists(model_path):
    #         os.makedirs(model_path)
    #     torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    # def load_model(self):
    #     model_path = os.path.join("models", self.dataset)
    #     self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))


    # def get_params(self):
    #     param=[]
    #     local_dict=self.model.cpu().state_dict()
    #     for key in local_dict:
    #         param = local_dict[key].data.view(-1) if not len(param) else torch.cat((param, local_dict[key].data.view(-1)))
    #     return param
