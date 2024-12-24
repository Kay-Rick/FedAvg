import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import config
from torch.utils.data import DataLoader, Subset


# 随机划分iid数据集
def random_split_iid_dataset(dataset, n_splits, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))  # 随机打乱数据索引
    split_size = len(dataset) // n_splits  # 每一份的大小
    splits = [indices[i * split_size:(i + 1) * split_size] for i in range(n_splits)]
    return splits


# 每个user分得的不同标签数据量相同
def label_split_iid_dataset(dataset, num_classes, n_splits, seed=42):
    np.random.seed(seed)
    train_labels = np.array(dataset.targets)
    # 构造类别索引
    class_indices = [[] for _ in range(num_classes)]
    for idx, label in enumerate(train_labels):
        class_indices[label].append(idx)
    # 随机打乱每个类别的索引
    for indices in class_indices:
        np.random.shuffle(indices)
    # 将每个类别的数据均匀分配到 n_splits 个子集
    splits = [[] for _ in range(n_splits)]
    for _, indices in enumerate(class_indices):
        samples_per_subset = len(indices) // n_splits
        for i in range(n_splits):
            start = i * samples_per_subset
            end = (i + 1) * samples_per_subset
            splits[i].extend(indices[start:end])
    return splits


# 为每个user分 shards_per_user 个类的数据
def label_noniid_split(dataset, num_classes, num_users, shards_per_user=3, seed=42):
    np.random.seed(seed)
    train_labels = np.array(dataset.targets)
    # 按类别分组数据索引
    class_indices = [[] for _ in range(num_classes)]
    for idx, label in enumerate(train_labels):
        class_indices[label].append(idx)
    
    for indices in class_indices:
        np.random.shuffle(indices)
    # 将每个类别的数据分成若干 shard (碎片)
    shards = []
    for idx in range(num_classes):
        shard_size = len(class_indices[idx]) // (num_users * shards_per_user // num_classes)
        shards.extend([class_indices[idx][i * shard_size: (i + 1) * shard_size] for i in range(len(class_indices[idx]) // shard_size)])

    np.random.shuffle(shards)  # 打乱所有 shard
    # 分配 shard 给每个客户端
    splits = [[] for _ in range(num_users)]
    for i in range(num_users):
        for _ in range(shards_per_user):
            splits[i].extend(shards.pop())
    return splits


# 为每个数据集分割构建 DataLoader
def create_dataloaders(dataset, splits, batch_size=64):
    loaders = []
    split_data_len = []
    for split_indices in splits:
        subset = Subset(dataset, split_indices)  # 使用 Subset 创建子数据集
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        loaders.append((len(subset), loader))
        split_data_len.append(len(subset))
    print("User data sample:")
    print(split_data_len)
    return loaders


# read all dataset and init
def read_data(dataset, batch_size, iid, num_users):
    if (dataset == "MNIST"):
        mnist_dataloaders = read_mnist_data(num_users, batch_size, iid)
        return mnist_dataloaders
    if (dataset == "Cifar10"):
        cifar10_dataloaders = read_cifar10_data(num_users, batch_size, iid)
        return cifar10_dataloaders


def read_cifar10_data(num_users, batch_size, iid=True):
    num_classes = 10
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_loc = config.cifar10_dir
    trainset = torchvision.datasets.CIFAR10(root=data_loc, train=True, download=True, transform=transform_train)
    # transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if iid == True:
        print("---------IID---------")
        splits = label_split_iid_dataset(trainset, num_classes, num_users)
    else:
        print("--------Non-IID--------")
        num_labels = 3
        splits = label_noniid_split(trainset, num_classes, num_users, num_labels)
    loaders = create_dataloaders(trainset, splits, batch_size=batch_size)
    return loaders


def read_mnist_data(num_users, batch_size, iid=True):
    num_classes = 10
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    data_loc = config.mnist_dir
    trainset = torchvision.datasets.MNIST(
        root=data_loc, train=True, download=True, transform=transform)
    if iid == True:
        print("---------IID---------")
        splits = label_split_iid_dataset(trainset, num_classes, num_users)
    else:
        print("--------Non-IID--------")
        num_labels = 3
        splits = label_noniid_split(trainset, num_classes, num_users, num_labels)
    loaders = create_dataloaders(trainset, splits, batch_size=batch_size)
    return loaders


def generate_server_testloader(dataset, evaluate_batch_size):
    if dataset == "Cifar10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        testset = torchvision.datasets.CIFAR10(root=config.cifar10_dir, train=False, download=True, transform=transform)
    if dataset == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        testset = torchvision.datasets.MNIST(root=config.mnist_dir, train=False, download=True, transform=transform)
    if dataset == "Cifar100":
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        testset = torchvision.datasets.CIFAR100(root=config.cifar100_dir, train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=evaluate_batch_size, shuffle=False)
    return testloader


# 自定义 Dataset 类
# class SplitDataSet(Dataset):
#     def __init__(self, images, labels):
#         self.images = images
#         self.labels = labels

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         return self.images[idx], self.labels[idx]


# def read_mnist_data(num_users, iid=True):
#     transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#     data_loc = config.mnist_dir
#     trainset = torchvision.datasets.MNIST(
#         root=data_loc, train=True, download=True, transform=transform)
#     testset = torchvision.datasets.MNIST(
#         root=data_loc, train=False, download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(
#         trainset, batch_size=len(trainset.data), shuffle=True)
#     testloader = torch.utils.data.DataLoader(
#         testset, batch_size=len(testset.data), shuffle=True)

#     for _, train_data in enumerate(trainloader, 0):
#         trainset.data, trainset.targets = train_data
#     for _, train_data in enumerate(testloader, 0):
#         testset.data, testset.targets = train_data

#     random.seed(1)
#     np.random.seed(1)
#     NUM_USERS = num_users  # should be muitiple of 10

#     if iid == True:
#         NUM_LABELS = 10
#         total_samples = len(trainset.data) + len(testset.data)
#         SamplesPerlabel = total_samples // (NUM_USERS * NUM_LABELS)
#     else:
#         NUM_LABELS = 3
#         total_samples = len(trainset.data) + len(testset.data)
#         SamplesPerlabel = total_samples // (NUM_USERS * NUM_LABELS)

#     mnist_data_image = []
#     mnist_data_label = []

#     mnist_data_image.extend(trainset.data.cpu().detach().numpy())
#     mnist_data_image.extend(testset.data.cpu().detach().numpy())
#     mnist_data_label.extend(trainset.targets.cpu().detach().numpy())
#     mnist_data_label.extend(testset.targets.cpu().detach().numpy())
#     mnist_data_image = np.array(mnist_data_image)
#     mnist_data_label = np.array(mnist_data_label)

#     mnist_data = []
#     for i in trange(10):
#         idx = mnist_data_label == i
#         mnist_data.append(mnist_data_image[idx])

#     print("\nNumb samples of each label:\n", [len(v) for v in mnist_data])
#     users_lables = []

#     ###### CREATE USER DATA SPLIT #######
#     # Assign 100 samples to each user
#     X = [[] for _ in range(NUM_USERS)]
#     y = [[] for _ in range(NUM_USERS)]
#     idx = np.zeros(10, dtype=np.int64)
#     for user in trange(NUM_USERS):
#         for j in range(NUM_LABELS):  # 3 labels for each users
#             # l = (2*user+j)%10
#             l = (user + j) % 10
#             # print("L:", l)
#             X[user] += mnist_data[l][idx[l]:idx[l]+SamplesPerlabel].tolist()
#             y[user] += (l*np.ones(SamplesPerlabel)).tolist()
#             idx[l] += SamplesPerlabel

#     print("IDX1:", idx)  # counting samples for each labels

#     # Create data structure
#     train_data = {'users': [], 'user_data': {}, 'num_samples': []}
#     test_data = {'users': [], 'user_data': {}, 'num_samples': []}

#     # Setup 5 users
#     # for i in trange(5, ncols=120):
#     for i in range(NUM_USERS):
#         uname = i
#         combined = list(zip(X[i], y[i]))
#         random.shuffle(combined)
#         X[i][:], y[i][:] = zip(*combined)

#         num_samples = len(X[i])
#         train_len = int(0.75*num_samples)
#         test_len = num_samples - train_len

#         # X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\

#         test_data['users'].append(uname)
#         test_data["user_data"][uname] = {
#             'x': X[i][:test_len], 'y': y[i][:test_len]}
#         test_data['num_samples'].append(test_len)

#         train_data["user_data"][uname] = {
#             'x': X[i][test_len:], 'y': y[i][test_len:]}
#         train_data['users'].append(uname)
#         train_data['num_samples'].append(train_len)

#     return train_data['users'], _, train_data['user_data'], test_data['user_data']


# def read_cifar10_data(num_users, iid=True):
#     users_train_dataloader = {}
#     transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     data_loc = config.cifar10_dir
#     trainset = torchvision.datasets.CIFAR10(
#         root=data_loc, train=True, download=True, transform=transform)
#     num_classes = 10
#     train_images, train_labels = trainset.data, np.array(trainset.targets)
#     if iid == True:
#         class_indices = [[] for _ in range(num_classes)]
#         for idx, label in enumerate(train_labels):
#             class_indices[label].append(idx)
        
#         # 随机打乱每个类别的索引
#         for indices in class_indices:
#             np.random.shuffle(indices)
        
#         # 将每个类别的数据均匀分配到 num_users 个子集
#         subsets = [[] for _ in range(num_users)]
#         for _, indices in enumerate(class_indices):
#             samples_per_subset = len(indices) // num_users
#             for i in range(num_users):
#                 start = i * samples_per_subset
#                 end = (i + 1) * samples_per_subset if i != num_users - 1 else len(indices)  # 确保最后一组包含剩余数据
#                 subsets[i].extend(indices[start:end])

#         # 形成数据集并构造dataloader
#         for i, subset_indices in enumerate(subsets):
#             subset_images = train_images[subset_indices]
#             subset_labels = train_labels[subset_indices]
#             # 改变维度并标准化处理
#             subset_images = torch.tensor(subset_images).permute(0, 3, 1, 2).float() / 255.0
#             subset_images = 2 * subset_images - 1
#             subset_labels = torch.tensor(subset_labels, dtype=torch.long)
#             # 构造dataloader
#             subDataset = SplitDataSet(subset_images, subset_labels)
#             dataloader = DataLoader(subDataset, batch_size=64, shuffle=True, num_workers=2)
#             users_train_dataloader[i] = (len(subset_indices), dataloader)
#     else:
#         print("Non-IID")
#     return users_train_dataloader


# def read_cifar10_data(num_users, iid=True):
#     transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     data_loc = config.cifar10_dir
#     trainset = torchvision.datasets.CIFAR10(
#         root=data_loc, train=True, download=True, transform=transform)
#     testset = torchvision.datasets.CIFAR10(
#         root=data_loc, train=False, download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(
#         trainset, batch_size=len(trainset.data), shuffle=False)
#     testloader = torch.utils.data.DataLoader(
#         testset, batch_size=len(testset.data), shuffle=False)

#     for _, train_data in enumerate(trainloader, 0):
#         trainset.data, trainset.targets = train_data
#     for _, test_data in enumerate(testloader, 0):
#         testset.data, testset.targets = test_data

#     random.seed(1)
#     np.random.seed(1)
#     # TODO modify user number and split data
#     NUM_USERS = num_users  # should be muitiple of 10

#     if iid == True:
#         NUM_LABELS = 10
#         total_samples = len(trainset.data) + len(testset.data)
#         SamplesPerlabel = total_samples // (NUM_USERS * NUM_LABELS)
#     else:
#         NUM_LABELS = 3
#         total_samples = len(trainset.data) + len(testset.data)
#         SamplesPerlabel = total_samples // (NUM_USERS * NUM_LABELS)

#     cifa_data_image = []
#     cifa_data_label = []

#     cifa_data_image.extend(trainset.data.cpu().detach().numpy())
#     cifa_data_image.extend(testset.data.cpu().detach().numpy())
#     cifa_data_label.extend(trainset.targets.cpu().detach().numpy())
#     cifa_data_label.extend(testset.targets.cpu().detach().numpy())
#     cifa_data_image = np.array(cifa_data_image)
#     cifa_data_label = np.array(cifa_data_label)

#     cifa_data = []
#     for i in trange(10):
#         idx = cifa_data_label == i
#         cifa_data.append(cifa_data_image[idx])

#     print("\nNumb samples of each label:\n", [len(v) for v in cifa_data])
#     users_lables = []

#     ###### CREATE USER DATA SPLIT #######
#     # Assign 100 samples to each user
#     X = [[] for _ in range(NUM_USERS)]
#     y = [[] for _ in range(NUM_USERS)]
#     idx = np.zeros(10, dtype=np.int64)
#     for user in trange(NUM_USERS):
#         for j in range(NUM_LABELS):  # 3 labels for each users
#             # l = (2*user+j)%10
#             l = (user + j) % 10
#             # print("L:", l)
#             X[user] += cifa_data[l][idx[l]:idx[l]+SamplesPerlabel].tolist()
#             y[user] += (l*np.ones(SamplesPerlabel)).tolist()
#             idx[l] += SamplesPerlabel

#     print("IDX1:", idx)  # counting samples for each labels

#     # Create data structure
#     train_data = {'users': [], 'user_data': {}, 'num_samples': []}
#     test_data = {'users': [], 'user_data': {}, 'num_samples': []}

#     # Setup 5 users
#     # for i in trange(5, ncols=120):
#     for i in range(NUM_USERS):
#         uname = i
#         combined = list(zip(X[i], y[i]))
#         random.shuffle(combined)
#         X[i][:], y[i][:] = zip(*combined)

#         num_samples = len(X[i])
#         train_len = int(0.75*num_samples)
#         test_len = num_samples - train_len

#         # X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])\

#         test_data['users'].append(uname)
#         test_data["user_data"][uname] = {
#             'x': X[i][:test_len], 'y': y[i][:test_len]}
#         test_data['num_samples'].append(test_len)

#         train_data["user_data"][uname] = {
#             'x': X[i][test_len:], 'y': y[i][test_len:]}
#         train_data['users'].append(uname)
#         train_data['num_samples'].append(train_len)

#     return train_data['users'], _, train_data['user_data'], test_data['user_data']


# # read each user data
# def read_user_data(index, data, dataset):
#     if (dataset == "Cifar100"):
#         id = data[0][index]
#         train_data = data[1][int(id)]
#         X_train, y_train = train_data['x'], train_data['y']
#     else:
#         id = data[0][index]
#         train_data = data[2][id]
#         test_data = data[3][id]
#         X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
#     if (dataset == "MNIST"):
#         X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
#         X_train = torch.Tensor(
#             X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
#         y_train = torch.Tensor(y_train).type(torch.int64)
#         X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS,
#                                            IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
#         y_test = torch.Tensor(y_test).type(torch.int64)
#     elif (dataset == "Cifar10"):
#         X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
#         X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR,
#                                              IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
#         y_train = torch.Tensor(y_train).type(torch.int64)
#         X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR,
#                                            IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
#         y_test = torch.Tensor(y_test).type(torch.int64)
#     # TODO not finish
#     elif (dataset == "Cifar100"):
#         # elif(dataset == "Cifar10" or dataset=="Cifar100"):
#         X_train, y_train = train_data['x'], train_data['y']
#         X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR,
#                                              IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
#         y_train = torch.Tensor(y_train).type(torch.int64)
#         # X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR,
#         #                                    IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(torch.float32)
#         # y_test = torch.Tensor(y_test).type(torch.int64)
#         # # train_data = [(x, y) for x, y in zip(X_train, y_train)]
#         # # return id, train_data['x'], train_data['y']

#     else:
#         X_train = torch.Tensor(X_train).type(torch.float32)
#         y_train = torch.Tensor(y_train).type(torch.int64)
#         X_test = torch.Tensor(X_test).type(torch.float32)
#         y_test = torch.Tensor(y_test).type(torch.int64)

#     train_data = [(x, y) for x, y in zip(X_train, y_train)]
#     test_data = [(x, y) for x, y in zip(X_test, y_test)]
#     return id, train_data, test_data