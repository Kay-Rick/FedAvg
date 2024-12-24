import argparse
from Models.alexnet import AlexNet
from Models.resnet import ResNet18
from Models.models import *
from Server.ServerAvg import FedAvg
import torch

torch.manual_seed(0)

def main(
    dataset,
    algorithm,
    batch_size,
    evaluate_batch_size,
    learning_rate,
    num_glob_iters,
    local_epochs,
    optimizer,
    numusers,
    times,
    gpu,
    iid,
):
    # Get device status: Check GPU or CPU
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # execute times round exp
    for i in range(times):
        print("---------------Running time:---------------", i)
        if dataset == "Cifar10":
            # model = AlexNet().to(device)
            model = ResNet18().to(device)
            print("Model:  ResNet18")
        if dataset == "SVHN":
            model = CifarNet().to(device)
        if dataset == "Cifar100":
            model = AlexNet(num_classes=20).to(device)
            # model = ResNet18().to(device)
        if dataset == "MNIST" or dataset == "FashionMNIST":
            model = LeNet5().to(device)
            # model = ResNet18().to(device)
            print("Model:  LeNet5")
        if dataset == "synthetic":
            model = Mclr_Logistic(60, 10).to(device)
        # select algorithm
        if algorithm == "FedAvg":
            server = FedAvg(
                device,
                dataset,
                algorithm,
                model,
                batch_size,
                evaluate_batch_size,
                learning_rate,
                num_glob_iters,
                local_epochs,
                optimizer,
                numusers,
                i,
                iid,
            )
        server.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cifar10",
        choices=["MNIST", "synthetic", "Cifar10",
                 "Cifar100", "SVHN", "FashionMNIST"],
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--evaluate_batch_size", type=int, default=64)
    parser.add_argument(
        "--learning_rate", type=float, default=0.1, help="Local learning rate"
    )
    parser.add_argument("--num_global_iters", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="FedAvg",
        choices=["pFedMe", "PerAvg", "FedAvg"],
    )
    parser.add_argument(
        "--numusers", type=int, default=50, help="Number of Users per round"
    )
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Which GPU to run the experiments, -1 mean CPU, 0,1,2 for GPU",
    )
    # parser.add_argument("--iid", type=bool, default=True, help="IID or NonIID")
    parser.add_argument("--iid", action="store_true", help="IID or NonIID")
    parser.set_defaults(iid=False)
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Evaluate_batch_size: {}".format(args.evaluate_batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Number of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("GPU           :{}".format(args.gpu))
    print("IID       : {}".format(args.iid))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm=args.algorithm,
        batch_size=args.batch_size,
        evaluate_batch_size=args.evaluate_batch_size,
        learning_rate=args.learning_rate,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer=args.optimizer,
        numusers=args.numusers,
        times=args.times,
        gpu=args.gpu,
        iid=args.iid,
    )
