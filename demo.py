import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint
import os

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=10000, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=1000, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
parser.add_argument('--dataset_name', type=str, default="MNIST", choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10"], help="dataset")
parser.add_argument('--strategy_name', type=str, default="RandomSampling", 
                    choices=["RandomSampling", 
                             "LeastConfidence", 
                             "MarginSampling", 
                             "EntropySampling", 
                             "LeastConfidenceDropout", 
                             "MarginSamplingDropout", 
                             "EntropySamplingDropout", 
                             "KMeansSampling",
                             "KCenterGreedy", 
                             "BALDDropout", 
                             "AdversarialBIM", 
                             "AdversarialDeepFool"], help="query strategy")
parser.add_argument('--log_file', type=str, default="", help="log file path")
parser.add_argument('--save_file_dir', type=str, default="", help="save file dir")
args = parser.parse_args()
pprint(vars(args))
print()

# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.enabled = False

# device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

dataset = get_dataset(args.dataset_name)                   # load dataset
net = get_net(args.dataset_name, device)                   # load network
strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy

# start experiment
dataset.initialize_labels(args.n_init_labeled)
print(f"number of labeled pool: {args.n_init_labeled}")
print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
print(f"number of testing pool: {dataset.n_test}")
print()

# round 0 accuracy
print("Round 0")
strategy.train()
preds = strategy.predict(dataset.get_test_data())
acc = dataset.cal_test_acc(preds)
print(f"Round 0 testing accuracy: {acc}")

if args.log_file:
    f = open(args.log_file, 'w') 
    f.write(str(acc))
    f.write("\n")
    f.close()


if args.save_file_dir:
    if not os.path.exists(args.save_file_dir):
        os.makedirs(args.save_file_dir)
        
    torch.save({"strategy":strategy, "rd":0, "n_rd":args.n_round+1, "n_init":args.n_init_labeled, "n_query":args.n_query}, os.path.join(args.save_file_dir, "save_0.pth"))
  

for rd in range(1, args.n_round+1):
    print(f"Round {rd}")

    # query
    query_idxs = strategy.query(args.n_query)

    # update labels
    strategy.update(query_idxs)
    strategy.train()

    # calculate accuracy
    preds = strategy.predict(dataset.get_test_data())
    acc = dataset.cal_test_acc(preds)
    print(f"Round {rd} testing accuracy: {acc}")

    if args.log_file:
        f = open(args.log_file, 'a') 
        f.write(str(acc))
        f.write("\n")
        f.close()
        
        
        
        
    if args.save_file_dir:
        if not os.path.exists(args.save_file_dir):
            os.makedirs(args.save_file_dir)

        torch.save({"strategy":strategy, "rd":rd, "n_rd":args.n_round+1, "n_init":args.n_init_labeled, "n_query":args.n_query}, os.path.join(args.save_file_dir, "save_" + str(rd) + ".pth"))
