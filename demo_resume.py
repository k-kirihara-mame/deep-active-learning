import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint
import os

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--log_file', type=str, default="", help="log file path")
parser.add_argument('--save_file_dir', type=str, default="", help="save file dir")
parser.add_argument('--resume_file', type=str, default="", help="resume file path")
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

if os.path.exists(args.resume_file):
    resume_dict = torch.load(args.resume_file)
    strategy = resume_dict["strategy"]
    dataset = strategy.dataset
    n_init = resume_dict["n_init"]
    n_rd = resume_dict["n_rd"]    
    rd = resume_dict["rd"]
    n_query =  resume_dict["n_query"]
else:
    print("resume file does not exist")
    exit()

print(f"number of labeled pool: {n_init}")
print(f"number of query: {n_query}")


      

for rd in range(rd+1, n_rd):
    print(f"Round {rd}")

    # query
    query_idxs = strategy.query(n_query)

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

        torch.save({"strategy":strategy, "rd":rd, "n_rd":n_rd, "n_init":resume_dict["n_init"], "n_query":n_query}, os.path.join(args.save_file_dir, "save_" + str(rd) + ".pth"))
