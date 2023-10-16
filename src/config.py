import argparse
from utils import get_rank

parser = argparse.ArgumentParser(description='TiPNN')

parser.add_argument("--gpus", nargs='+', type=int, default=[0],
                    help="gpus")
parser.add_argument("--batch_size", type=int, default=32,
                    help="batch size")
parser.add_argument("--n_worker", type=int, default=0,
                    help="number of workers for dataloader")
parser.add_argument("-d", "--dataset", type=str, default='ICEWS14',
                    help="dataset to use")
parser.add_argument("--test", action='store_true', default=False,
                    help="load stat from dir and directly test")
parser.add_argument("--pretrain_name", type=str, default=None,
                    help="specify the pretrain_name if this is TEST mode")


# configuration for stat training
parser.add_argument("--n_epoch", type=int, default=20,
                    help="number of minimum training epochs on each time step")
parser.add_argument("--lr", type=float, default=0.0005,
                    help="learning rate")
parser.add_argument("--grad_norm", type=float, default=1.0,
                    help="norm to clip gradient to")
parser.add_argument("--negative_num", type=int, default=64,
                    help="number of negative sample")       
parser.add_argument("--adversarial_temperature", type=float, default=0.5,
                    help="adversarial temperature setting")               

# configuration for evaluating
parser.add_argument("--metric", type=list, default=['mrr', 'hits@1', 'hits@3', 'hits@10'],
                    help="evaluating metrics")


# configuration for sequences stat
parser.add_argument("--history_len", type=int, default=10,
                    help="history length")


# configuration for layers
parser.add_argument("--input_dim", type=int, default=64,
                    help="dimension of layer input")
parser.add_argument("--hidden_dims", nargs='+', type=int, default=[64, 64, 64, 64, 64, 64],
                    help="dimension list of hidden layers")
                      # note that you can specify this item using like this
                      # --hidden_dims 16 16 16 16 16 16
parser.add_argument("--message_func", type=str, default='distmult',
                    help="which message_func you use")
parser.add_argument("--aggregate_func", type=str, default='pna',
                    help="which aggregate_func you use")
parser.add_argument("--time_encoding", action='store_true', default=True,
                    help="whether use time encoding")
parser.add_argument("--time_encoding_independent", action='store_true', default=False,
                    help="whether use relation specific time encoder")

parser.add_argument("--short_cut", action='store_true', default=True,
                    help="whether residual connection")
parser.add_argument("--layer_norm", action='store_true', default=True,
                    help="whether layer_norm")                    


args, unparsed = parser.parse_known_args()
if get_rank() == 0:
  print(args)  
  print(unparsed)  
