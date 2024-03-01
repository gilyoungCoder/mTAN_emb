from physionet import *
import torch
import utils
import argparse
from setmodels import *

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--gen-hidden', type=int, default=50)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_rnn')
parser.add_argument('--dec', type=str, default='mtan_rnn')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--quantization', type=float, default=0.1, 
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true', 
                    help="Include binary classification loss")
parser.add_argument('--freq', type=float, default=10.)
parser.add_argument('--k-iwae', type=int, default=10)
parser.add_argument('--norm', action='store_true')
parser.add_argument('--kl', action='store_true')
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--dataset', type=str, default='physionet')
parser.add_argument('--alpha', type=int, default=100.)
parser.add_argument('--old-split', type=int, default=1)
parser.add_argument('--nonormalize', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=1)
parser.add_argument('--dec-num-heads', type=int, default=1)
parser.add_argument('--num-ref-points', type=int, default=128)
parser.add_argument('--classify-pertp', action='store_true')
args = parser.parse_args()



if __name__ == '__main__':
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    aug = SetTransformer(dim_input=83, num_outputs=406, dim_output=83).to(device)    
    # if args.dataset == 'physionet':
    data_obj = utils.get_physionet_data(args, 'cpu', args.quantization)
    # elif args.dataset == 'mimiciii':
    #     data_obj = utils.get_mimiciii_data(args)
    
    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]
    # DataLoader의 반복자 생성
    train_iterator = iter(train_loader)

    # 첫 번째 배치 가져오기
    train_batch, label = next(train_iterator)

    train_batch, label = train_batch.to(device), label.to(device)
    batch_len  = train_batch.shape[0]
    vals, mask, tt \
            = train_batch[:, :, :dim], train_batch[:, :, dim:2*dim], train_batch[:, :, -1]

    print(f"vals : {vals.shape, vals[0]}")
    print(f"mask : {mask.shape, mask[0]}")
    print(f"tt : {tt.shape, tt[0]}")


    print(f"vals : {vals.shape, vals[0][1][-1]}")
    print(f"mask : {mask.shape, mask[0][1][-1]}")
    
    train_aug = aug(train_batch)
    print(f"train_aug : {train_aug.shape}")

    train_aug[:, :, dim:2*dim] = torch.round(train_aug[:, :, dim:2*dim])

    vals, mask, tt \
            = train_aug[:, :, :dim], train_aug[:, :, dim:2*dim], train_aug[:, :, -1]
            
    print(f"vals : {vals.shape, vals[0]}")
    print(f"mask : {mask.shape, mask[0]}")
    print(f"tt : {tt.shape, tt[0]}")


    print(f"vals : {vals.shape, vals[0][1][-1]}")
    print(f"mask : {mask.shape, mask[0][1][-1]}")
