import argparse 

import torch 
import torch.nn as nn
import torch.optim as optim

from sleep_classification.trainer import Trainer
from sleep_classification.data_loader import load_dataloader_for_featureNet

from sleep_classification.models import DeepSleepNet

def define_argparser():
    p = argparse.ArgumentParser()   
    
    p.add_argument('--model_fn', required=True)
    p.add_argument('--log_dir', default="/tensorboard_logs")
    p.add_argument('--gpu_id', type= int,default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float,default=0.9)
    p.add_argument('--data_dir', type=str,default='G:/내 드라이브/EEG_classification/output')
    p.add_argument('--n_fold',type=int,default=20)
    p.add_argument('--fold_idx', type=int,required=True)
    
    p.add_argument('--batch_size',type=int,default=512)
    p.add_argument('--n_epochs',type=int,default=200)
    p.add_argument('--verbose',type=int,default=2)

    p.add_argument('--use_dropout',type=bool, default=True)
    p.add_argument('--use_rnn',type=bool, default=True)
    
    p.add_argument('--max_grad', type=float, default=-1)

    config = p.parse_args()

    return config

def main(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader = load_dataloader_for_featureNet(config)

    print("Train:", len(train_loader.dataset))
    print("Valid:", len(valid_loader.dataset))
    print("Test:", len(test_loader.dataset))

    model = DeepSleepNet(input_dim=1,n_classes=5,is_train=True,use_dropout=config.use_dropout,use_rnn=config.use_rnn).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.CrossEntropyLoss()

    trainer = Trainer(config)

    trainer.tb_logger.writer.add_graph(model=model,input_to_model=torch.randn(128,1,3000).to(device),verbose=True)

    if config.verbose >= 2:
        print(model)
        print(optimizer)
        print(crit)

    trainer.train(model, crit, optimizer, train_loader, valid_loader)
    trainer.test(test_loader)
    trainer.tb_logger.close()



if __name__ == '__main__':
    config = define_argparser()
    main(config)
