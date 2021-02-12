import argparse 

import torch 
import torch.nn as nn
import torch.optim as optim
from ignite.engine import Events
from ignite.metrics import Accuracy
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.metrics import RunningAverage


from sleep_classification.models import DeepSleepNet
from sleep_classification.data_loader import load_dataloader_for_featureNet
from sleep_classification.trainer import Trainer, MyEngine

def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_fn', required=True)
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

    model = DeepSleepNet(input_dim=1,n_classes=5,is_train=True,use_dropout=config.use_dropout,use_rnn=config.use_rnn).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.CrossEntropyLoss()

    data = torch.load("./folder0_model.pth")
    model.load_state_dict(data["model"])

    def validate(engine, mini_batch):
        print(1)
        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch
            x, y = x.to(engine.device), y.to(engine.device)

            y_hat = engine.model(x)

            loss = engine.crit(y_hat,y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy =  (torch.argmax(y_hat,dim = -1) == y).sum() /float(y.size(0))
            else:
                accuracy = 0
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy)
        }

    test_engine = MyEngine(
        validate,
        model,
        crit,
        optimizer,
        config
    )
    

    if config.verbose >= 2:
        print(model)
        print(optimizer)
        print(crit)

    def log_metrics(engine, title):
        print(engine.state.metrics.items())
        print(f"{title} accuracy: {engine.state.metrics['accuracy']:.2f}")

    test_engine.add_event_handler(
        Events.EPOCH_COMPLETED,
        log_metrics,
        'test'
    )

    RunningAverage(output_transform=lambda x: x['accuracy']).attach(test_engine, 'accuracy')
    pbar = ProgressBar()
    pbar.attach(test_engine, ['accuracy'])

    test_engine.run(
            test_loader,
            max_epochs=1
    )

if __name__ == '__main__':
    config = define_argparser()
    main(config)
