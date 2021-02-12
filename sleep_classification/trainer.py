from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils

from sleep_classification.utils import get_grad_norm,get_parameter_norm

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

class MyEngine(Engine):
    def __init__(self, func, model, crit, optimizer, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config

        super().__init__(func)

        self.best_loss =np.inf
        self.best_model = None

        self.device = next(model.parameters()).device

    @staticmethod
    def train(engine, mini_batch):
        engine.model.train()
        engine.optimizer.zero_grad()

        x,y = mini_batch
        x,y = x.to(engine.device), y.to(engine.device)

        y_hat = engine.model(x)

        loss = engine.crit(y_hat, y)
        loss.backward()

        #y 가 one-hot인지
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy =  (torch.argmax(y_hat,dim = -1) == y).sum() /float(y.size(0))
        else:
            accuracy = 0
        
        p_norm = float(get_parameter_norm(engine.model.parameters())) #모델의 복잡도 학습됨에 따라 커져야함
        g_norm = float(get_grad_norm(engine.model.parameters()))    #클수록 뭔가 배우는게 변하는게 많다 (학습의 안정성)

        if engine.config.max_grad > 0:
            torch_utils.clip_grad_norm_(
                engine.model.parameters(),
                engine.config.max_grad,
                norm_type=2,
            )

        engine.optimizer.step()

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm ,
        }

    @staticmethod
    def validate(engine, mini_batch):
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

    @staticmethod
    def test(engine,mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x,y = mini_batch
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

    @staticmethod
    def attach(train_engine, validation_engine, verbose = VERBOSE_BATCH_WISE): 
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform = lambda x: x[metric_name]).attach(
                engine,
                metric_name
            )

        training_metric_name = ['loss', 'accuracy', '|param|', '|g_param|']

        for metric_name in training_metric_name:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format = None, ncols=120)
            pbar.attach(train_engine, training_metric_name)
        
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                print('Epoch {} - |params| = {:.2e} |g_param| = {:.2e} loss = {:.4e} accuracy = {:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['|param|'],
                    engine.state.metrics['|g_param|'],
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                ))

        validation_metrics_name = ['loss','accuracy']

        for metrics in validation_metrics_name:
            attach_running_average(validation_engine, metrics)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format = None, ncols=120)
            pbar.attach(validation_engine, validation_metrics_name)
        
        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('Validation - loss = {:.4e} accuracy = {:.4f} best_loss = {:.4e}'.format(             
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                    engine.best_loss
                ))

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss
            engine.best_model = deepcopy(engine.model.state_dict())
        
    @staticmethod
    def save_model(engine, train_engine, config, **kwargs):
        torch.save(
            {
                'model':engine.best_model,
                'config': config,
                **kwargs
            },"folder" + str(config.fold_idx) +"_"+config.model_fn
        )
        engine.model.load_state_dict(engine.best_model)




class Trainer():
    def __init__(self,config):
        self.config = config
        super().__init__()

    def train(
        self,
        model,crit,optimizer,train_loader,valid_loader
    ):
        self.train_engine = MyEngine(
            MyEngine.train,
            model, crit, optimizer, self.config
        )

        self.valid_engine = MyEngine(
            MyEngine.validate,
            model, crit, optimizer,self.config
        )

        MyEngine.attach(
            self.train_engine,
            self.valid_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, valid_engine, valid_loader):
            valid_engine.run(valid_loader, max_epochs =1)

        self.train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, #event
            run_validation, #func
            self.valid_engine, valid_loader #args
        )

        self.valid_engine.add_event_handler(
            Events.EPOCH_COMPLETED, #event
            MyEngine.check_best #func
        )

        self.valid_engine.add_event_handler(
            Events.EPOCH_COMPLETED, #event
            MyEngine.save_model, # func
            self.train_engine, self.config, #args
        )

        self.train_engine.run(
            train_loader,
            max_epochs = self.config.n_epochs
        )

        return model

    def test(self,test_loader):

        self.valid_engine.run(
            test_loader,
            max_epochs=1
        )