import time

from config import make_cfg
from dataset import train_valid_data_loader
from loss import EvalFunction, OverallLoss, OverallLoss_Cof
from anpei_model import create_model_an
from anpei_model_sal import create_model_an_sal

from vision3d.engine import EpochBasedTrainer
from vision3d.utils.optimizer import build_optimizer, build_scheduler


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg)
        loading_time = time.time() - start_time
        self.log(f"Data loader created: {loading_time:.3f}s collapsed.", level="DEBUG")
        self.log(f"Calibrate neighbors: {neighbor_limits}.")
        self.register_loader(train_loader, val_loader)
    
        '''
        note-0229:
            our 2d-3d registration model *-*
        '''
        # model = create_model_an(cfg)
        # model = self.register_model(model)
        '''
        note-0522:
            stage two training with co-observable saliency map
        '''
        model = create_model_an_sal(cfg)
        model = self.register_model(model)

        # optimizer, scheduler
        optimizer = build_optimizer(model, cfg)
        self.register_optimizer(optimizer)
        scheduler = build_scheduler(optimizer, cfg)
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = OverallLoss_Cof(cfg)
        self.eval_func = EvalFunction(cfg)

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(data_dict, output_dict)
        # result_dict = self.eval_func(data_dict, output_dict)
        # loss_dict.update(result_dict)
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(data_dict, output_dict)
        result_dict = self.eval_func(data_dict, output_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict


def main():
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run_only_train()


if __name__ == "__main__":
    main()
