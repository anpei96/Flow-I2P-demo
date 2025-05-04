import time

from config import make_cfg
from dataset import train_valid_data_loader
from loss import EvalFunction, OverallLoss, OverallLoss_Cof
from anpei_model import create_model_an

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
        model = create_model_an(cfg)
        model = self.register_model(model)

        '''
        note-0524:
            in the second stage, training talk-net
        '''
        base_path = "/media/anpei/DiskC/2d-3d-reg-work/codes/2D3DMATR-main/scene_plus/model_zoos/"
        # ckpt_path = base_path + "base_grad_5p5p.pth"
        # self.load(ckpt_path)
        # self._max_epoch = 30 #25

        # ckpt_path = base_path + "baseline-all-9.pth"
        # ckpt_path = base_path + "epoch-15.pth" # full-model
        # self.load(ckpt_path)
        # self._max_epoch = 20

        # optimizer, scheduler
        optimizer = build_optimizer(model, cfg)
        self.register_optimizer(optimizer)
        scheduler = build_scheduler(optimizer, cfg)
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = OverallLoss(cfg)
        self.eval_func = EvalFunction(cfg)

        '''
        note-0527:
            in the second stage, using another loss, 
        '''
        # self.loss_func = OverallLoss_Cof(cfg)

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(data_dict, output_dict)
        # result_dict = self.eval_func(data_dict, output_dict)
        # loss_dict.update(result_dict)

        # test model params and flops
        from thop import profile
        flops, params = profile(self.model, inputs=(data_dict,))
        print(flops)
        print(params)
        assert 1==-1

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
