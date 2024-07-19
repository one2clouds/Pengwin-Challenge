import re
from typing import Any, Callable, Tuple, Optional

import torch
# from pytorch_lightning import LightningModule
from lightning import LightningModule
from torch import nn
from torchmetrics import MeanMetric, MinMetric
from monai.transforms import EnsureType, AsDiscrete, ScaleIntensityRange

from monai.metrics.meandice import compute_dice
from monai.metrics.meaniou import compute_iou

from PIL import Image
from lightning.pytorch.loggers import WandbLogger
import imageio
import torchvision

import numpy as np
from torchvision.transforms import Compose, ToTensor, ToPILImage

# def a_3d_img_to_list_of_numpy(original_img, targets_img, pred_img) -> None:
#     frames_org, frames_tar, frames_pred = [], [], []
#     for org_img, tar_img, pred_img in zip(original_img, targets_img, pred_img):
#         org_img = torchvision.transforms.ToPILImage()(org_img)
        # tar_img = torchvision.transforms.ToPILImage()(tar_img)
        # pred_img = torchvision.transforms.ToPILImage()(pred_img)

        # frames_org.append(org_img)
        # frames_tar.append(tar_img)
        # frames_pred.append(pred_img)

    # print(type(frames))
    # return frames
    # frames_org[0].save('original_img.gif',save_all=True,append_images=frames_org,optimize=True,duration=200,loop=0)
    # frames_tar[0].save('target_img.gif',save_all=True,append_images=frames_tar,optimize=True,duration=200,loop=0)
    # frames_pred[0].save('predicted_img.gif',save_all=True,append_images=frames_pred,optimize=True,duration=200,loop=0)
    # gif = imageio.get_reader('image.gif')
    # return gif

def label_to_PIL(img, num_of_labels=4):
    "Only image of 2 dimension is expected for example 128 x 128"
    rgb_img = img.tile(3,1,1) # making it 3 dimensional by stacking up layers
    for i in range(rgb_img.shape[0]): # only applying it across 3 dimensional as our image has 3 channels R G B
        if i == 0:
            rgb_img[i] = torch.where(rgb_img[i] == 1,255,0)
        elif i == 1:
            rgb_img[i] = torch.where(rgb_img[i] == 2,255,0)
        elif i == 2:
            rgb_img[i] = torch.where(rgb_img[i] == 3,255,0)
    return rgb_img


def a_3d_img_to_tif(a_3d_img, file_name, segmentation:bool=False) -> None:
    frames_org = []
    # my_transform = Compose([
    #     ToTensor(), # ToTensor is used to make scaling possible 
    #     ToPILImage() # To pil image is used to make tensor a PIL Image
    # ])
    for img in a_3d_img.permute(1,2,0): # permute to change the angle to feed in wandb
        # print(img.max())
        # print(img.min())
        scale_img = ScaleIntensityRange(a_min=img.min(), a_max=img.max(), b_min=0, b_max=1)
        if segmentation:
            # Divide by 255 because topilimage expects image in range 0 to 1 and we have provided image of 255
            img = ToPILImage()(torch.as_tensor(label_to_PIL(img)/255, dtype=torch.float32))
        else:
            # ToTensor expects image to be in H W Channel dim so we need to permute it 
            img = ToPILImage()(torch.as_tensor(scale_img(img), dtype=torch.float32))
        frames_org.append(img)

    frames_org[0].save(file_name,save_all=True,append_images=frames_org,optimize=True,duration=200,loop=0)
    # gif = imageio.get_reader('image.gif')
    # return gif

class BaseModule(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        dice_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        iou_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scheduler: Optional[torch.optim.lr_scheduler.LinearLR] = None,
        scheduler_monitor: str = "val/loss",
        out_channels: int = 4,
        ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.net = net
        # loss function
        self.loss_fn = loss_fn

        self.dice_metric = dice_metric
        self.iou_metric = iou_metric

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_loss_min = MinMetric()

        # putting dimension of AsDiscrete as 1 because we want to make one hot in 1th dimension and 0th dimension is batchsize
        self.post_pred = Compose([AsDiscrete(to_onehot=self.hparams.out_channels, threshold=0.5, dim=1)])

    def forward(self, **kwargs):
        return self.net(**kwargs)
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch["image"], batch["label"]

        # print(x.shape) # torch.Size([4, 1, 128, 128, 128])
        # print(y.shape) # torch.Size([4, 1, 128, 128, 128])
        # print(x.max()) # tensor(9.2463) # because of normalize intensity
        # print(x.min()) # tensor(-7.0969)
        # print(y.unique()) # tensor([0., 1., 2., 3.])

        logits = self.forward(**batch)
        softmax_logits = nn.Softmax(dim=1)(logits)

        # print(softmax_logits.max()) #tensor(0.9980, device='cuda:0')
        # print(softmax_logits.min()) #tensor(3.7885e-07, device='cuda:0')
        # print(softmax_logits.shape) #torch.Size([4, 5, 128, 128, 128])
        # print(y.shape) # torch.Size([4, 1, 128, 128, 128])
        y_post_pred_to_onehot = self.post_pred(y)
        # print(y_post_pred_to_onehot.max()) # tensor(1.)
        # print(y_post_pred_to_onehot.min()) # tensor(0.)
        # print(y_post_pred_to_onehot.shape) # torch.Size([4, 5, 128, 128, 128])


        loss = self.loss_fn(softmax_logits, y_post_pred_to_onehot)
        # print(loss)

        as_discrete = AsDiscrete(argmax=True, to_onehot=self.hparams.out_channels, threshold=0.5, dim=1)

        # print(as_discrete(softmax_logits).shape) # torch.Size([1, 5, 128, 128, 128])
        # print(y_post_pred_to_onehot.shape) # torch.Size([1, 5, 128, 128, 128])

        dice_score = self.dice_metric(as_discrete(softmax_logits), y_post_pred_to_onehot)
        iou_score = self.iou_metric(as_discrete(softmax_logits), y_post_pred_to_onehot)

        return loss, as_discrete(softmax_logits), y_post_pred_to_onehot, dice_score, iou_score, x

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        loss, preds, targets, dice_score, iou_score, original_img = self.model_step(batch)
        # print(preds.shape) # torch.Size([2, 4, 128, 128, 128]) # batch size=2
        # print(targets.shape) # torch.Size([2, 4, 128, 128, 128])
        # print(original_img.shape) # torch.Size([2, 1, 128, 128, 128])

        # update and log metrics
        pred_arg = preds[0].argmax(dim=0) # Taking 1st term of batch and argmax it 
        targets_arg = targets[0].argmax(dim=0).squeeze() # Taking 1st term of batch and argmax it
        original_img = original_img[0][0] # Taking 1st term of batch and taking 1st channel as there is only 1 channel

        # print(pred_arg.shape) #torch.Size([128, 128, 128])
        # print(targets_arg.shape) #torch.Size([128, 128, 128])
        # print(original_img.shape) #torch.Size([128, 128, 128])
        
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({"dice_score_train" : dice_score.mean(), "iou_score_train" : iou_score.mean()})

        if isinstance(self.logger, WandbLogger):
            a_3d_img_to_tif(original_img, 'original_img.gif')
            self.logger.log_video(key="train/original_img", videos=['original_img.gif'])
            a_3d_img_to_tif(targets_arg, 'target_img.gif', segmentation=True)
            self.logger.log_video(key="train/target_img", videos=['target_img.gif'])
            a_3d_img_to_tif(pred_arg, 'predicted_img.gif', segmentation=True)
            self.logger.log_video(key="train/pred_img", videos=['predicted_img.gif'])

        # return loss or backpropagation will fail
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets, dice_score, iou_score, original_img = self.model_step(batch)

        # update and log metrics
        pred_arg = preds[0].argmax(dim=0) # Taking 1st term of batch and argmax it 
        targets_arg = targets[0].argmax(dim=0).squeeze() # Taking 1st term of batch and argmax it
        original_img = original_img[0][0] # Taking 1st term of batch and taking 1st channel as there is only 1 channel

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss.compute().item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({"dice_score_val" : dice_score.mean(), "iou_score_val" : iou_score.mean()})

        if isinstance(self.logger, WandbLogger):
            a_3d_img_to_tif(original_img, 'original_img.gif')
            self.logger.log_video(key="val/original_img", videos=['original_img.gif'])
            a_3d_img_to_tif(targets_arg, 'target_img.gif', segmentation=True)
            self.logger.log_video(key="val/target_img", videos=['target_img.gif'])
            a_3d_img_to_tif(pred_arg, 'predicted_img.gif', segmentation=True)
            self.logger.log_video(key="val/pred_img", videos=['predicted_img.gif'])

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute() # gives current loss from batch 
        self.val_loss_min(loss) # logs our loss into the min 

        # self.log("val/loss", self.val_loss_min(loss).item(), sync_dist=True, prog_bar=True) # get the lowest value from the above


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets, dice_score, iou_score, original_img = self.model_step(batch)

        # update and log metrics
        pred_arg = preds[0].argmax(dim=0) # Taking 1st term of batch and argmax it 
        targets_arg = targets[0].argmax(dim=0).squeeze() # Taking 1st term of batch and argmax it
        original_img = original_img[0][0] # Taking 1st term of batch and taking 1st channel as there is only 1 channel

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({"dice_score_test" : dice_score.mean(), "iou_score_test" : iou_score.mean()})

        if isinstance(self.logger, WandbLogger):
            a_3d_img_to_tif(original_img, 'original_img.gif')
            self.logger.log_video(key="test/original_img", videos=['original_img.gif'])
            a_3d_img_to_tif(targets_arg,'target_img.gif', segmentation=True )
            self.logger.log_video(key="test/target_img", videos=['target_img.gif'])
            a_3d_img_to_tif(pred_arg, 'predicted_img.gif', segmentation=True)
            self.logger.log_video(key="test/pred_img", videos=['predicted_img.gif'])


    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.scheduler_monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)

