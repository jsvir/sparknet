import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from model import EncDecClassificationModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import time
import torch
from thop import profile


class AuxModel(torch.nn.Module):
    def __init__(self, asr_model):
        super().__init__()
        self.fs = asr_model.fs.cuda()
        self.output_layer = asr_model.output_layer.cuda()
        self.freq_linear_proj = asr_model.freq_linear_proj.cuda()

    def forward(self, x, x_len):
        with torch.no_grad():
            o, _ = self.fs(x, x_len)
            o = self.output_layer(o)
            stochastic_gate = torch.clamp(o + 0.5, 0.0, 1.0)
            o = self.freq_linear_proj(stochastic_gate.mean(dim=2))
        return o


@hydra_runner(config_path="conf", config_name="cfg_labels_12_channels_16_data_v2")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    seed_everything(1948)
    trainer = pl.Trainer(**cfg.trainer)
    cfg.exp_manager.version = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    exp_manager(trainer, cfg.get("exp_manager", None))
    kws_model = EncDecClassificationModel(cfg=cfg.model, trainer=trainer)
    trainer.fit(kws_model)
    if kws_model.prepare_test(trainer):
        trainer.test(kws_model)
        macs, params = profile(
            model=AuxModel(kws_model),
            inputs=(torch.rand(1, 32, 101).float().cuda(), torch.ones(1).long().cuda() * 101))
        print("MACS:", macs)
        print("PARAMS:", params)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
