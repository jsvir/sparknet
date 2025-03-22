from abc import abstractmethod
from math import ceil
from typing import Dict, Optional, Union
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from nemo.collections.asr.data import audio_to_label_dataset
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.common.metrics import TopKClassificationAccuracy
from nemo.core.neural_types import *
from nemo.utils import logging, model_utils
import math
from typing import List



__all__ = ['EncDecClassificationModel']


class _EncDecBaseModel(ASRModel, ExportableEncDecModel):
    """Encoder decoder Classification models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_gpus
        # Convert config to a DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        # Convert config to support Hydra 1.0+ instantiation
        cfg = model_utils.maybe_update_config_version(cfg)
        self.is_regression_task = cfg.get('is_regression_task', False)
        super().__init__(cfg=cfg, trainer=trainer)
        if hasattr(self._cfg, 'spec_augment') and self._cfg.spec_augment is not None:
            self.spec_augmentation = ASRModel.from_config_dict(self._cfg.spec_augment)
        else:
            self.spec_augmentation = None
        if hasattr(self._cfg, 'crop_or_pad_augment') and self._cfg.crop_or_pad_augment is not None:
            self.crop_or_pad = ASRModel.from_config_dict(self._cfg.crop_or_pad_augment)
        else:
            self.crop_or_pad = None
        self.preprocessor = self._setup_preprocessor()
        if hasattr(self._cfg, 'feature_selector') and self._cfg.feature_selector is not None:
            self.fs = self._setup_fs()
            self.fs.reg_lamba = self._cfg.reg_lamba
        else:
            self.fs = None
        self.loss = self._setup_loss()
        self._setup_metrics()

    @abstractmethod
    def _setup_preprocessor(self):
        """
        Setup preprocessor for audio data
        Returns: Preprocessor

        """
        pass

    @abstractmethod
    def _setup_encoder(self):
        """
        Setup encoder for the Encoder-Decoder network
        Returns: Encoder
        """
        pass

    @abstractmethod
    def _setup_fs(self):
        """
        Setup encoder for the Encoder-Decoder network
        Returns: Encoder
        """
        pass

    @abstractmethod
    def _setup_decoder(self):
        """
        Setup decoder for the Encoder-Decoder network
        Returns: Decoder
        """
        pass

    @abstractmethod
    def _setup_loss(self):
        """
        Setup loss function for training
        Returns: Loss function

        """
        pass

    @abstractmethod
    def _setup_metrics(self):
        """
        Setup metrics to be tracked in addition to loss
        Returns: void

        """
        pass

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        if hasattr(self.preprocessor, '_sample_rate'):
            audio_eltype = AudioSignal(freq=self.preprocessor._sample_rate)
        else:
            audio_eltype = AudioSignal()
        return {
            "input_signal": NeuralType(('B', 'T'), audio_eltype),
            "input_signal_length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @abstractmethod
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        pass

    def transcribe(self, paths2audio_files: List[str], batch_size: int = 4) -> List[str]:
        pass
    def forward(self, input_signal, input_signal_length, labels=None):
        processed_signal, processed_signal_len = self.preprocessor(
            input_signal=input_signal, length=input_signal_length,
        )
        # Crop or pad is always applied
        if self.crop_or_pad is not None:
            processed_signal, processed_signal_len = self.crop_or_pad(
                input_signal=processed_signal, length=processed_signal_len
            )
        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_len)

        processed_signal_min = processed_signal.min(dim=-1, keepdim=True).values.min(dim=0, keepdim=True).values
        processed_signal_normalized = processed_signal - processed_signal_min
        fs_out, _ = self.fs(audio_signal=processed_signal, length=processed_signal_len)
        fs_out = self.output_layer(fs_out)
        noise = torch.normal(mean=torch.zeros_like(fs_out, device=fs_out.device), std=torch.ones_like(fs_out, device=fs_out.device) * 0.5)
        z = fs_out + noise * self.training
        stochastic_gate = torch.clamp(z + 0.5, 0.0, 1.0)
        gates_logits = self.freq_linear_proj(stochastic_gate.mean(dim=2))
        return gates_logits, processed_signal_normalized, processed_signal, fs_out, gates_logits

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True
        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=DictConfig(train_data_config))

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=DictConfig(val_data_config))

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=DictConfig(test_data_config))

    def test_dataloader(self):
        if self._test_dl is not None:
            return self._test_dl

    def _setup_dataloader_from_config(self, config: DictConfig):
        OmegaConf.set_struct(config, False)
        config.is_regression_task = self.is_regression_task
        OmegaConf.set_struct(config, True)
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None
        featurizer = WaveformFeaturizer(
            sample_rate=config['sample_rate'], int_values=config.get('int_values', False), augmentor=augmentor
        )
        shuffle = config['shuffle']
        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                    'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` is None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            if 'vad_stream' in config and config['vad_stream']:
                logging.warning("VAD inference does not support tarred dataset now")
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_label_dataset.get_tarred_classification_label_dataset(
                featurizer=featurizer,
                config=OmegaConf.to_container(config),
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
            )
            shuffle = False
            batch_size = config['batch_size']
            collate_func = dataset.collate_fn

        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` is None. Provided config : {config}")
                return None

            if 'vad_stream' in config and config['vad_stream']:
                logging.info("Perform streaming frame-level VAD")
                dataset = audio_to_label_dataset.get_speech_label_dataset(
                    featurizer=featurizer, config=OmegaConf.to_container(config)
                )
                batch_size = 1
                collate_func = dataset.vad_frame_seq_collate_fn
            else:
                dataset = audio_to_label_dataset.get_classification_label_dataset(
                    featurizer=featurizer, config=OmegaConf.to_container(config)
                )
                batch_size = config['batch_size']
                collate_func = dataset.collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_func,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

class EncDecClassificationModel(_EncDecBaseModel):
    """Encoder decoder Classification models."""
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        if cfg.get("is_regression_task", False):
            raise ValueError(f"EndDecClassificationModel requires the flag is_regression_task to be set as false")
        super().__init__(cfg=cfg, trainer=trainer)
        self.output_layer = torch.nn.Sequential(
            torch.nn.Conv1d(cfg.feature_selector.jasper[-1].filters, 32, 1, 1),
            torch.nn.BatchNorm1d(32),
            torch.nn.Tanh()
        )
        self.freq_linear_proj = torch.nn.Linear(32, len(cfg.labels))

    def _setup_preprocessor(self):
        return EncDecClassificationModel.from_config_dict(self._cfg.preprocessor)

    def _setup_encoder(self):
        return EncDecClassificationModel.from_config_dict(self._cfg.encoder)

    def _setup_fs(self):
        return EncDecClassificationModel.from_config_dict(self._cfg.feature_selector)

    def _setup_decoder(self):
        return EncDecClassificationModel.from_config_dict(self._cfg.decoder)

    def _setup_loss(self):
        return CrossEntropyLoss()

    def _setup_metrics(self):
        self._accuracy = TopKClassificationAccuracy(dist_sync_on_step=True)
        self._accuracy_gates = TopKClassificationAccuracy(dist_sync_on_step=True)

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"outputs": NeuralType(('B', 'D'), LogitsType()),
                "features": NeuralType(('B', 'D', 'T'), MFCCSpectrogramType()),
                "gated_features": NeuralType(('B', 'D', 'T'), MFCCSpectrogramType()),
                "fs_out": NeuralType(('B', 'D', 'T'), LogitsType()),
                "logits_gates": NeuralType(('B','D'), LogitsType()),
                }

    def regularization_loss(self, mu, y):
        reg_loss = torch.mean(0.5 - 0.5 * torch.erf((-1 / 2 - mu) / (math.sqrt(2) * 0.5)))
        self.log('reg_loss', reg_loss.item())
        return reg_loss

    def training_step(self, batch, batch_nb):
        audio_signal, audio_signal_len, labels, labels_len = batch
        if self.fs is not None:
            gates_labels = labels.clone()
            logits, features, gated_features, fs_out, logits_gates = self.forward(
                input_signal=audio_signal,
                input_signal_length=audio_signal_len,
                labels=labels)

            loss_value_gates = self.loss(logits=logits_gates, labels=gates_labels)
            self.log('train_loss_gates', loss_value_gates.item())
            reg_loss = self.fs.reg_lamba * self.regularization_loss(fs_out, gates_labels)
            loss_value = 100*loss_value_gates + reg_loss
            self._accuracy(logits=logits, labels=labels)
            self._accuracy_gates(logits=logits_gates, labels=gates_labels)
            topk_scores_gates = self._accuracy_gates.compute()
            self._accuracy_gates.reset()
            for top_k, score in zip(self._accuracy_gates.top_k, topk_scores_gates):
                self.log('training_batch_accuracy_gates_top@{}'.format(top_k), score)
        else:
            logits, features, gated_features, fs_out = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
            loss_value = self.loss(logits=logits, labels=labels)
            self._accuracy(logits=logits, labels=labels)

        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])

        topk_scores = self._accuracy.compute()
        self._accuracy.reset()

        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            self.log('training_batch_accuracy_top@{}'.format(top_k), score)

        return {
            'loss': loss_value,
        }

    def configure_optimizers(self):
        self.setup_optimization()

        if self._scheduler is None:
            return self._optimizer
        else:
            return [self._optimizer], [self._scheduler]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits, features, gated_features, fs_out, logits_gates = self.forward(
            input_signal=audio_signal,
            input_signal_length=audio_signal_len)
        gates_labels = labels
        loss_value_gates = self.loss(logits=logits_gates, labels=gates_labels)
        acc_gates = self._accuracy_gates(logits=logits_gates, labels=gates_labels)
        correct_counts_gates, total_counts_gates = self._accuracy_gates.correct_counts_k, self._accuracy_gates.total_counts_k
        loss_value = self.loss(logits=logits, labels=labels)
        acc = self._accuracy(logits=logits, labels=labels)
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k

        return {
            'val_loss': loss_value,
            'val_correct_counts': correct_counts,
            'val_total_counts': total_counts,
            'val_acc': acc,
            'val_loss_gates': loss_value_gates,
            'val_acc_gates': acc_gates,
            'val_correct_counts_gates': correct_counts_gates,
            'val_total_counts_gates': total_counts_gates,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits, features, gated_features, fs_out, logits_gates = self.forward(
            input_signal=audio_signal,
            input_signal_length=audio_signal_len)
        loss_value = self.loss(logits=logits, labels=labels)
        acc = self._accuracy(logits=logits, labels=labels)
        gates_labels = labels
        loss_value_gates = self.loss(logits=logits_gates, labels=gates_labels)
        acc_gates = self._accuracy_gates(logits=logits_gates, labels=gates_labels)
        correct_counts_gates, total_counts_gates = self._accuracy_gates.correct_counts_k, self._accuracy_gates.total_counts_k
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k
        return {
            'test_loss': loss_value,
            'test_correct_counts': correct_counts,
            'test_total_counts': total_counts,
            'test_acc': acc,

            'test_loss_gates': loss_value_gates,
            'test_acc_gates': acc_gates,
            'test_correct_counts_gates': correct_counts_gates,
            'test_total_counts_gates': total_counts_gates,
        }

    def multi_validation_epoch_end_gates(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss_gates'] for x in outputs]).mean()
        correct_counts = torch.stack([x['val_correct_counts_gates'] for x in outputs]).sum(axis=0)
        total_counts = torch.stack([x['val_total_counts_gates'] for x in outputs]).sum(axis=0)

        self._accuracy_gates.correct_counts_k = correct_counts
        self._accuracy_gates.total_counts_k = total_counts
        topk_scores = self._accuracy_gates.compute()
        self._accuracy_gates.reset()

        tensorboard_log = {'val_loss_gates': val_loss_mean}
        for top_k, score in zip(self._accuracy_gates.top_k, topk_scores):
            tensorboard_log['val_gates_epoch_top@{}'.format(top_k)] = score

        return tensorboard_log
    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct_counts = torch.stack([x['val_correct_counts'] for x in outputs]).sum(axis=0)
        total_counts = torch.stack([x['val_total_counts'] for x in outputs]).sum(axis=0)

        self._accuracy.correct_counts_k = correct_counts
        self._accuracy.total_counts_k = total_counts
        topk_scores = self._accuracy.compute()
        self._accuracy.reset()

        tensorboard_log = {'val_loss': val_loss_mean}
        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            tensorboard_log['val_epoch_top@{}'.format(top_k)] = score
        tensorboard_log.update(self.multi_validation_epoch_end_gates(outputs))
        return {'log': tensorboard_log}

    def multi_test_epoch_end_gates(self, outputs, dataloader_idx: int = 0):
        test_loss_mean_gates = torch.stack([x['test_loss_gates'] for x in outputs]).mean()
        correct_counts_gates = torch.stack([x['test_correct_counts_gates'].unsqueeze(0) for x in outputs]).sum(axis=0)
        total_counts_gates = torch.stack([x['test_total_counts_gates'].unsqueeze(0) for x in outputs]).sum(axis=0)

        self._accuracy_gates.correct_counts_k = correct_counts_gates
        self._accuracy_gates.total_counts_k = total_counts_gates
        topk_scores_gates = self._accuracy_gates.compute()
        self._accuracy_gates.reset()

        tensorboard_log = {'test_loss_gates': test_loss_mean_gates}
        for top_k, score in zip(self._accuracy_gates.top_k, topk_scores_gates):
            tensorboard_log['test_gates_epoch_top@{}'.format(top_k)] = score
        return tensorboard_log

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        correct_counts = torch.stack([x['test_correct_counts'].unsqueeze(0) for x in outputs]).sum(axis=0)
        total_counts = torch.stack([x['test_total_counts'].unsqueeze(0) for x in outputs]).sum(axis=0)

        self._accuracy.correct_counts_k = correct_counts
        self._accuracy.total_counts_k = total_counts
        topk_scores = self._accuracy.compute()
        self._accuracy.reset()

        tensorboard_log = {'test_loss': test_loss_mean}
        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            tensorboard_log['test_epoch_top@{}'.format(top_k)] = score
        tensorboard_log.update(self.multi_test_epoch_end_gates(outputs))
        return {'log': tensorboard_log}

    # @typecheck()
    def forward(self, input_signal, input_signal_length, labels=None):
        logits = super().forward(input_signal=input_signal, input_signal_length=input_signal_length, labels=labels)
        return logits
