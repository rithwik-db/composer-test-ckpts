import torch
from torch.utils.data import Dataset
from composer.models import ComposerClassifier
import contextlib
import os
import tempfile
from pathlib import Path

import torch
from composer.core import Callback, State
from composer.core.state import fsdp_state_dict_type_context, fsdp_get_optim_state_dict
from composer.loggers import Logger
from composer.loggers.remote_uploader_downloader import RemoteUploaderDownloader
from composer.utils import (dist, format_name_with_dist_and_time, parse_uri,
                            reproducibility)

class RandomClassificationDataset(Dataset):
    """Classification dataset drawn from a normal distribution.

    Args:
        shape (Sequence[int]): shape of features (default: (1, 1, 1))
        size (int): number of samples (default: 100)
        num_classes (int): number of classes (default: 2)
    """

    def __init__(self, shape = (1, 1, 1), size: int = 100, num_classes: int = 2):
        self.size = size
        self.shape = shape
        self.num_classes = num_classes
        self.x = None
        self.y = None

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
        # Note: lazily generate data so it runs after Composer seeds everything, giving the same
        # dataset across multiple calls when using the same seed.
        if self.x is None:
            self.x = torch.randn(self.size, *self.shape)
        if self.y is None:
            self.y = torch.randint(0, self.num_classes, size=(self.size,))
        return self.x[index], self.y[index]



class SimpleMLP(ComposerClassifier):
    def __init__(self, num_features: int, num_classes: int = 4, 
                 train_metrics = None,
                 val_metrics = None):
        net = torch.nn.Sequential(
            torch.nn.Linear(num_features, num_features, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_features, num_classes, bias=False),
        )
        super().__init__(module=net, num_classes=num_classes,
                         train_metrics=train_metrics, val_metrics=val_metrics)



class MonolithicCheckpointSaver(Callback):
    """Save a monolithic checkpoint every N batches.

    Args:
        save_folder (str): Folder to save checkpoints to (can be a URI)
        filename (str): Filename to save checkpoints to.
        batch_interval (int): Number of batches between checkpoints.
        overwrite (bool): Whether to overwrite previous checkpoints.
        keep_optimizer(bool): Whether to save the optimizer state in the monolithic checkpoint.
    """

    def __init__(self,
                 save_folder: str,
                 batch_interval: int,
                 filename: str = 'ep{epoch}-ba{batch}.pt',
                 overwrite: bool = False,
                 keep_optimizers: bool = False):
        self.backend, self.bucket_name, self.save_dir_format_str = parse_uri(
            save_folder)
        self.filename_format_str = filename
        self.batch_interval = batch_interval
        self.upload_to_object_store = (self.backend != '')
        self.overwrite = overwrite
        self.keep_optimizers = keep_optimizers
        if self.upload_to_object_store:
            self.remote_ud = RemoteUploaderDownloader(
                bucket_uri=f'{self.backend}://{self.bucket_name}')
        else:
            self.remote_ud = None

    def init(self, state: State, logger: Logger):
        if self.upload_to_object_store and self.remote_ud is not None:
            self.remote_ud.init(state, logger)
            # updated_logger_destinations = [*logger.destinations, new_remote_ud]
            # logger.destinations = tuple(updated_logger_destinations)
            state.callbacks.append(self.remote_ud)

    def batch_checkpoint(self, state: State, logger: Logger):
        if state.timestamp.batch.value % self.batch_interval == 0:
            self._save_checkpoint(state, logger)

    def fit_end(self, state: State, logger: Logger):
        if state.timestamp.batch.value % self.batch_interval != 0:
            self._save_checkpoint(state, logger)

    def _save_checkpoint(self, state: State, logger: Logger):
        filename = format_name_with_dist_and_time(self.filename_format_str,
                                                  state.run_name,
                                                  state.timestamp)
        save_dir = format_name_with_dist_and_time(self.save_dir_format_str,
                                                  state.run_name,
                                                  state.timestamp)
        dir_context_mgr = tempfile.TemporaryDirectory(
        ) if self.upload_to_object_store else contextlib.nullcontext(
            enter_result=save_dir)
        with dir_context_mgr as temp_save_dir:
            save_path = str(Path(temp_save_dir) / Path(filename))
            dirname = os.path.dirname(save_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            state_dict = {
                'state': state.state_dict(),
                'rng': reproducibility.get_rng_state()
            }
            # Remove sharded model and optimizer state dicts
            state_dict['state'].pop('optimizers')
            state_dict['state'].pop('model')

            # Add in unsharded model params.
            with fsdp_state_dict_type_context(state.model,
                                              state_dict_type='full'):
                state_dict['state']['model'] = state.model.state_dict()

            # Add in unsharded optimizer state dict.
            if self.keep_optimizers:
                optimizer = state.optimizers[0]
                state_dict['state']['optimizers'] = {
                    type(optimizer).__qualname__: fsdp_get_optim_state_dict(
                                                                    state.model,
                                                                    optimizer, 
                                                                    state_dict_type='full')}
            if dist.get_global_rank() == 0:
                torch.save(state_dict, save_path)
            if self.upload_to_object_store and self.remote_ud is not None and dist.get_global_rank(
            ) == 0:
                remote_file_name = str(Path(save_dir) / Path(filename))
                self.remote_ud.upload_file(state=state,
                                           remote_file_name=remote_file_name,
                                           file_path=Path(save_path),
                                           overwrite=self.overwrite)
