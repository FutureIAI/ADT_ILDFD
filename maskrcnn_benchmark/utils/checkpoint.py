# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch

from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.model_zoo import cache_url


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, trim=False, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if not trim:
            if self.optimizer is not None:
                data["optimizer"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                data["scheduler"] = self.scheduler.state_dict()
            data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        if not trim:
            self.tag_last_checkpoint(save_file)

    def load(self, f=None,load_CTN=False):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            self.logger.info("Overriding ckpt config with last_checkpoint")
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint,load_CTN)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def load_i(self, f=None,i=None,load_CTN=False):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            self.logger.info("Overriding ckpt config with last_checkpoint")
            f = self.get_checkpoint_file()
            file_name = os.path.split(f)[-1]

            test_number = (i+1)*self.cfg.SOLVER.CHECKPOINT_PERIOD
            test_file_name = "model_" + str(test_number)+".pth"
            test_path = os.path.join(os.path.dirname(f),test_file_name)
            f = test_path
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        print(f)
        self._load_model(checkpoint,load_CTN)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint,load_CTN):
        load_state_dict(self.model, checkpoint.pop("model"),load_CTN)


class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "maskrcnn_benchmark.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded:
            loaded = dict(model=loaded)
        return loaded
