import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
import os
from utils.util import set_global_seed
import numpy as np
import time
from utils.visualization.generic import AnimationRenderer

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, criterion_params, metrics, optimizer, config, seed):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.seed = seed
        self.logger.info(f"Training seed: {seed}")

        self.model = model
        self.criterion = criterion
        self.criterion_params = criterion_params
        self.metrics = metrics
        self.optimizer = optimizer


        cfg_trainer = config['trainer']
        self.num_samples_to_track = cfg_trainer['num_samples_to_track'] if "num_samples_to_track" in cfg_trainer else 0
        self.tracking_period = cfg_trainer['tracking_period'] if "tracking_period" in cfg_trainer else -1
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
            self.mnt_best_stored = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.mnt_best_stored = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """

        # if resuming => we iterate the data_loader "self.start_epoch" times, in order to keep results reproducible as if
        # we did the training from the beginning without stopping.
        for epoch in range(1, self.start_epoch):
            set_global_seed(self.seed + epoch) # to allow for reproducibility after resuming checkpoint
            self._phantom_train_epoch()

        last_epoch_stored = -1
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            set_global_seed(self.seed + epoch) # to allow for reproducibility after resuming checkpoint
            t0 = time.time()
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            if self.lr_scheduler is not None:
                log['lr'] = self.lr_scheduler.get_last_lr()[-1]
            log.update(result)

            # print logged informations to the screen
            t = time.time() - t0
            message = f"[{int(round(t))}s]" if t < 600 else f"[{int(round(t / 60))}mins]"
            for key, value in log.items():
                message += '   {}: {:.6f}'.format(str(key), value) if isinstance(value, float) else '    {}: {:04d}'.format(str(key), int(value))
                #self.logger.info('    {:15s}: {}'.format(str(key), value)) # OLD FORMAT
            self.logger.info(message)

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                            (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                else:
                    not_improved_count += 1
                    
            if epoch % self.save_period == 0:
                self._process_checkpoint(epoch, log)
                last_epoch_stored = epoch

            if epoch % self.tracking_period == 0:
                self._track_samples(epoch)

            if self.mnt_mode != 'off' and not_improved_count > self.early_stop:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.early_stop))
                break

        if epoch % self.save_period != 0:
            self._process_checkpoint(epoch, log)
            last_epoch_stored = epoch
            
        self._track_samples(epoch)
        return last_epoch_stored

    def _process_checkpoint(self, epoch, log):
        best_to_be_stored = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best_stored) or \
                                (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best_stored)
        if best_to_be_stored:
            self.mnt_best_stored = log[self.mnt_metric]

        self._save_checkpoint(epoch, save_best=best_to_be_stored, val_metric=self.mnt_best_stored)
        if epoch // self.save_period > 1:
            self._delete_checkpoint(epoch - self.save_period)
        return epoch

    def _delete_checkpoint(self, epoch, save_best=False):
        """
        Deleting checkpoints
        :param epoch: checkpoint epoch number to remove
        """
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        if os.path.exists(filename):
            os.remove(filename)

    def _save_checkpoint(self, epoch, save_best=False, val_metric=None):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'monitor_best_stored': self.mnt_best_stored,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info(f"Saving current best: model_best.pth (mnt={val_metric})...")
        return filename

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))

        checkpoint = torch.load(resume_path)

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.mnt_best_stored = checkpoint['monitor_best_stored']

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))



    def _track_samples(self, epoch):
        if self.valid_data_loader is None or self.tracking_period <= 0 or self.num_samples_to_track == 0:
            return
        
        self.logger.info('Tracking samples...')
        self.model.eval()
        with torch.no_grad():
            pose_generator = self.get_pose_generator()

            algos = ["sample", ]
            skeleton = self.valid_data_loader.dataset.skeleton

            output_dir = str(self.checkpoint_dir / str(epoch))
            os.makedirs(output_dir, exist_ok=True)
            AnimationRenderer(skeleton, pose_generator, algos, self.model.obs_length, self.model.pred_length, 
                                ncol=4, size=4, output_dir=output_dir, baselines=["context", "gt"], type=skeleton.dim).store_all()

        self.logger.info('Tracking samples done.')


    def get_pose_generator(self):
        raise NotImplementedError