import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from models.diffusion.resample import UniformSampler
import os
from einops import repeat
from eval_belfusion import BASELINES

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class DiffusionTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, diffusion, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, validation_frequency=1,
                 samples_epoch=None, seed=0, disable_profilers=True, load_mmgt=False,
                 ema_active=False, ema_decay=0.995, step_start_ema=2000, update_ema_every=10, debug=False):

        self.ema_active = ema_active # it needs to be done before super() for the "resume" scenario
        if self.ema_active:
            self.step_start_ema = step_start_ema
            self.update_ema_every = update_ema_every
            self.ema = EMA(ema_decay)

            #self.ema_model = copy.deepcopy(self.model) # this crashes
            self.ema_model = model.deepcopy().to(device)

        super().__init__(model, None, None, None, optimizer, config, seed)
        self.diffusion = diffusion
        self.schedule_sampler = UniformSampler(self.diffusion)
        self.debug = debug

        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size
        self.load_mmgt = load_mmgt

        if self.load_mmgt:
            valid_path = config["arch"]["args"]["embedder_pred_path"].replace(".pth", f"_mmGT_{valid_data_loader.dataset._get_hash_str()}.json")
            train_path = config["arch"]["args"]["embedder_pred_path"].replace(".pth", f"_mmGT_{data_loader.dataset._get_hash_str()}.json")
            valid_data_loader.dataset.load_mmgt(valid_path)
            data_loader.dataset.load_mmgt(train_path)

        assert not (self.data_loader.split_validation() is None and valid_data_loader is None and validation_frequency > 0), "No validation splits defined!"
        assert (self.data_loader.split_validation() is None or valid_data_loader is None) or validation_frequency == 0, "Two validation splits defined!"
        self.valid_data_loader = self.data_loader.split_validation() if valid_data_loader is None else valid_data_loader
        self.do_validation = self.valid_data_loader is not None and validation_frequency > 0
        self.validation_frequency = 0 if not self.do_validation else validation_frequency
        assert round(validation_frequency) == validation_frequency

        if samples_epoch is None:
            # epoch-based training -> samples_epoch is total num of samples from dataset
            self.n_iters = len(self.data_loader)
            self.n_samples = data_loader.n_samples
            self.epoch_based = True
        else:
            # num_samples-based training
            self.n_iters = samples_epoch // self.batch_size
            self.n_samples = samples_epoch
            self.epoch_based = False

        self.lr_scheduler = lr_scheduler
        self.log_step = max(1, round(self.n_iters / 10)) # we log 10 times per epoch

        self.train_metrics = MetricTracker('loss', writer=self.writer)
        self.valid_metrics = MetricTracker('loss', writer=self.writer)

        if disable_profilers:
            torch.autograd.set_detect_anomaly(False) # this was originally added for autocast, but it slows down training when true (x4)
            torch.autograd.profiler.profile(False)
            torch.autograd.profiler.emit_nvtx(False)


    def get_evaluation_model(self):
        return self.model if not self.ema_active else self.ema_model

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self, step):
        if step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def _phantom_train_epoch(self):
        # to make the data_loader behave in a reproducible way when training is resumed. It simulates an epoch-worth of data_loader iteration.
        for batch_idx, batch in enumerate(self.data_loader):
            if batch_idx == self.n_iters:
                break

    def _train_epoch(self, epoch): # recurrent not supported for DiffusionTrainer
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        times_validated = 0
        validation_steps = sorted([round(i * self.n_iters / self.validation_frequency) for i in range(1, self.validation_frequency + 1)]) # the last on
        log = {}

        self.train_metrics.reset()
        
        for batch_idx, batch in enumerate(inf_loop(self.data_loader)): # in case n_iters > len(data_loader) => break finishes loop
            self.model.train()
            obs, target, extra = batch
            obs, target = obs.to(self.device), target.to(self.device)
            mm_gt = extra["mm_gt"].to(self.device) if self.load_mmgt else None

            self.optimizer.zero_grad()

            t, weights = self.schedule_sampler.sample(obs.shape[0], self.device) # sample timesteps
            # args for model inference
            model_args = {
                "obs": obs, # for conditioning generation
            }
            # args for losses
            all_kwargs = {
                "obs": obs, # for conditioning generation
                "mm_gt": mm_gt if self.load_mmgt else None, # for multimodal GT --> raw coordinates that need to be embedded --> [batch_size, N_GT, ...]
            }
            losses = self.diffusion.training_losses(
                self.model,
                target,
                t, 
                model_kwargs=model_args,
                all_kwargs = all_kwargs
            )

            loss = (losses["loss"] * weights).mean()

            loss.backward()
            self.optimizer.step() # old without mixed precision

            self.writer.set_step((epoch - 1) * self.n_iters + batch_idx)

            for loss_name, loss_val in losses.items():
                if not self.train_metrics.contains(loss_name):
                    self.train_metrics.add_new(loss_name)
                self.train_metrics.update(loss_name, (loss_val * weights).mean().item())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if self.ema_active and batch_idx % self.update_ema_every == 0:
                step = (epoch - 1) * self.n_iters + batch_idx
                self.step_ema(step)

            # we run validation every X epochs (where X can be decimal)
            if self.do_validation and batch_idx + 1 in validation_steps:
                log = {}
                log.update(**{'val_'+k : v for k, v in list(self._valid_epoch(epoch, batch_idx).items())})
                times_validated += 1

            if batch_idx == self.n_iters:
                break

        if self.lr_scheduler is not None:
            self.lr_scheduler.step() # once per epoch

        if log is None:
            log = self.train_metrics.result()
        else:
            train_log = self.train_metrics.result()
            log.update(**{k : v for k, v in list(train_log.items())})

        return log

    def _valid_epoch(self, epoch, step):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :param step: Integer, training steps performed within that epoch
        :return: A log that contains information about validation
        """

        # TODO: test it works correctly
        self.model.eval()
        self.valid_metrics.reset()
        av_losses = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                obs, target, extra = batch
                obs, target = obs.to(self.device), target.to(self.device)
                mm_gt = extra["mm_gt"].to(self.device) if self.load_mmgt else None

                t, weights = self.schedule_sampler.sample(obs.shape[0], self.device) # sample timesteps
                # args for model inference
                model_args = {
                    "obs": obs, # for conditioning generation
                }
                # args for losses
                all_kwargs = {
                    "obs": obs, # for conditioning generation
                    "mm_gt": mm_gt if self.load_mmgt else None, # for multimodal GT --> raw coordinates that need to be embedded --> [batch_size, N_GT, ...]
                }

                losses = self.diffusion.training_losses(
                    self.model,
                    target,
                    t, 
                    model_kwargs=model_args,
                    all_kwargs = all_kwargs
                )

                loss = (losses["loss"] * weights).mean()

                for loss_name, loss_val in losses.items():
                    if loss_name not in av_losses:
                        av_losses[loss_name] = 0
                    av_losses[loss_name] += (loss_val * weights).mean().item() / len(self.valid_data_loader)

        self.writer.set_step((epoch - 1) * len(self.data_loader) + step, 'valid') # this way, train_loss and val_loss are sync

        for loss_name in av_losses:
            if not self.valid_metrics.contains(loss_name):
                self.valid_metrics.add_new(loss_name)
            self.valid_metrics.update(loss_name, av_losses[loss_name])
            
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if not self.epoch_based:#hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.batch_size
            total = self.n_samples
        else:
            current = batch_idx
            total = self.n_iters
        return base.format(current, total, 100.0 * current / total)

    def _delete_checkpoint(self, epoch, save_best=False):
        """
        Deleting checkpoints
        :param epoch: checkpoint epoch number to remove
        """
        super()._delete_checkpoint(epoch, save_best=save_best)

        if self.ema_active:
            # we additionally remove the EMA model
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}_ema.pth'.format(epoch))
            if os.path.exists(filename):
                os.remove(filename)

    def _save_checkpoint(self, epoch, save_best=False, val_metric=None):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        if self.ema_active:
            # we additionally store the EMA model
            arch = type(self.ema_model).__name__
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.ema_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'monitor_best': self.mnt_best,
                'monitor_best_stored': self.mnt_best_stored,
                'config': self.config
            }
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}_ema.pth'.format(epoch))
            torch.save(state, filename)
            if save_best:
                best_path = str(self.checkpoint_dir / 'model_best_ema.pth')
                torch.save(state, best_path)
                self.logger.info(f"Saving current best: model_best_ema.pth (mnt={val_metric})...")
        return super()._save_checkpoint(epoch, save_best=save_best, val_metric=val_metric)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        if self.ema_active:
            if "_ema" in resume_path:
                # this means that we are evaluating it => main model is the EMA one
                # we do this like this in order to avoid modifying the evaluation code
                # TODO improve when released
                ema_resume_path = resume_path
            else:
                ema_resume_path = str(resume_path).replace(".pth", "_ema.pth")

            self.logger.info("Loading EMA checkpoint: {} ...".format(ema_resume_path))
            checkpoint = torch.load(ema_resume_path)
            self.ema_model.load_state_dict(checkpoint['state_dict'])

        super()._resume_checkpoint(resume_path)

    def get_pose_generator(self):
        tracked_data_loader = self.valid_data_loader.get_tracked_sampler()
        for batch_idx, batch in enumerate(tracked_data_loader):
            obs_cpu, target_cpu, extra = batch
            idces = extra["sample_idx"]
            obs = obs_cpu.to(self.device)
            
            pred_length = target_cpu.shape[1]
            
            bs, obs_length, p, j, f = obs.shape
            
            #for i in range(self.num_samples_to_track):
            shape = (bs * self.num_samples_to_track, pred_length, p, j, f) # shape -> [N, Seq_length, Partic, Joints, Feat]
            
            model_args = {
                "obs": repeat(obs, 'b s p j f -> (repeat b) s p j f', repeat=self.num_samples_to_track) # for conditioning generation
            }
            predictions = self.diffusion.p_sample_loop(self.get_evaluation_model(), shape, progress=True, model_kwargs=model_args)["sample"].cpu().numpy()

            # gt
            all_x_gt = tracked_data_loader.dataset.recover_landmarks(obs_cpu, rrr=True, fill_root=True)
            all_y_gt = tracked_data_loader.dataset.recover_landmarks(target_cpu, rrr=True, fill_root=True)


            for sample_idx, sample_name in enumerate(idces):
                x_gt, y_gt = all_x_gt[sample_idx], all_y_gt[sample_idx]

                poses = {}

                for baseline in BASELINES:
                    poses[baseline] = BASELINES[baseline](x_gt, y_gt, 1)

                for s in range(self.num_samples_to_track):
                    y_pred = predictions[s * bs + sample_idx]
                    y_pred = tracked_data_loader.dataset.recover_landmarks(y_pred, rrr=True, fill_root=True)
                    poses[f"sample_{s}"] = np.concatenate([x_gt, y_pred], axis=0)

                yield poses, sample_name
