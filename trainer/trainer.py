import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from einops import repeat
from eval_belfusion import BASELINES

class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, criterion_params, metrics, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, validation_frequency=1,
                 samples_epoch=None, seed=0, disable_profilers=True):

        super().__init__(model, criterion, criterion_params, metrics, optimizer, config, seed)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size

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
            #self.data_loader = inf_loop(data_loader)
            self.n_iters = samples_epoch // self.batch_size
            self.n_samples = samples_epoch
            self.epoch_based = False

        self.lr_scheduler = lr_scheduler
        self.log_step = max(1, round(self.n_iters / 10)) # we log 10 times per epoch
        #self.log_step = int(np.sqrt(self.batch_size))

        self.train_metrics = MetricTracker('loss', *[m["alias"] for m in self.metrics], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m["alias"] for m in self.metrics], writer=self.writer)

        if disable_profilers:
            torch.autograd.set_detect_anomaly(False) # this was originally added for autocast, but it slows down training when true (x4)
            torch.autograd.profiler.profile(False)
            torch.autograd.profiler.emit_nvtx(False)

    def _phantom_train_epoch(self):
        # to make the data_loader behave in a reproducible way when training is resumed. It simulates an epoch-worth of data_loader iteration.
        for batch_idx, batch in enumerate(self.data_loader):
            if batch_idx == self.n_iters:
                break

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        times_validated = 0
        validation_steps = sorted([round(i * self.n_iters / self.validation_frequency) for i in range(1, self.validation_frequency + 1)]) # the last on
        log = None

        self.model.train()
        self.train_metrics.reset()
        
        for batch_idx, batch in enumerate(inf_loop(self.data_loader)): # in case n_iters > len(data_loader) => break finishes loop
            self.model.train()
            data, target, extra = batch
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            
            output = self.model(data, target)

            if not isinstance(output, tuple) and not isinstance(output, list):
                output = (output, ) # always a tuple where first element is the prediction that needs to match "target"

            output = list(output)
            data, target, output[0] = [self.data_loader.dataset.recover_landmarks(d, rrr=False) for d in [data, target, output[0]]]

            loss, losses_values, losses_names = self.criterion(data, target, output, **self.criterion_params)

            loss.backward()
            self.optimizer.step() # old without mixed precision

            self.writer.set_step((epoch - 1) * self.n_iters + batch_idx)
            #self.train_metrics.update('loss', loss.item()) # old
            for loss_name, loss_val in zip(losses_names, losses_values):
                if not self.train_metrics.contains(loss_name):
                    self.train_metrics.add_new(loss_name)
                self.train_metrics.update(loss_name, loss_val)
            for met in self.metrics:
                self.train_metrics.update(met["alias"], met["fn"](target, output, **met["params"]).item())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            # we run validation every X epochs (where X can be decimal)
            if self.do_validation and batch_idx + 1 in validation_steps:
                #self.logger.info(f"Valid Epoch: {epoch} (batch_idx={batch_idx})...")
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

    def _valid_epoch(self, epoch, step):#epoch_frequency, times_validated):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :param step: Integer, training steps performed within that epoch
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        av_loss = 0
        av_mets = {}
        for met in self.metrics:
            av_mets[met["alias"]] = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                data, target, extra = batch
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data, target)

                if not isinstance(output, tuple) and not isinstance(output, list):
                    output = (output, ) # always a tuple where first element is the prediction that needs to match "target"

                output = list(output)
                data, target, output[0] = [self.data_loader.dataset.recover_landmarks(d, rrr=False) for d in [data, target, output[0]]]
                
                loss, losses_values, losses_names = self.criterion(data, target, output, **self.criterion_params)
                av_loss += loss.item() / len(self.valid_data_loader)

                #self.writer.set_step((epoch - 1) * len(self.valid_data_loader) * epoch_frequency + len(self.valid_data_loader) * times_validated + batch_idx, 'valid')
                #self.valid_metrics.update('loss', loss.item())
                #TODO add losses too?
                for met in self.metrics:
                    #self.valid_metrics.update(met.__name__, met(output, target))
                    av_mets[met["alias"]] += met["fn"](target, output, **met["params"]).item() / len(self.valid_data_loader)
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True)) # It crashes I don't know why

        self.writer.set_step((epoch - 1) * len(self.data_loader) + step, 'valid') # this way, train_loss and val_loss are sync
        self.valid_metrics.update('loss', av_loss)  
        for met in self.metrics:
            self.valid_metrics.update(met["alias"], av_mets[met["alias"]])
            
        # add histogram of model parameters to the tensorboard
        #for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins='auto')
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


    def get_pose_generator(self):
        tracked_data_loader = self.valid_data_loader.get_tracked_sampler()
        for batch_idx, batch in enumerate(tracked_data_loader):
            obs_cpu, target_cpu, extra = batch
            idces = extra["sample_idx"]
            obs = obs_cpu.to(self.device)
            
            pred_length = target_cpu.shape[1]
            
            bs, obs_length, p, j, f = obs.shape

            obs_repeated = repeat(obs, 'b s p j f -> (repeat b) s p j f', repeat=self.num_samples_to_track)
            if hasattr(self.model, "sample_prior") and callable(getattr(self.model, "sample_prior")): # stochastic methods
                predictions = self.model.sample_prior(obs_repeated)
            else: # deterministic methods
                predictions = self.model(obs_repeated, None)

            if isinstance(predictions, tuple) or isinstance(predictions, list):
                predictions = predictions[0] # always a tuple where first element is the prediction that needs to match "target"
            predictions = predictions.cpu().numpy()

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