import torch
from trainer import Trainer
from utils import inf_loop
import os

class BehaviorDecouplerTrainer(Trainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, criterion_params, metrics, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, validation_frequency=1,
                 samples_epoch=None, seed=0, disable_profilers=True,
                 
                 train_aux=True,
                 auxiliary_model=None, auxiliary_criterion=None, auxiliary_criterion_params=None, 
                 auxiliary_metrics=None, auxiliary_optimizer=None, auxiliary_lr_scheduler=None
                 ):

        assert auxiliary_model is not None, "auxiliary_model is required"
        assert not train_aux or auxiliary_criterion is not None, "auxiliary_criterion is required"
        assert not train_aux or auxiliary_criterion_params is not None, "auxiliary_criterion_params is required"
        assert not train_aux or auxiliary_metrics is not None, "auxiliary_metrics is required"
        assert not train_aux or auxiliary_optimizer is not None, "auxiliary_optimizer is required"
        assert not train_aux or auxiliary_lr_scheduler is not None, "auxiliary_lr_scheduler is required"

        self.auxiliary_model = auxiliary_model
        self.auxiliary_criterion = auxiliary_criterion
        self.auxiliary_criterion_params = auxiliary_criterion_params
        self.auxiliary_metrics = auxiliary_metrics
        self.auxiliary_optimizer = auxiliary_optimizer
        self.auxiliary_lr_scheduler = auxiliary_lr_scheduler
        self.train_aux = train_aux

        super().__init__(model, criterion, criterion_params, metrics, optimizer, config, device,
                 data_loader, valid_data_loader=valid_data_loader, lr_scheduler=lr_scheduler, validation_frequency=validation_frequency,
                 samples_epoch=samples_epoch, seed=seed, disable_profilers=disable_profilers,
                 )

        


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
            #if batch_idx == self.n_iters:
            #    break
            #continue
            self.model.train()
            data, target = [b.to(self.device) for b in batch[:2]]

            self.optimizer.zero_grad()

            if self.train_aux:
                self.auxiliary_optimizer.zero_grad()
                # STEP 1 - Auxiliary reconstruction
                b = self.model.encode(data)
                output = self.auxiliary_model(b)

                data, target, output = [self.data_loader.dataset.recover_landmarks(d, rrr=False) for d in [data, target, output]]

                loss, losses_values, losses_names = self.auxiliary_criterion(data, target, output, **self.auxiliary_criterion_params)

                loss.backward()
                self.auxiliary_optimizer.step()

                # losses and metrics tracking
                self.writer.set_step((epoch - 1) * self.n_iters + batch_idx)
                for loss_name, loss_val in zip(losses_names, losses_values):
                    if not self.train_metrics.contains("aux_" + loss_name):
                        self.train_metrics.add_new("aux_" + loss_name)
                    self.train_metrics.update("aux_" + loss_name, loss_val)
                for met in self.metrics:
                    self.train_metrics.update("aux_" + met["alias"], met["fn"](target, output, **met["params"]).item())

            # STEP 2 - Main reconstruction
            output = self.model(data, target)
            b = output[1]
            aux_output = self.auxiliary_model(b)
            
            if not isinstance(output, tuple) and not isinstance(output, list):
                output = (output, ) # always a tuple where first element is the prediction that needs to match "target"

            output = list(output)
            data, target, output[0], aux_output = [self.data_loader.dataset.recover_landmarks(d, rrr=False) for d in [data, target, output[0], aux_output]]

            loss, losses_values, losses_names = self.criterion(data, target, output, aux_output, **self.criterion_params)

            loss.backward()
            self.optimizer.step()


            # losses and metrics tracking
            for loss_name, loss_val in zip(losses_names, losses_values):
                if not self.train_metrics.contains(loss_name):
                    self.train_metrics.add_new(loss_name)
                self.train_metrics.update(loss_name, loss_val)
            for met in self.metrics:
                self.train_metrics.update(met["alias"], met["fn"](target, output, **met["params"]).item())

            # debugging
            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            # validation every X epochs (where X can be decimal)
            if self.do_validation and batch_idx + 1 in validation_steps:
                log = {}
                log.update(**{'val_'+k : v for k, v in list(self._valid_epoch(epoch, batch_idx).items())})
                times_validated += 1

            # we break the epoch if we got to the end of training iterations
            if batch_idx == self.n_iters:
                break

        if self.lr_scheduler is not None:
            self.lr_scheduler.step() # once per epoch

        # log values stored for the epoch
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
        self.model.eval()
        self.valid_metrics.reset()
        av_loss = 0
        av_mets = {}
        for met in self.metrics:
            av_mets[met["alias"]] = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):
                data, target = [b.to(self.device) for b in batch[:2]]

                output = self.model(data, target)
                b = output[1]
                aux_output = self.auxiliary_model(b) # needed for loss computation
                
                if not isinstance(output, tuple) and not isinstance(output, list):
                    output = (output, ) # always a tuple where first element is the prediction that needs to match "target"

                output = list(output)
                data, target, output[0], aux_output = [self.data_loader.dataset.recover_landmarks(d, rrr=False) for d in [data, target, output[0], aux_output]]

                loss, losses_values, losses_names = self.criterion(data, target, output, aux_output, **self.criterion_params)
                av_loss += loss.item() / len(self.valid_data_loader)

                for met in self.metrics:
                    #self.valid_metrics.update(met.__name__, met(output, target))
                    av_mets[met["alias"]] += met["fn"](target, output, **met["params"]).item() / len(self.valid_data_loader)


        self.writer.set_step((epoch - 1) * len(self.data_loader) + step, 'valid') # this way, train_loss and val_loss are sync
        self.valid_metrics.update('loss', av_loss)  
        for met in self.metrics:
            self.valid_metrics.update(met["alias"], av_mets[met["alias"]])
            
        return self.valid_metrics.result()

    def _delete_checkpoint(self, epoch, save_best=False):
        """
        Deleting checkpoints
        :param epoch: checkpoint epoch number to remove
        """
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        if os.path.exists(filename):
            os.remove(filename)
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}_aux.pth'.format(epoch))
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
        aux_arch = type(self.auxiliary_model).__name__
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

        if self.train_aux:
            state_aux = {
                'arch': aux_arch,
                'epoch': epoch,
                'state_dict': self.auxiliary_model.state_dict(),
                'optimizer': self.auxiliary_optimizer.state_dict(),
                'monitor_best': self.mnt_best,
                'monitor_best_stored': self.mnt_best_stored,
                'config': self.config
            }
            filename_aux = str(self.checkpoint_dir / 'checkpoint-epoch{}_aux.pth'.format(epoch))
            torch.save(state_aux, filename_aux)

            self.logger.info("Saving checkpoint: {} ...".format(filename_aux))
            if save_best:
                best_path_aux = str(self.checkpoint_dir / 'model_best_aux.pth')
                torch.save(state_aux, best_path_aux)
        return filename

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        resume_path_aux = str(resume_path).replace(".pth", "_aux.pth")
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))

        checkpoint = torch.load(resume_path)
        checkpoint_aux = torch.load(resume_path_aux)

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])
        self.auxiliary_model.load_state_dict(checkpoint_aux['state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.mnt_best_stored = checkpoint['monitor_best_stored']

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.train_aux:
                self.auxiliary_optimizer.load_state_dict(checkpoint_aux['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

