# ============================================
__author__ = "Sachin Mehta and Ximing Lu"
__maintainer__ = "Sachin Mehta and Ximing Lu"
# ============================================

import torch
from utilities.print_utilities import *
import os
from utilities.lr_scheduler import get_lr_scheduler
from metrics.metric_utils import accuracy
from metrics.statistics import Statistics
import gc
from utilities.utils import save_checkpoint, load_checkpoint, save_arguments
from utilities.build_dataloader import get_data_loader
from utilities.build_model import build_model
from utilities.build_optimizer import build_optimizer, update_optimizer, read_lr_from_optimzier
from utilities.build_criteria import build_criteria
import numpy as np
import math
import json
from utilities.save_dict_to_file import DictWriter
from train_and_eval.train_utils import prediction


class Trainer(object):
    '''This class implemetns the training and validation functionality for training ML model for medical imaging'''

    def __init__(self, opts):
        super(Trainer, self).__init__()
        self.opts = opts
        self.best_acc = 0
        self.start_epoch = 0
        # maximum batch size for CNN on single GPU
        self.max_bsz_cnn_gpu0 = opts.max_bsz_cnn_gpu0

        self.resume = self.opts.checkpoint if self.opts.checkpoint is not None and os.path.isdir(
            self.opts.checkpoint) else None

        self.global_setter()

    def global_setter(self):
        self.setup_device()
        self.setup_directories()
        self.setup_logger()
        self.setup_lr_scheduler()
        self.setup_dataloader()
        self.setup_model_optimizer_lossfn()

    def setup_directories(self):
        if not os.path.isdir(self.opts.savedir):
            os.makedirs(self.opts.savedir)

    def setup_device(self):
        num_gpus = torch.cuda.device_count()
        self.num_gpus = num_gpus
        if num_gpus > 0:
            print_log_message('Using {} GPUs'.format(num_gpus))
        else:
            print_log_message('Using CPU')

        self.device = torch.device("cuda:0" if num_gpus > 0 else "cpu")
        self.use_multi_gpu = True if num_gpus > 1 else False

        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            cudnn.deterministic = True

    def setup_logger(self):
        # Let's visualize logs on tensorboard. It's awesome
        try:
            from torch.utils.tensorboard import SummaryWriter
        except:
            from utilities.summary_writer import SummaryWriter

        self.logger = SummaryWriter(log_dir=self.opts.savedir, comment='Training and Validation logs')

    def setup_lr_scheduler(self):
        # fetch learning rate scheduler
        self.lr_scheduler = get_lr_scheduler(self.opts)

    def setup_dataloader(self):
        from model.base_feature_extractor import BaseFeatureExtractor
        base_feature_extractor = BaseFeatureExtractor(opts=self.opts)
        base_feature_extractor = base_feature_extractor.to(device=self.device)
        # We do not want the base extractor to train, so setting it to eval mode
        if self.use_multi_gpu:
            base_feature_extractor = torch.nn.DataParallel(base_feature_extractor)
        self.base_feature_extractor = base_feature_extractor
        self.base_feature_extractor.eval()

        # sanity check
        if self.base_feature_extractor.training:
            print_warning_message('Base feature extractor is in training mode. Moving to evaluation mode')
            self.base_feature_extractor.eval()

        train_loader, val_loader, diag_classes, class_weights = get_data_loader(opts=self.opts)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.diag_classes = diag_classes
        self.class_weights = torch.from_numpy(class_weights)

    def setup_model_optimizer_lossfn(self):
        # Build Model
        odim = self.base_feature_extractor.module.output_feature_sz if self.use_multi_gpu else self.base_feature_extractor.output_feature_sz
        mi_model = build_model(opts=self.opts,
                               diag_classes=self.diag_classes,
                               base_feature_odim=odim
                               )

        if self.resume is not None:
            resume_ep, resume_model_state, resume_optim_state, resume_perf = load_checkpoint(
                checkpoint_dir=self.opts.checkpoint,
                device=self.device)
            self.start_epoch = resume_ep
            self.best_acc = resume_perf
            self.mi_model.load_state_dict(resume_model_state)
            self.optimizer.load_state_dict(resume_optim_state)

            # move optimizer state to the device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device=self.device)

            print_log_message('Resuming from checkpoint saved at {}th epoch'.format(self.start_epoch))

        mi_model = mi_model.to(device=self.device)
        if self.use_multi_gpu:
            mi_model = torch.nn.DataParallel(mi_model)
        self.mi_model = mi_model

        # Build Loss function
        criteria = build_criteria(opts=self.opts, class_weights=self.class_weights.float())
        self.criteria = criteria.to(device=self.device)

        # Build optimizer
        self.optimizer = build_optimizer(model=self.mi_model, opts=self.opts)

    def training(self, epoch, lr, *args, **kwargs):
        train_stats = Statistics()

        self.mi_model.train()

        num_samples = len(self.train_loader)
        epoch_start_time = time.time()

        for batch_id, batch in enumerate(self.train_loader):
            words, true_diag_labels = batch
            true_diag_labels = true_diag_labels.to(device=self.device)

            # prediction
            pred_diag_labels = prediction(
                words=words,
                cnn_model=self.base_feature_extractor,
                mi_model=self.mi_model,
                max_bsz_cnn_gpu0=self.max_bsz_cnn_gpu0,
                num_gpus=self.num_gpus,
                device=self.device
            )

            # compute loss
            loss = self.criteria(pred_diag_labels, true_diag_labels)

            # compute metrics
            top1_acc = accuracy(pred_diag_labels, true_diag_labels, topk=(1,))

            loss.backward()
            # Gradient accumulation is useful, when batch size is very small say 1
            # Gradients will be accumulated for accum_count iterations
            # After accum_count iterations, weights are updated and graph is freed.
            if (batch_id + 1) % self.opts.accum_count == 0 or batch_id + 1 == len(self.train_loader):
                self.optimizer.step()
                self.optimizer.zero_grad()

            train_stats.update(loss=loss.item(), acc=top1_acc[0].item())

            if batch_id % self.opts.log_interval == 0 and batch_id > 0:  # print after every 100 batches
                train_stats.output(epoch=epoch, batch=batch_id, n_batches=num_samples, start=epoch_start_time, lr=lr)

        return train_stats.avg_acc(), train_stats.avg_loss()

    def warm_up(self, *args, **kwargs):
        self.mi_model.train()

        num_samples = len(self.train_loader)

        warm_up_iterations = int(math.ceil((self.opts.warm_up_iterations * 1.0) / num_samples) * num_samples)

        print_info_message('Warming Up')
        print_log_message(
            'LR will linearly change from {} to {} in about {} steps'.format(self.opts.warm_up_min_lr, self.opts.lr,
                                                                             warm_up_iterations))
        lr_list = np.linspace(1e-7, self.opts.lr, warm_up_iterations)
        epoch_start_time = time.time()
        iteration = -1
        while iteration < warm_up_iterations:
            warm_up_stats = Statistics()
            for batch_id, batch in enumerate(self.train_loader):
                if iteration >= warm_up_iterations:
                    break

                iteration += 1
                try:
                    lr_iter = lr_list[iteration]
                except:
                    # fall back to final LR after warm-up step if iteration is outsize lr_list range
                    lr_iter = self.opts.lr

                # update learning rate at every iteration
                self.optimizer = update_optimizer(optimizer=self.optimizer, lr_value=lr_iter)

                words, true_diag_labels = batch
                true_diag_labels = true_diag_labels.to(device=self.device)

                # prediction
                pred_diag_labels = prediction(
                    words=words,
                    cnn_model=self.base_feature_extractor,
                    mi_model=self.mi_model,
                    max_bsz_cnn_gpu0=self.max_bsz_cnn_gpu0,
                    num_gpus=self.num_gpus,
                    device=self.device
                )

                # compute loss
                loss = self.criteria(pred_diag_labels, true_diag_labels)

                # compute metrics
                top1_acc = accuracy(pred_diag_labels, true_diag_labels, topk=(1,))

                loss.backward()
                # Gradient accumulation is useful, when batch size is very small say 1
                # Gradients will be accumulated for accum_count iterations
                # After accum_count iterations, weights are updated and graph is freed.
                if (batch_id + 1) % self.opts.accum_count == 0 or batch_id + 1 == len(self.train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                warm_up_stats.update(loss=loss.item(), acc=top1_acc[0].item())

                if batch_id % self.opts.log_interval == 0 and batch_id > 0:  # print after every 100 batches
                    warm_up_stats.output(epoch=-1, batch=iteration, n_batches=warm_up_iterations,
                                         start=epoch_start_time,
                                         lr=lr_iter)

            gc.collect()

        print_log_message('Warming Up... Done!!!')

    def validation(self, epoch, lr, *args, **kwargs):
        val_stats = Statistics()

        self.mi_model.eval()
        num_samples = len(self.val_loader)

        with torch.no_grad():
            epoch_start_time = time.time()
            for batch_id, batch in enumerate(self.val_loader):

                # bags, bag_hist_arr, words, word_hist_arr, true_diag_labels = batch
                words, true_diag_labels = batch
                true_diag_labels = true_diag_labels.to(device=self.device)

                # prediction
                pred_diag_labels = prediction(
                    words=words,
                    cnn_model=self.base_feature_extractor,
                    mi_model=self.mi_model,
                    max_bsz_cnn_gpu0=self.max_bsz_cnn_gpu0,
                    num_gpus=self.num_gpus,
                    device=self.device
                )

                # compute loss
                loss = self.criteria(pred_diag_labels, true_diag_labels)

                # compute metrics
                top1_acc = accuracy(pred_diag_labels, true_diag_labels, topk=(1,))

                val_stats.update(loss=loss.item(), acc=top1_acc[0].item())

                if batch_id % self.opts.log_interval == 0 and batch_id > 0:  # print after every 100 batches
                    val_stats.output(epoch=epoch, batch=batch_id, n_batches=num_samples, start=epoch_start_time, lr=lr)

        gc.collect()
        avg_acc = val_stats.avg_acc()
        avg_loss = val_stats.avg_loss()

        print_log_message('* Validation Stats')
        print_log_message('* Loss: {:5.2f}, Mean Acc: {:3.2f}'.format(avg_loss, avg_acc))

        return avg_acc, avg_loss

    def run(self, *args, **kwargs):
        kwargs['need_attn'] = False

        if self.opts.warm_up:
            self.warm_up(args=args, kwargs=kwargs)

        if self.resume is not None:
            # find the LR value
            for epoch in range(self.start_epoch):
                self.lr_scheduler.step(epoch)

        eval_stats_dict = dict()

        for epoch in range(self.start_epoch, self.opts.epochs):
            epoch_lr = self.lr_scheduler.step(epoch)

            self.optimizer = update_optimizer(optimizer=self.optimizer, lr_value=epoch_lr)

            # Uncomment this line if you want to check the optimizer's LR is updated correctly
            # assert read_lr_from_optimzier(self.optimizer) == epoch_lr

            train_acc, train_loss = self.training(epoch=epoch, lr=epoch_lr, args=args, kwargs=kwargs)
            val_acc, val_loss = self.validation(epoch=epoch, lr=epoch_lr, args=args, kwargs=kwargs)
            eval_stats_dict[epoch] = val_acc
            gc.collect()

            # remember best accuracy and save checkpoint for best model
            is_best = val_acc >= self.best_acc
            self.best_acc = max(val_acc, self.best_acc)

            model_state = self.mi_model.module.state_dict() if isinstance(self.mi_model, torch.nn.DataParallel) \
                else self.mi_model.state_dict()

            optimizer_state = self.optimizer.state_dict()

            save_checkpoint(epoch=epoch,
                            model_state=model_state,
                            optimizer_state=optimizer_state,
                            best_perf=self.best_acc,
                            save_dir=self.opts.savedir,
                            is_best=is_best,
                            keep_best_k_models=self.opts.keep_best_k_models
                            )

            self.logger.add_scalar('LR', round(epoch_lr, 6), epoch)
            self.logger.add_scalar('TrainingLoss', train_loss, epoch)
            self.logger.add_scalar('TrainingAcc', train_acc, epoch)

            self.logger.add_scalar('ValidationLoss', val_loss, epoch)
            self.logger.add_scalar('ValidationAcc', val_acc, epoch)

        # dump the validation epoch id and accuracy data, so that it could be used for filtering later on
        eval_stats_dict_sort = {k: v for k, v in sorted(eval_stats_dict.items(),
                                                        key=lambda item: item[1],
                                                        reverse=True
                                                        )}

        eval_stats_fname = '{}/val_stats_bag_{}_word_{}_{}_{}'.format(
            self.opts.savedir,
            self.opts.bag_size,
            self.opts.word_size,
            self.opts.attn_fn,
            self.opts.attn_type,
        )

        writer = DictWriter(file_name=eval_stats_fname, format='json')
        # if json file does not exist
        if not os.path.isfile(eval_stats_fname):
            writer.write(data_dict=eval_stats_dict_sort)
        else:
            with open(eval_stats_fname, 'r') as json_file:
                eval_stats_dict_old = json.load(json_file)
            eval_stats_dict_old.update(eval_stats_dict_sort)

            eval_stats_dict_updated = {k: v for k, v in sorted(eval_stats_dict_old.items(),
                                                               key=lambda item: item[1],
                                                               reverse=True
                                                               )}
            writer.write(data_dict=eval_stats_dict_updated)

        self.logger.close()