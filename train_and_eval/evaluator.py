# ============================================
__author__ = "Sachin Mehta and Ximing Lu"
__maintainer__ = "Sachin Mehta and Ximing Lu"
# ============================================

import torch
from utilities.print_utilities import *
import os
from utilities.build_dataloader import get_test_data_loader
from utilities.build_model import build_model
from torch import nn
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from utilities.roc_utils import *
import json
from metrics.cmat_metrics import CMMetrics, CMResults
from data_loader.data_utils.bag_word_extractor import convert_image_to_words
from data_loader.data_utils.collate_function import get_bag_word_pairs
from utilities.save_dict_to_file import DictWriter
from train_and_eval.train_utils import prediction


class Evaluator(object):
    '''This class implements the evaluation code
    '''

    def __init__(self, opts):
        super(Evaluator, self).__init__()
        self.opts = opts
        # maximum batch size for CNN on single GPU
        self.max_bsz_cnn_gpu0 = opts.max_bsz_cnn_gpu0
        self.global_setter()

    def global_setter(self):
        self.setup_device()
        self.setup_dataloader()
        self.setup_model()

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

        test_loader, diag_classes, class_names = get_test_data_loader(opts=self.opts)

        self.test_loader = test_loader
        self.diag_classes = diag_classes
        self.class_names = class_names

    def setup_model(self):
        # Build Model
        odim = self.base_feature_extractor.module.output_feature_sz if self.use_multi_gpu else self.base_feature_extractor.output_feature_sz
        mi_model = build_model(opts=self.opts,
                               diag_classes=self.diag_classes,
                               base_feature_odim=odim
                               )

        if self.opts.weights_test:
            wts_loc = self.opts.weights_test
            if not os.path.isfile(wts_loc):
                print_error_message('No file exists here: {}'.format(wts_loc))

            print_log_message('Loading MiModel trained weights')
            pretrained_dict = torch.load(wts_loc, map_location=torch.device('cpu'))
            mi_model.load_state_dict(pretrained_dict)
            print_log_message('Loading over')
        else:
            print_error_message('--weights-test arguments needs to be supplied during inference')

        mi_model = mi_model.to(device=self.device)
        if self.use_multi_gpu:
            mi_model = torch.nn.DataParallel(mi_model)
        self.mi_model = mi_model
        self.mi_model.eval()

    def run(self, *args, **kwargs):
        self.mi_model.eval()

        if self.mi_model.training:
            print_warning_message('Model is in training mode. Moving to evaluation mode')
            self.mi_model.eval()

        with torch.no_grad():
            y_true = []
            # get multi-scale or single scale bag-word pairs
            bag_word_pairs = get_bag_word_pairs(bag_word_size=(self.opts.bag_size, self.opts.word_size),
                                                scale_factor=self.opts.scale_factor, scale_multipliers=self.opts.scale_multipliers) \
                                                if self.opts.msc_eval else [(self.opts.bag_size, self.opts.word_size)]

            predictions_dict = {}
            for batch_id, batch in tqdm(enumerate(self.test_loader)):
                image_name, true_diag_labels = batch
                if isinstance(image_name, (tuple, list)):
                    image_name = image_name[0]

                true_diag_labels = true_diag_labels.byte().cpu()
                y_true.append(true_diag_labels)

                for (bag_sz_sc, word_sz_sc) in bag_word_pairs:
                    assert bag_sz_sc % word_sz_sc == 0
                    num_bags_words_wh = bag_sz_sc // word_sz_sc
                    words = convert_image_to_words(image_name=image_name,
                                                   bag_width=bag_sz_sc,
                                                   bag_height=bag_sz_sc,
                                                   num_bags_h=num_bags_words_wh,
                                                   num_bags_w=num_bags_words_wh,
                                                   word_width=word_sz_sc,
                                                   word_height=word_sz_sc,
                                                   is_training=False,
                                                   return_orig_wsi_np=False
                                                   )
                    # add dummy batch dimension
                    words = words.unsqueeze(dim=0)

                    # prediction
                    model_pred = prediction(
                        words=words,
                        cnn_model=self.base_feature_extractor,
                        mi_model=self.mi_model,
                        max_bsz_cnn_gpu0=self.max_bsz_cnn_gpu0,
                        num_gpus=self.num_gpus,
                        device=self.device
                    )

                    scale_key = 'bag_{}_word_{}'.format(bag_sz_sc, word_sz_sc)
                    if scale_key not in predictions_dict:
                        predictions_dict[scale_key] = [model_pred.cpu()]
                    else:
                        predictions_dict[scale_key].append(model_pred.cpu())

            # Images x 1
            y_true = torch.cat(y_true, dim=0).numpy().tolist()

            # Different scale predictions
            scale_keys = predictions_dict.keys()
            preds_list_sc = []
            for key in scale_keys:
                # Images x C
                stacked = torch.cat(predictions_dict[key], dim=0)
                preds_list_sc.append(stacked)

            # Images x Classes X scales
            preds_diff_scales = torch.stack(preds_list_sc, dim=-1)

            # generating stats for each scale
            predictions_sc_sm = nn.Softmax(dim=-2)(preds_diff_scales)  # Images x Classes x scales
            pred_label_sc = torch.max(predictions_sc_sm, dim=-2)[1]  # Images x Scales
            pred_label_sc = pred_label_sc.byte().cpu().numpy()  # Images x Scales

            for sc, key in enumerate(scale_keys):
                preds = pred_label_sc[:, sc].tolist()  # Images x 1
                probs = predictions_sc_sm[:, :, sc].float().cpu().numpy()  # Images x Classes
                self.compute_stats(y_true=y_true, y_pred=preds, y_prob=probs, postfix=key)


            if self.opts.msc_eval:
                # Maximum of all scales
                predictions_max = torch.max(preds_diff_scales, dim=-1)[0]  # Image x Classes
                predictions_max_sm = nn.Softmax(dim=-1)(predictions_max)  # Image x Classes
                pred_label_max = torch.max(predictions_max_sm, dim=-1)[1]  # Image x 1
                pred_label_max = pred_label_max.byte().cpu().numpy().tolist()  # Image x 1
                pred_conf_max = predictions_max_sm.float().cpu().numpy()  # Image x Classes
                self.compute_stats(y_true=y_true, y_pred=pred_label_max, y_prob=pred_conf_max,
                                   postfix='bag_{}_word_{}_max'.format(self.opts.bag_size, self.opts.word_size))

                # Average of all scales
                predictions_avg = torch.mean(preds_diff_scales, dim=-1)  # Images x Classes
                predictions_avg_sm = nn.Softmax(dim=-1)(predictions_avg)  # Images x Classes
                pred_label_avg = torch.max(predictions_avg_sm, dim=-1)[1]  # Images x 1
                pred_label_avg = pred_label_avg.byte().cpu().numpy().tolist()  # Images x 1
                pred_conf_avg = predictions_avg_sm.float().cpu().numpy()  # Images x Classes

                self.compute_stats(y_true=y_true, y_pred=pred_label_avg, y_prob=pred_conf_avg,
                                   postfix='bag_{}_word_{}_avg'.format(self.opts.bag_size, self.opts.word_size))


    def compute_stats(self, y_true, y_pred, y_prob, postfix=''):
        split_name = self.opts.test_file.split(os.sep)[-1].replace('.txt', '') if os.path.isfile(
            self.opts.test_file) else 'dummy'

        save_loc = self.opts.savedir if os.path.isdir(self.opts.savedir) else './'
        cmat_file_name = 'mimodel_{}_{}_{}_{}_{}'.format('cmat',
                                                         self.opts.attn_fn,
                                                         self.opts.attn_type,
                                                         split_name,
                                                         postfix)

        results_file_name = 'mimodel_{}_{}_{}_{}_{}'.format('results',
                                                            self.opts.attn_fn,
                                                            self.opts.attn_type,
                                                            split_name, postfix)
        roc_file_name = 'mimodel_{}_{}_{}_{}_{}'.format('roc',
                                                        self.opts.attn_fn,
                                                        self.opts.attn_type,
                                                        split_name, postfix)

        results_summary = dict()
        # Add true labels and predictions
        results_summary['true_labels'] = [int(x) for x in y_true]
        results_summary['pred_labels'] = [int(x) for x in y_pred]

        cmat = confusion_matrix(y_true, y_pred)

        cmat_np_arr = np.array(cmat)
        results_summary['confusion_mat'] = cmat_np_arr.tolist()

        plot_confusion_matrix(
            cmat_array=cmat_np_arr,
            class_names=self.class_names,
            save_loc=save_loc,
            file_name=cmat_file_name
        )
        print_log_message('Confusion matrix plot is saved here: {}'.format(save_loc))

        # Compute results from confusion matrix
        conf_mat_eval = CMMetrics()
        cmat_results: CMResults = conf_mat_eval.compute_metrics(conf_mat=cmat_np_arr)

        cmat_results_dict = cmat_results._asdict()
        for key, values in cmat_results_dict.items():
                results_summary['{}'.format(key)] = values

        # plot the ROC curves
        print_log_message('Plotting ROC curves')
        y_true = np.array(y_true)
        y_true_oh = np.eye(self.diag_classes)[y_true]
        y_prob = np.array(y_prob)

        plot_roc(
            ground_truth=y_true_oh,
            pred_probs=y_prob,
            n_classes=self.diag_classes,
            class_names=self.class_names,
            dataset_name=self.opts.dataset,
            save_loc=save_loc,
            file_name=roc_file_name
        )
        print_log_message(
            'Done with plotting ROC curves. See here for plots: {}'.format(self.opts.savedir))

        # write the results to csv file
        writer = DictWriter(file_name=results_file_name, format='csv')
        writer.write(results_summary)

        # Also print results on the screen
        print(json.dumps(results_summary, indent=4, sort_keys=True))
