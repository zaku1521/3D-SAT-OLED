import os
import joblib
import json
import numpy as np
import pandas as pd
from collections import OrderedDict
from addict import Dict
from tqdm import tqdm
import logging
from logging import Formatter

from sklearn.model_selection import GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader

from draw import draw_oled, init_oled_draw
from SAT_OLED_Model.utils import Metrics
from SAT_OLED_Model.data import DataHub
from SAT_OLED_Model.models.oled import SAT_OLED_Model, swiglu
from SAT_OLED_Model.utils import YamlHandler
from SAT_OLED_Model.models.loss import GHMC_Loss, FocalLossWithLogits, myCrossEntropyLoss

DIR_PATH = 'SAT_OLED_Model/oled/'
print(os.path.abspath(DIR_PATH))

np.set_printoptions(precision=3, suppress=False)

logger_initialized = {}
logger_with_file_initialized = {}

Formatter.default_msec_format = '%s.%03d'


def record_loss_log(namespace='OLED(QSAR)-loss', log_file=None, log_level=logging.INFO, file_mode='a'):
    logger = logging.getLogger(namespace)
    if namespace in logger_initialized:
        if log_file:
            if not logger.handlers or log_file != getattr(logger.handlers[-1], 'baseFilename', None):
                logger.manager.loggerDict.clear()
                logger_initialized.pop(namespace)
                logger_with_file_initialized.pop(namespace)

                logger = logging.getLogger(namespace)
            else:
                return logger
        else:
            return logger

    logger.setLevel(logging.DEBUG)

    c_formatter = logging.Formatter("%(asctime)s | %(lineno)s | %(levelname)s | %(name)s | %(message)s",
                                    "%Y-%m-%d %H:%M:%S")
    c_handler = logging.StreamHandler()
    c_handler.setLevel(log_level)
    c_handler.setFormatter(c_formatter)
    logger.addHandler(c_handler)

    if log_file:
        f_formatter = logging.Formatter("%(asctime)s | %(lineno)s | %(levelname)s | %(name)s | %(message)s",
                                        "%Y-%m-%d %H:%M:%S")
        f_handler = logging.FileHandler(log_file, encoding='utf-8', mode=file_mode)
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(f_formatter)
        logger.addHandler(f_handler)

        logger_with_file_initialized[namespace] = True

    logger_initialized[namespace] = True

    return logger


def get_root_logger(namespace='OLED(QSAR)-oled', log_file=None, log_level=logging.INFO, file_mode='a'):
    """
    log.info(msg) or higher will print to console and file
    log.debug(msg) will only print to file
    """
    logger = logging.getLogger(namespace)
    # if logger.hasHandlers():
    #     logger.handlers.clear()
    if namespace in logger_initialized:
        if log_file:
            if not logger.handlers or log_file != getattr(logger.handlers[-1], 'baseFilename', None):
                logger.manager.loggerDict.clear()
                logger_initialized.pop(namespace)
                logger_with_file_initialized.pop(namespace)

                logger = logging.getLogger(namespace)
            else:
                return logger
        else:
            return logger

    logger.setLevel(logging.DEBUG)

    c_formatter = logging.Formatter("%(asctime)s | %(lineno)s | %(levelname)s | %(name)s | %(message)s",
                                    "%Y-%m-%d %H:%M:%S")
    c_handler = logging.StreamHandler()
    c_handler.setLevel(log_level)
    c_handler.setFormatter(c_formatter)
    logger.addHandler(c_handler)

    if log_file:
        f_formatter = logging.Formatter("%(asctime)s | %(lineno)s | %(levelname)s | %(name)s | %(message)s",
                                        "%Y-%m-%d %H:%M:%S")
        f_handler = logging.FileHandler(log_file, encoding='utf-8', mode=file_mode)
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(f_formatter)
        logger.addHandler(f_handler)

        logger_with_file_initialized[namespace] = True

    logger_initialized[namespace] = True

    return logger


class R2Loss(torch.nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()

    def forward(self, y_pred, y_true):
        sse = torch.sum(torch.square(y_true - y_pred), dim=0)
        sst = torch.sum(torch.square(y_true - torch.mean(y_true, dim=0)), dim=0)
        loss = (1 - sse / sst).sum() - 1
        return - loss


class WR2Loss(torch.nn.Module):
    def __init__(self):
        super(WR2Loss, self).__init__()

    def forward(self, y_pred, y_true):
        W = torch.tensor([0.4, 0.2, 0.2, 0.2]).reshape((4, 1)).to(y_true.device)
        sse = torch.sum(torch.square(y_true - y_pred), dim=0)
        sst = torch.sum(torch.square(y_true - torch.mean(y_true, dim=0)), dim=0)
        loss = torch.einsum("ij,ji->i", (1 - sse / sst).reshape(1, 4), W) - 1
        return -loss


class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, prediction, target):
        weights = torch.tensor(self.weights).cuda()
        squared_error = (prediction - target) ** 2
        weighted_error = squared_error * weights
        loss = torch.mean(weighted_error)
        return loss


NNMODEL_REGISTER = {
    'SAT_OLED_Model': SAT_OLED_Model
}

LOSS_RREGISTER = {
    'classification': myCrossEntropyLoss,
    'multiclass': myCrossEntropyLoss,
    'regression': nn.MSELoss(),
    'multilabel_classification': {
        'default': FocalLossWithLogits,
        'bce': nn.BCEWithLogitsLoss(),
        'ghm': GHMC_Loss(bins=10, alpha=0.5),
        'focal': FocalLossWithLogits,
    },
    'multilabel_regression': {
        'default': nn.MSELoss(),
        'bce': nn.BCEWithLogitsLoss(),
        'MSELoss': nn.MSELoss(),
        'WR2Loss': WR2Loss(),
        'R2Loss': R2Loss(),
        'WeightedMSELoss': WeightedMSELoss(weights=[2.0, 1.4, 1.0, 1.0]),
    }
}
ACTIVATION_FN = {
    'classification': lambda x: F.softmax(x, dim=-1)[:, 1:],
    'multiclass': lambda x: F.softmax(x, dim=-1),
    'regression': lambda x: x,
    'multilabel_classification': lambda x: F.sigmoid(x),
    'multilabel_regression': {
        'default': lambda x: x,
        'sigmoid': lambda x: F.sigmoid(x),
        'swiglu': swiglu
    }
}
OUTPUT_DIM = {
    'classification': 2,
    'regression': 1,
}

oled_overall_metrics_dict = {}
oled_per_target_metrics_dict = {}


def convert_tuple_keys(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                str_key = "_".join(map(str, k))
                new_dict[str_key] = convert_tuple_keys(v)
            else:
                new_dict[k] = convert_tuple_keys(v)
        return new_dict
    elif isinstance(obj, list):
        return [convert_tuple_keys(item) for item in obj]
    else:
        return convert_numpy_types(obj)


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    else:
        return obj


class NNModel(object):
    def __init__(self, data, trainer, **params):
        self.data = data
        self.num_classes = self.data['num_classes']
        self.target_scaler = self.data['target_scaler']
        self.features = data['input']
        self.model_name = params.get('model_name', 'SAT_OLED_Model')
        self.data_type = params.get('data_type', 'molecule')
        self.loss_key = params.get('loss_key', 'default')
        self.activate_key = params.get('activate_key', 'default')
        self.trainer: Trainer = trainer
        self.splitter = self.trainer.splitter
        self.model_params = params.copy()
        self.task = params['task']
        if self.task in OUTPUT_DIM:
            self.model_params['output_dim'] = OUTPUT_DIM[self.task]
        elif self.task == 'multiclass':
            self.model_params['output_dim'] = self.data['multiclass_cnt']
        else:  # multilabel_regression
            self.model_params['output_dim'] = self.num_classes
        self.model_params['device'] = self.trainer.device
        self.cv = dict()
        self.metrics = self.trainer.metrics
        if isinstance(LOSS_RREGISTER[self.task], dict):
            self.loss_func = LOSS_RREGISTER[self.task][self.loss_key]  # MSELoss()
        else:
            self.loss_func = LOSS_RREGISTER[self.task]

        if isinstance(ACTIVATION_FN[self.task], dict):
            self.activation_fn = ACTIVATION_FN[self.task][self.activate_key]
        else:
            self.activation_fn = ACTIVATION_FN[self.task]
        self.save_path = self.trainer.save_path
        self.trainer.set_seed(self.trainer.seed)
        self.model = self._init_model(**self.model_params)

    def _init_model(self, model_name, **params):
        model: SAT_OLED_Model = NNMODEL_REGISTER[model_name](**params)
        if params['data_type'] == 'molecule':
            model.classification_head.dropout = nn.Dropout(p=0)
        else:
            if params['drop_out'] > 0:
                model.classification_head.dropout = nn.Dropout(p=params['drop_out'])
        return model

    def collect_data(self, X, y, idx):
        assert isinstance(y, np.ndarray), 'y must be numpy array'
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X[idx]).float(), torch.from_numpy(y[idx])
        elif isinstance(X, list):
            return {k: v[idx] for k, v in X.items()}, torch.from_numpy(y[idx])
        else:
            raise ValueError('X must be numpy array or dict')

    def run(self):
        get_root_logger().info("start test OLED QM BY 3D-SAT-OLED-Model:{}".format(self.model_name))
        X = np.asarray(self.features)
        y = np.asarray(self.data['target'])

        val_indices = list(range(len(X)))
        X_valid, y_valid = X[val_indices], y[val_indices]
        validdataset = NNDataset(X_valid, y_valid)

        HARTREE_TO_EV = 27.2114
        TARGETS_TO_CONVERT = {1, 2, 3}
        TARGET_NAMES = [
            'plqy',
            'e_ad',
            'homo',
            'lumo'
        ]
        TARGET_UNITS = {
            0: '-',
            1: 'eV',
            2: 'eV',
            3: 'eV'
        }

        _y_pred = self.trainer.fit_predict(self.model, None, validdataset, self.loss_func,
                                           self.activation_fn, self.save_path, None, self.target_scaler)

        label_cnt = self.data['multiclass_cnt'] if 'multiclass_cnt' in self.data else None

        y_valid_original = self.data['target_scaler'].inverse_transform(y_valid)
        y_pred_original = self.data['target_scaler'].inverse_transform(_y_pred)

        y_valid_converted = y_valid_original.copy()
        y_pred_converted = y_pred_original.copy()
        for target_idx in TARGETS_TO_CONVERT:
            if target_idx < y_valid_original.shape[1]:
                y_valid_converted[:, target_idx] *= HARTREE_TO_EV
                y_pred_converted[:, target_idx] *= HARTREE_TO_EV

        overall_metrics_original = self.metrics.cal_metric(
            y_valid_original,
            y_pred_original,
            label_cnt=label_cnt
        )
        get_root_logger().info(f"Overall result (unit: Hartree): {overall_metrics_original}")
        oled_overall_metrics_dict['Hartree'] = overall_metrics_original

        overall_metrics_converted = self.metrics.cal_metric(
            y_valid_converted,
            y_pred_converted,
            label_cnt=label_cnt
        )
        get_root_logger().info(f"Overall result (unit: eV): {overall_metrics_converted}")
        oled_overall_metrics_dict['eV'] = overall_metrics_converted

        num_targets = y_valid.shape[1]
        for i in range(num_targets):
            target_name = TARGET_NAMES[i] if i < len(TARGET_NAMES) else f"target_{i}"
            target_unit = TARGET_UNITS.get(i, 'unknown')

            per_target_metric_hartree = self.metrics.cal_metric(
                y_valid_original[:, i].reshape(-1, 1),
                y_pred_original[:, i].reshape(-1, 1),
                label_cnt=label_cnt
            )
            original_unit = '-' if i == 0 else 'Hartree'
            oled_per_target_metrics_dict[(target_name, original_unit)] = per_target_metric_hartree
            get_root_logger().info(
                f"Result for target {i} ({target_name}) (unit: {original_unit}): {per_target_metric_hartree}")

            per_target_metric_ev = self.metrics.cal_metric(
                y_valid_converted[:, i].reshape(-1, 1),
                y_pred_converted[:, i].reshape(-1, 1),
                label_cnt=label_cnt
            )

            oled_per_target_metrics_dict[(target_name, target_unit)] = per_target_metric_ev
            get_root_logger().info(
                f"Result for target {i} ({target_name}) (unit: {target_unit}): {per_target_metric_ev}")

        oled_data = {
            'overall': oled_overall_metrics_dict,
            'per_target': oled_per_target_metrics_dict
        }

        oled_data_converted = convert_tuple_keys(oled_data)
        y_valid_split = [y_valid_converted[:, i].reshape(-1, 1) for i in range(num_targets)]
        y_pred_split = [y_pred_converted[:, i].reshape(-1, 1) for i in range(num_targets)]

        return oled_data_converted, y_valid_split, y_pred_split


def dump(self, data, dir, name):
    path = os.path.join(dir, name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    joblib.dump(data, path)


def count_parameters(self, model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def NNDataset(data, label=None, aug_level=0):
    return TorchDataset(data, label, aug_level)


class TorchDataset(Dataset):
    def __init__(self, data, label=None, aug_level=0):
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))
        self.aug_level = aug_level

    def __getitem__(self, idx):
        if self.aug_level == 0:
            pass
        else:
            col, row = self.data[idx]['src_distance'].shape
            distance_ = np.random.randn(col, row) * np.random.uniform(0.0005, 0.001) * self.aug_level
            distance_ = np.triu(distance_)
            distance_ += distance_.T - 2 * np.diag(distance_.diagonal())
            distance_ = np.clip(distance_, 0, None)
            self.data[idx]['src_distance'] += distance_
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


class Splitter(object):
    def __init__(self, split_method='5fold_random', seed=42):
        self.split_method = split_method
        self.seed = seed
        self.splitter = self._init_split(self.split_method, self.seed)
        self.n_splits = 5
        self.skf = None

    def _init_split(self, split_method, seed=42):
        if split_method == '5fold_random':
            splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
        elif split_method == '5fold_scaffold':
            splitter = GroupKFold(n_splits=5)
        elif split_method == 'fixed_index':
            splitter = None
        else:
            raise ValueError('Unknown splitter method: {}'.format(split_method))

        return splitter

    def split(self, data, target=None, group=None):
        if self.split_method in ['5fold_random']:
            self.skf = self.splitter.split(data)
        elif self.split_method in ['5fold_scaffold']:
            self.skf = self.splitter.split(data, target, group)
        elif self.split_method == 'fixed_index':
            n_samples = len(data)
            train_indices = list(range(900))
            val_indices = list(range(900, n_samples))
            self.skf = [(train_indices, val_indices)]
        else:
            raise ValueError('Unknown splitter method: {}'.format(self.split_method))
        return self.skf


class Trainer(object):
    def __init__(self, save_path=None, **params):
        self.save_path = save_path
        self.task = params.get('task', None)
        self.cfg_params = params

        if self.task != 'repr':  # multilabel_regression !=repr
            self.metrics_str = params['metrics']
            self.metrics = Metrics(self.task, self.metrics_str)
        self._init_trainer(**params)

    def _init_trainer(self, **params):
        ### init common params ###
        self.split_method = params.get('split_method', '5fold_random')
        self.split_seed = params.get('split_seed', 42)
        self.seed = params.get('seed', 42)
        self.set_seed(self.seed)
        self.splitter = Splitter(self.split_method, self.split_seed)
        self.logger_level = int(params.get('logger_level', 1))
        ### init NN trainer params ###
        self.learning_rate = float(params.get('learning_rate', 1e-4))
        self.batch_size = params.get('batch_size', 32)
        self.max_epochs = params.get('epochs', 50)
        self.warmup_ratio = params.get('warmup_ratio', 0.1)
        self.patience = params.get('patience', 10)
        self.max_norm = params.get('max_norm', 1.0)
        self.cuda = params.get('cuda', False)
        self.amp = params.get('amp', False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' and self.amp == True else None

    def decorate_batch(self, batch, feature_name=None):
        return self.decorate_torch_batch(batch)

    def decorate_graph_batch(self, batch):
        net_input, net_target = {'net_input': batch.to(
            self.device)}, batch.y.to(self.device)
        if self.task in ['classification', 'multiclass', 'multilabel_classification']:
            net_target = net_target.long()
        else:
            net_target = net_target.float()
        return net_input, net_target

    def decorate_torch_batch(self, batch):
        """function used to decorate batch data
        """
        net_input, net_target = batch
        if isinstance(net_input, dict):
            net_input, net_target = {
                                        k: v.to(self.device) for k, v in net_input.items()}, net_target.to(self.device)
        else:
            net_input, net_target = {'net_input': net_input.to(
                self.device)}, net_target.to(self.device)
        if self.task == 'repr':
            net_target = None
        elif self.task in ['classification', 'multiclass', 'multilabel_classification']:
            net_target = net_target.long()
        else:
            net_target = net_target.float()
        return net_input, net_target

    def fit_predict(self, model, train_dataset, valid_dataset, loss_func, activation_fn, dump_dir, fold, target_scaler,
                    feature_name=None):
        model = model.to(self.device)
        y_preds, _, _ = self.predict(model, valid_dataset, loss_func, activation_fn,
                                     dump_dir, fold, target_scaler, 1, load_model=True, feature_name=feature_name)
        return y_preds

    def _early_stop_choice(self, wait, loss, min_loss, metric_score, max_score, model, dump_dir, fold, patience, epoch):
        ### hpyerparameter need to tune if you want to use early stop, currently find use loss is suitable in benchmark test. ###
        if not isinstance(self.metrics_str, str) or self.metrics_str in ['loss', 'none', '']:
            is_early_stop, min_val_loss, wait = self._judge_early_stop_loss(
                wait, loss, min_loss, model, dump_dir, fold, patience, epoch)
        else:
            is_early_stop, min_val_loss, wait, max_score = self.metrics._early_stop_choice(
                wait, min_loss, metric_score, max_score, model, dump_dir, fold, patience, epoch)
        return is_early_stop, min_val_loss, wait, max_score

    def _judge_early_stop_loss(self, wait, loss, min_loss, model, dump_dir, fold, patience, epoch):
        is_early_stop = False
        if loss <= min_loss:
            min_loss = loss
            wait = 0
            info = {'model_state_dict': model.state_dict()}
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(info, os.path.join(dump_dir, f'model_{fold}.pth'))
        elif loss >= min_loss:
            wait += 1
            if wait == self.patience:
                get_root_logger().warning(f'Early stopping at epoch: {epoch + 1}')
                is_early_stop = True
        return is_early_stop, min_loss, wait

    def predict(self, model, dataset, loss_func, activation_fn, dump_dir, fold, target_scaler=None, epoch=1,
                load_model=False, feature_name=None):
        model = model.to(self.device)
        if load_model == True:
            load_model_path = 'SAT_OLED_Model/weights/model_oled.pth'
            model_dict = torch.load(load_model_path, map_location=self.device)[
                "model_state_dict"]
            model.load_state_dict(model_dict, strict=False)
            get_root_logger().info("load model success!")
        dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=model.batch_collate_fn,
        )
        model = model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
                         position=0, leave=False, desc='test', ncols=5)
        val_loss = []
        y_preds = []
        y_truths = []
        for i, batch in enumerate(dataloader):
            net_input, net_target = self.decorate_batch(batch, feature_name)
            # Get model outputs
            with torch.no_grad():
                outputs = model(**net_input)
                if not load_model:
                    loss = loss_func(outputs, net_target)
                    val_loss.append(float(loss.data))
            y_preds.append(activation_fn(outputs).cpu().numpy())
            y_truths.append(net_target.detach().cpu().numpy())
            if not load_model:
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch + 1, self.max_epochs),
                    loss="{:.04f}".format(float(np.sum(val_loss) / (i + 1))))

            batch_bar.update()
        y_preds = np.concatenate(y_preds)
        y_truths = np.concatenate(y_truths)

        try:
            label_cnt = model.output_dim
        except:
            label_cnt = None

        if target_scaler is not None:
            inverse_y_preds = target_scaler.inverse_transform(y_preds)
            inverse_y_truths = target_scaler.inverse_transform(y_truths)
            metric_score = self.metrics.cal_metric(
                inverse_y_truths, inverse_y_preds, label_cnt=label_cnt) if not load_model else None
        else:
            metric_score = self.metrics.cal_metric(
                y_truths, y_preds, label_cnt=label_cnt) if not load_model else None
        batch_bar.close()
        return y_preds, val_loss, metric_score

    def set_seed(self, seed):
        """function used to set a random seed
        Arguments:
            seed {int} -- seed number, will set to torch and numpy
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)


def NNDataLoader(feature_name=None, dataset=None, batch_size=None, shuffle=False, collate_fn=None, drop_last=False):
    dataloader = TorchDataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=drop_last)
    return dataloader


class MolTrain(object):
    def __init__(self,
                 task='classification',
                 data_type='molecule',
                 epochs=10,
                 learning_rate=1e-4,
                 batch_size=16,
                 patience=5,
                 metrics="none",
                 split='random',
                 save_path='./exp',
                 remove_hs=False,
                 **kwargs
                 ):
        config = {
            # data
            'target_col_prefix': "TARGET",
            'target_normalize': "auto",
            'anomaly_clean': True,
            'model_name': "SAT_OLED_Model",
            # trainer
            'split_method': "5fold_random",
            'split_seed': 42,
            'seed': 42,
            'logger_level': 1,
            'patience': 10,
            'max_epochs': 100,
            'learning_rate': 1e-4,
            'warmup_ratio': 0.03,
            'batch_size': 16,
            'max_norm': 5.0,
            'cuda': True,
            'amp': True,
            **kwargs
        }
        self.yamlhandler = YamlHandler('/')
        config = Dict(config)
        config.task = task
        config.data_type = data_type
        config.epochs = epochs
        config.learning_rate = learning_rate
        config.batch_size = batch_size
        config.patience = patience
        config.metrics = metrics
        config.split = split
        config.remove_hs = remove_hs
        self.save_path = save_path
        self.config = config

    def fit(self, data):
        self.datahub = DataHub(data=data, is_train=True,
                               save_path=self.save_path, **self.config)
        self.data = self.datahub.data
        self.update_and_save_config()
        self.trainer = Trainer(save_path=self.save_path, **self.config)
        self.model = NNModel(self.data, self.trainer, **self.config)
        get_root_logger().info('Model: \n' + repr(self.model.model))
        oled_metric, y_valid, y_pred = self.model.run()
        return oled_metric, y_valid, y_pred

    def update_and_save_config(self):
        self.config['num_classes'] = self.data['num_classes']
        self.config['target_cols'] = ','.join(self.data['target_cols'])
        if self.config['task'] == 'multiclass':  # 'multilabel_regression'
            self.config['multiclass_cnt'] = self.data['multiclass_cnt']

        if self.config['split'] == 'random':  # random
            self.config['split'] = 'random_5fold'
        else:
            self.config['split'] = 'scaffold_5fold'

        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                get_root_logger().info('Create output directory: {}'.format(self.save_path))
                os.makedirs(self.save_path)
            else:
                get_root_logger().info('Output directory already exists: {}'.format(self.save_path))
                get_root_logger().info('Warning: Overwrite output directory: {}'.format(self.save_path))
            out_path = os.path.join(self.save_path, 'config.yaml')
            self.yamlhandler.write_yaml(
                data=self.config, out_file_path=out_path)
        return


class NormalizeTarget():
    def __init__(self, target_mean_std=None):
        self.target_mean_std = target_mean_std

    def normalize(self, data):
        if self.target_mean_std is None:
            return data

        for key in self.target_mean_std:
            mean = self.target_mean_std[key]['mean']
            std = self.target_mean_std[key]['std']
            data[key] = (data[key] - mean) / std
        return data

    def invert(self, data, invert_rescale=1.0):
        if self.target_mean_std is None:
            return data

        for key in self.target_mean_std:
            mean = self.target_mean_std[key]['mean']
            std = self.target_mean_std[key]['std']
            data[key] = data[key] * std * invert_rescale + mean
        return data


def collect_data(
        target=['plqy', 'e_ad', 'homo', 'lumo'],
        normlize=True,
        repeat=1
):
    data = np.load(os.path.join(DIR_PATH, 'test.npz'), allow_pickle=True)

    data_target = dict(map(lambda x: (x, data[x]), target))
    data_target = pd.DataFrame(data_target)
    if normlize:
        normlizer = NormalizeTarget(data_target.describe().iloc[1:3])
        data_target = normlizer.normalize(data_target)
    else:
        normlizer = NormalizeTarget()
    data_target = data_target.values.tolist()
    data_coords = [np.array(item) for item in data['coord']]
    data_atoms = [list(item) for item in data['symbol']]
    if repeat > 1:
        data_coords = data_coords * int(repeat)
        data_atoms = data_atoms * int(repeat)
        data_target = data_target * int(repeat)

    data = {
        'target': data_target,
        'coordinates': data_coords,
        'atoms': data_atoms
    }

    return data


def main(cfg):
    target_columns = cfg.pop('target_columns')
    exp_suffix = '_'.join(target_columns + list(map(str, cfg.values())))
    exp_suffix = 'oled_test' + '_' + exp_suffix
    os.makedirs(f'./exp_{exp_suffix}', exist_ok=True)

    logger = get_root_logger(log_file=f'./exp_{exp_suffix}/root.log')
    data = collect_data(
        target=target_columns,
        normlize=False,
        repeat=cfg['repeat']
    )
    clf = MolTrain(**cfg,
                   split='random',
                   remove_hs=True,
                   save_path=f'./exp_{exp_suffix}')
    logger.info('exp_suffix: ' + exp_suffix)
    logger.info('Config: \n' + json.dumps(clf.config, indent=4))
    logger.info(data['target'][0])
    logger.info(data['coordinates'][0])
    logger.info(data['atoms'][0])

    oled_metrics, y_valid, y_pred = clf.fit(data)

    metrics_path = os.path.join('./oled_metrics', 'oled_metrics.json')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(oled_metrics, f, ensure_ascii=False, indent=4)
        logger.info(f'Saved OLED QM DataSet metrics to {metrics_path} ')

    logger.info(f'Start drawing OLED QM dataset metrics, saving to: oled_metrics')
    init_oled_draw()
    draw_oled(y_valid, y_pred)
    return


if __name__ == '__main__':
    config = OrderedDict(
        target_columns=['plqy', 'e_ad', 'homo', 'lumo'],  # 'plqy', 'e_ad', 'homo', 'lumo'
        model_name='SAT_OLED_Model',
        data_type='molecule',
        task='multilabel_regression',
        epochs=40,
        learning_rate=0.0002,
        batch_size=8,
        patience=10,
        metrics='r2',
        loss_key='default',
        drop_out=0.5,
        activate_key='default',
        target_normalize='auto',
        pre_norm=True,
        repeat=1,
        lr_type='linear',
        optim_type='AdamW',
        seed=42,
        split_seed=42,
    )
    main(config)
