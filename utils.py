import time, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import operator
from itertools import product
from functools import reduce, partial
from data_loaders import load_list
from timm.models.layers import trunc_normal_
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################## Helper Funcs ##########################
def infer_labels(loader, k = 10):
    from sklearn.cluster import k_means, MiniBatchKMeans
    
    X, Y = loader.dataset.tensors[0].cpu(), loader.dataset.tensors[1].cpu().numpy()
    try:
        Z = loader.dataset.tensors[2].cpu()
    except:
        Z = None

    Y = Y.reshape(len(Y), -1)

    if len(Y) <= 10000:
        labeling_fun = lambda Y: torch.LongTensor(k_means(Y, k)[1])
        Y = labeling_fun(Y).unsqueeze(1)
    else:
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10000).fit(Y)
        Y = torch.LongTensor(kmeans.predict(Y)).unsqueeze(1)

    if Z is None:
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True), k
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y, Z), batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True), k


def load_by_class(loader, num_classes):
    train_set = loader.dataset
    subsets = {}

    if len(train_set.__getitem__(0)) == 3:
        try:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y == target]) for target in range(num_classes)}
        except:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}
    else:
        try:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y == target]) for target in range(num_classes)}
        except:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}
    loaders = {target: torch.utils.data.DataLoader(subset, batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True) for target, subset in subsets.items()}
    class_weights = {target: len(subset)/len(train_set) for target, subset in subsets.items()}
    
    return loaders, class_weights


def create_position_ids_from_inputs_embeds(inputs_embeds, padding_idx=1):
    input_shape = inputs_embeds.size()[:-1]
    sequence_length = input_shape[1]

    position_ids = torch.arange(padding_idx + 1, sequence_length + padding_idx + 1, dtype=torch.long, device=inputs_embeds.device)
    return position_ids.unsqueeze(0).expand(input_shape)


def scale(Z, numpy=False):
    if numpy:
        factor = np.trace(Z.transpose() @ Z)
        return Z / np.sqrt(factor)
    
    factor = torch.trace(Z.transpose(0, 1) @ Z)
    return Z / torch.sqrt(factor)


def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)

########################## Metrics ##########################

def coding_rate(Z, eps=1e-4, numpy=False):
    n, d = Z.shape 
    if numpy:
        _, rate = np.linalg.slogdet((np.eye(d) + 1 / (n * eps) * Z.transpose() @ Z))
    else:
        _, rate = torch.linalg.slogdet((torch.eye(d).to(Z.device) + 1 / (n * eps) * Z.transpose(0, 1) @ Z))
    return .5 * rate

def transrate_one_class(Z, eps=1e-4):
    if len(Z.shape) > 2:
        Z = Z.mean(1)
    Z = scale(Z)
    Z = Z - Z.mean(dim=0, keepdim=True)
    RZ = coding_rate(Z, eps)
    return RZ

def transrate(Z, y, eps=1e-4):
    if len(Z.shape) > 2:
        Z = Z.mean(1)
    y = y.reshape(len(y), 1)
    Z = scale(Z, numpy=True)
    Z = Z - np.mean(Z, axis=0, keepdims=True)
    RZ = coding_rate(Z, eps, numpy=True)
    RZY = 0
    K = int(y.max() + 1)
    num_invalid = 0
    for i in range(K):
        if (y == i).flatten().sum() == 0:
            num_invalid += 1
            continue
        RZY += coding_rate(Z[(y == i).flatten()], eps, numpy=True)
    r = RZ - RZY / (K - num_invalid)
    return r


def otdd(feats, ys=None, src_train_dataset=None, exact=True):
    if feats.shape[0] == 1: 
        feats = torch.cat([feats] * 10, 0)
    if len(feats.shape) > 2:
        feats = feats.mean(1)
    ys = torch.zeros(len(feats)) if ys is None else ys

    if not torch.is_tensor(feats):
        feats = torch.from_numpy(feats).to('cpu')
        ys = torch.from_numpy(ys).long().to('cpu')

    dataset = torch.utils.data.TensorDataset(feats, ys)
    dist = DatasetDistance(src_train_dataset, dataset,
                                    inner_ot_method = 'exact' if exact else 'gaussian_approx',
                                    debiased_loss = True, inner_ot_debiased=True,
                                    p = 2, inner_ot_p=2, entreg = 1e-1, ignore_target_labels = False,
                                    device=feats.device, load_prev_dyy1=None)
                
    d = dist.distance(maxsamples = len(src_train_dataset)) * 1e-2
    return d


def l2(feats, src_train_dataset=None):
    if len(feats.shape) > 2:
        feats = feats.mean(1)
    if not torch.is_tensor(feats):
        feats = torch.from_numpy(feats).to('cpu')
    pdist = nn.PairwiseDistance(p=2)
    src_train_loader = torch.utils.data.DataLoader(src_train_dataset, batch_size=len(feats), shuffle=True, num_workers=4, pin_memory=True)

    d = 0
    for i, data in enumerate(src_train_loader):
        x_, y_ = data 
        x_ = x_.to(feats.device)
        if len(feats) == len(x_):
            d += pdist(feats, x_).mean()
        elif i == 0:
            d += pdist(feats[:len(x_)], x_).mean()

    return d / (i + 1)


class MMD_loss(nn.Module):
    def __init__(self, src_data, maxsamples, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

        self.src_data = src_data
        self.src_data_len = len(src_data)
        self.maxsamples = maxsamples 

    def guassian_kernel(self, source, target):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if self.fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def guassian_kernel_numpy(self, source, target):
        
        n_samples = int(source.shape[0])+int(target.shape[0])
        total = np.concatenate([source, target], 0)

        total0 = np.broadcast_to(np.expand_dims(total, 0), (int(total.shape[0]), int(total.shape[0]), int(total.shape[1])))
        total1 = np.broadcast_to(np.expand_dims(total, 1), (int(total.shape[0]), int(total.shape[0]), int(total.shape[1])))
        L2_distance = ((total0-total1)**2).sum(2) 
        if self.fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = np.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]
        kernel_val = [np.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, target):
        if len(target.shape) > 2:
            target = target.mean(1)

        target_len = len(target)
        indices = np.random.choice(self.src_data_len, size=target_len)
        if torch.is_tensor(target):
            source = self.src_data[indices].to(target.device)
            kernels = self.guassian_kernel(source, target)
        else:
            source = self.src_data[indices]
            kernels = self.guassian_kernel_numpy(source, target)

        XX = kernels[:target_len, :target_len]
        YY = kernels[target_len:, target_len:]
        XY = kernels[:target_len, target_len:]
        YX = kernels[target_len:, :target_len] 

        loss = torch.mean(XX + YY - XY - YX) if torch.is_tensor(target) else np.mean(XX + YY - XY - YX)
        return loss

def set_grad_state(module, state):
    for n, m in module.named_modules():
        if len(n) == 0: continue
        if not state and 'position' in n: continue
        if not state and 'tunable' in n: continue
        for param in m.parameters():
            param.requires_grad = state


"""Customized Task Metrics"""

def fnr(output, target):
    metric = maskMetric(output.squeeze().detach().cpu().numpy() > 0, target.squeeze().cpu().numpy()) 
    TP, TN, FP, FN = metric[0], metric[1], metric[2], metric[3]
    TPR = TP / (TP + FN)
    return 1-TPR


def tpr(output, target):
    metric = maskMetric(output.squeeze().detach().cpu().numpy() > 0, target.squeeze().cpu().numpy())
    TP, TN, FP, FN = metric[0], metric[1], metric[2], metric[3]
    TPR = TP / (TP + FN)
    return TPR


class psicov_mae(object):
    def __init__(self, root):
        self.pdb_list = load_list(root + '/protein/psicov.lst')
        self.length_dict = {}
        for pdb in self.pdb_list:
            (ly, seqy, cb_map) = np.load(root + '/protein/psicov/distance/' + pdb + '-cb.npy', allow_pickle = True)
            self.length_dict[pdb] = ly

    def __call__(self, output, target):
        if len(output.shape) == 3:
            output = output.unsqueeze(1)
        targets = []
        for i in range(len(target)):
            targets.append(np.expand_dims(target[i].cpu().numpy().transpose(1, 2, 0), axis=0))
        P = output.cpu().detach().numpy().transpose(0, 2, 3, 1)

        LMAX, pad_size = 512, 10

        Y = np.full((len(targets), LMAX, LMAX, 1), np.nan)
        for i, xy in enumerate(targets):
            Y[i, :, :, 0] = xy[0, :, :, 0]
        # Average the predictions from both triangles
        for j in range(0, len(P[0, :, 0, 0])):
            for k in range(j, len(P[0, :, 0, 0])):
                P[ :, j, k, :] = (P[ :, k, j, :] + P[ :, j, k, :]) / 2.0
        P[ P < 0.01 ] = 0.01
        # Remove padding, i.e. shift up and left by int(pad_size/2)
        P[:, :LMAX-pad_size, :LMAX-pad_size, :] = P[:, int(pad_size/2) : LMAX-int(pad_size/2), int(pad_size/2) : LMAX-int(pad_size/2), :]
        Y[:, :LMAX-pad_size, :LMAX-pad_size, :] = Y[:, int(pad_size/2) : LMAX-int(pad_size/2), int(pad_size/2) : LMAX-int(pad_size/2), :]

        PRED = P
        YTRUE = Y

        mae_lr_d8_list = np.zeros(len(PRED[:, 0, 0, 0]))
        mae_mlr_d8_list = np.zeros(len(PRED[:, 0, 0, 0]))
        mae_lr_d12_list = np.zeros(len(PRED[:, 0, 0, 0]))
        mae_mlr_d12_list = np.zeros(len(PRED[:, 0, 0, 0]))
        for i in range(0, len(PRED[:, 0, 0, 0])):
            L = self.length_dict[self.pdb_list[i]]
            PAVG = np.full((L, L), 100.0)
            # Average the predictions from both triangles
            for j in range(0, L):
                for k in range(j, L):
                    PAVG[j, k] = (PRED[i, k, j, 0] + PRED[i, j, k, 0]) / 2.0
            # at distance 8 and separation 24
            Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
            P = np.copy(PAVG)
            for p in range(len(Y)):
                for q in range(len(Y)):
                    if q - p < 24:
                        Y[p, q] = np.nan
                        P[p, q] = np.nan
                        continue
                    if Y[p, q] > 8:
                        Y[p, q] = np.nan
                        P[p, q] = np.nan
            mae_lr_d8 = np.nan
            if not np.isnan(np.abs(Y - P)).all():
                mae_lr_d8 = np.nanmean(np.abs(Y - P))
                #mae_lr_d8 = np.sqrt(np.nanmean(np.abs(Y - P) ** 2))
            # at distance 8 and separation 12
            Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
            P = np.copy(PAVG)
            for p in range(len(Y)):
                for q in range(len(Y)):
                    if q - p < 12:
                        Y[p, q] = np.nan
                        P[p, q] = np.nan
                        continue
                    if Y[p, q] > 8:
                        Y[p, q] = np.nan
                        P[p, q] = np.nan
            mae_mlr_d8 = np.nan
            if not np.isnan(np.abs(Y - P)).all():
                mae_mlr_d8 = np.nanmean(np.abs(Y - P))
            # at distance 12 and separation 24
            Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
            P = np.copy(PAVG)
            for p in range(len(Y)):
                for q in range(len(Y)):
                    if q - p < 24:
                        Y[p, q] = np.nan
                        P[p, q] = np.nan
                        continue
                    if Y[p, q] > 12:
                        Y[p, q] = np.nan
                        P[p, q] = np.nan
            mae_lr_d12 = np.nan
            if not np.isnan(np.abs(Y - P)).all():
                mae_lr_d12 = np.nanmean(np.abs(Y - P))
            # at distance 12 and separation 12
            Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
            P = np.copy(PAVG)
            for p in range(len(Y)):
                for q in range(len(Y)):
                    if q - p < 12:
                        Y[p, q] = np.nan
                        P[p, q] = np.nan
                        continue
                    if Y[p, q] > 12:
                        Y[p, q] = np.nan
                        P[p, q] = np.nan
            mae_mlr_d12 = np.nan
            if not np.isnan(np.abs(Y - P)).all():
                mae_mlr_d12 = np.nanmean(np.abs(Y - P))
            # add to list
            mae_lr_d8_list[i] = mae_lr_d8
            mae_mlr_d8_list[i] = mae_mlr_d8
            mae_lr_d12_list[i] = mae_lr_d12
            mae_mlr_d12_list[i] = mae_mlr_d12

        mae = np.nanmean(mae_lr_d8_list)
        
        if np.isnan(mae):
            return np.inf
        return mae


def maskMetric(PD, GT):
    if len(PD.shape) == 2:
        PD = PD.reshape(1, *PD.shape)
    if len(GT.shape) == 2:
        GT = GT.reshape(1, *GT.shape)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(GT.shape[0]):
        P = GT[i].sum()
        TP += (PD[i][GT[i] == 1] == 1).sum()
        TN += (PD[i][GT[i] == 0] == 0).sum()
        FP += (PD[i][GT[i] == 0] == 1).sum()
        FN += (PD[i][GT[i] == 1] == 0).sum()
    return np.array([TP, TN, FP, FN])


def ecg_f1(output, target):
    target = target.cpu().detach().numpy()
    output = np.argmax(output.cpu().detach().numpy(), axis=1)
    tmp_report = metrics.classification_report(target, output, output_dict=True, zero_division=0)
    f1_score = []
    for i, (y, scores) in enumerate(tmp_report.items()): 
        if y == '0' or y == '1' or y == '2' or y == '3':
            f1_score.append(tmp_report[y]['f1-score'])
    f1_score = np.mean(f1_score)
    return f1_score


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    res = res[0] if len(res) == 1 else res
    return res


def accuracy_onehot(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    target = torch.argmax(target, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    res = res[0] if len(res) == 1 else res
    return res


def map_value(output, target):
    val_preds = torch.sigmoid(output).float().cpu().detach().numpy()
    val_gts = target.cpu().detach().numpy()
    map_value = metrics.average_precision_score(val_gts, val_preds, average="macro")
    return map_value


def inv_auroc(output, target):
    output = output.reshape(len(output), -1)
    target = target.reshape(len(target), -1)
    output = torch.sigmoid(output).float()
    result = output.cpu().detach().numpy()

    y = target.cpu().detach().numpy()
    result_shape = np.shape(result)

    fpr_list, tpr_list, auroc_list = [], [], []
    precision_list, recall_list, aupr_list = [], [], []
    for i in range(result_shape[1]):
        fpr_temp, tpr_temp, auroc_temp  = calculate_auroc(result[:, i], y[:, i])
        precision_temp, recall_temp, aupr_temp = calculate_aupr(result[:, i], y[:, i])

        fpr_list.append(fpr_temp)
        tpr_list.append(tpr_temp)
        precision_list.append(precision_temp)
        recall_list.append(recall_temp)
        auroc_list.append(auroc_temp)
        aupr_list.append(aupr_temp)

    avg_auroc = np.nanmean(auroc_list)
    avg_aupr = np.nanmean(aupr_list)
    return 1 - avg_auroc


def auroc(output, target):
    output = torch.sigmoid(output).float()
    result = output.cpu().detach().numpy()

    y = target.cpu().detach().numpy()
    result_shape = np.shape(result)

    fpr_list, tpr_list, auroc_list = [], [], []
    precision_list, recall_list, aupr_list = [], [], []
    for i in range(result_shape[1]):
        fpr_temp, tpr_temp, auroc_temp  = calculate_auroc(result[:, i], y[:, i])
        precision_temp, recall_temp, aupr_temp = calculate_aupr(result[:, i], y[:, i])

        fpr_list.append(fpr_temp)
        tpr_list.append(tpr_temp)
        precision_list.append(precision_temp)
        recall_list.append(recall_temp)
        auroc_list.append(auroc_temp)
        aupr_list.append(aupr_temp)

    avg_auroc = np.nanmean(auroc_list)
    avg_aupr = np.nanmean(aupr_list)
    return avg_auroc


def calculate_auroc(predictions, labels):
    fpr_list, tpr_list, threshold_list = metrics.roc_curve(y_true=labels, y_score=predictions)
    score = metrics.auc(fpr_list, tpr_list)
    return fpr_list, tpr_list, score


def calculate_stats(output, target, class_indices=None):
    classes_num = target.shape[-1]
    if class_indices is None:
        class_indices = range(classes_num)
    stats = []

    for k in class_indices:
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)
        dict = {'AP': avg_precision}
        stats.append(dict)

    return stats


def calculate_aupr(predictions, labels):
    precision_list, recall_list, threshold_list = metrics.precision_recall_curve(y_true=labels, probas_pred=predictions)
    aupr = metrics.auc(recall_list, precision_list)
    return precision_list, recall_list, aupr


"""Customized Task Losses"""

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean' if self.size_average else 'sum')
        if torch.cuda.is_available():
             self.criterion =  self.criterion.cuda()

    def forward(self, output, target):
        target = torch.eye(18)[target].to(device)
        model_out = F.softmax(output, dim = 1) + 1e-9

        ce = torch.multiply(target, -torch.log(model_out))
        weight = torch.multiply(target, (1 - model_out) ** self.gamma)
        fl = self.alpha * torch.multiply(weight, ce)
        reduced_fl = torch.sum(fl, axis=1)
        return reduced_fl.mean()


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.cuda = cuda
        self.criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='mean')


    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            
            if self.cuda:
                self.criterion = self.criterion.cuda()
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        
        loss = self.criterion(logit, target.long())
        
        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n = logit.size()[0]
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


def logCoshLoss(y_t, y_prime_t, reduction='mean', eps=1e-12):
    if reduction == 'mean':
        reduce_fn = torch.mean
    elif reduction == 'sum':
        reduce_fn = torch.sum
    else:
        reduce_fn = lambda x: x
    x = y_prime_t - y_t
    return reduce_fn(torch.log((torch.exp(x) + torch.exp(-x)) / 2))


#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


"""Hepler Funcs"""

def get_params_to_update(model, finetune_method):
    print("Params to learn:")

    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
    
    return params_to_update


class embedder_placeholder(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x=None, inputs_embeds=None, *args, **kwargs):
        if x is not None:
            return x

        return inputs_embeds

import torch.nn.init as init

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        if m.bias is not None:
            init.constant_(m.bias, 0)


def embedder_init(source, target, train_embedder=False, match_stats=False):
    print(train_embedder)
    if train_embedder:
        if hasattr(source, 'patch_embeddings'):
            if match_stats:
                weight_mean, weight_std = source.patch_embeddings.projection.weight.mean(), source.patch_embeddings.projection.weight.std()
                nn.init.normal_(target.projection.weight, weight_mean, weight_std)
                
                bias_mean, bias_std = source.patch_embeddings.projection.bias.mean(), source.patch_embeddings.projection.bias.std()
                nn.init.normal_(target.projection.bias, bias_mean, bias_std)
            else:
                rep_num = target.projection.in_channels // source.patch_embeddings.projection.in_channels + 1
                rep_weight = torch.cat([source.patch_embeddings.projection.weight.data] * rep_num, 1)
                
                target.projection.weight.data.copy_(rep_weight[:, :target.projection.in_channels, ...])        
                target.projection.bias.data.copy_(source.patch_embeddings.projection.bias.data)
                print("warm")

            target.norm.weight.data.copy_(source.norm.weight.data)
            target.norm.bias.data.copy_(source.norm.bias.data)

        else:
            target.norm.weight.data.copy_(source.LayerNorm.weight.data)
            target.norm.bias.data.copy_(source.LayerNorm.bias.data)
     
            target.position_embeddings = copy.deepcopy(source.position_embeddings)
            print("warm")

    else:
        for n, m in target.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)  

        try:
            target.position_embeddings = copy.deepcopy(source.position_embeddings)
        except:
            pass

def count_params(model):
    c = 0
    for p in model.parameters():
        try:
            c += reduce(operator.mul, list(p.size()))
        except:
            pass

    return c

def count_trainable_params(model):
    c = 0
    for p in model.parameters():
        try:
            if p.requires_grad:
                c += reduce(operator.mul, list(p.size()))
        except:
            pass

    return c

def print_grad(model, kernel_choices, dilation_choices):
    param_values, ks, ds = [], [], []
    for name, param in model.named_arch_params():
        param_values.append(param.data.argmax(0))
        print(name, param.data)
        ks.append(kernel_choices[int(param_values[-1] // len(dilation_choices))])
        ds.append(dilation_choices[int(param_values[-1] % len(dilation_choices))])

    print("Kernel pattern (name, k, d):", ks, np.sum(ks), ds, np.sum(ds))

def mask(img, ignore):
    return img * (1 - ignore)


try:
    import os
    import tqdm
    import glob
    import numpy as np
    import librosa
    import torch
    import torchaudio
    import json
    import random
    import soundfile as sf
    import pandas as pd
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    from typing import Tuple, Optional
    from scipy import stats
    from sklearn import metrics

    '''
    Metrics
    '''
    def d_prime(auc):
        standard_normal = stats.norm()
        d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
        return d_prime


    def calculate_stats(output, target, class_indices=None):
        """Calculate statistics including mAP, AUC, etc.

        Args:
          output: 2d array, (samples_num, classes_num)
          target: 2d array, (samples_num, classes_num)
          class_indices: list
            explicit indices of classes to calculate statistics for

        Returns:
          stats: list of statistic of each class.
        """

        classes_num = target.shape[-1]
        if class_indices is None:
            class_indices = range(classes_num)
        stats = []

        # Class-wise statistics
        for k in class_indices:
            # Average precision
            avg_precision = metrics.average_precision_score(
                target[:, k], output[:, k], average=None)

            # AUC
            auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

            # Precisions, recalls
            (precisions, recalls, thresholds) = metrics.precision_recall_curve(
                target[:, k], output[:, k])

            # FPR, TPR
            (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

            save_every_steps = 1000     # Sample statistics to reduce size
            dict = {'precisions': precisions[0::save_every_steps],
                    'recalls': recalls[0::save_every_steps],
                    'AP': avg_precision,
                    'fpr': fpr[0::save_every_steps],
                    'fnr': 1. - tpr[0::save_every_steps],
                    'auc': auc}
            stats.append(dict)

        return stats

    '''
    Audio Mixer
    '''
    def get_random_sample(dataset):
        rnd_idx = random.randint(0, len(dataset) - 1)
        rnd_image, _, rnd_target = dataset.__get_item_helper__(rnd_idx)
        return rnd_image, rnd_target


    class BackgroundAddMixer:
        def __init__(self, alpha_dist='uniform'):
            assert alpha_dist in ['uniform', 'beta']
            self.alpha_dist = alpha_dist

        def sample_alpha(self):
            if self.alpha_dist == 'uniform':
                return random.uniform(0, 0.5)
            elif self.alpha_dist == 'beta':
                return np.random.beta(0.4, 0.4)

        def __call__(self, dataset, image, target):
            rnd_idx = random.randint(0, dataset.get_bg_len() - 1)
            rnd_image = dataset.get_bg_feature(rnd_idx)

            alpha = self.sample_alpha()
            image = (1 - alpha) * image + alpha * rnd_image
            return image, target


    class AddMixer:
        def __init__(self, alpha_dist='uniform'):
            assert alpha_dist in ['uniform', 'beta']
            self.alpha_dist = alpha_dist

        def sample_alpha(self):
            if self.alpha_dist == 'uniform':
                return random.uniform(0, 0.5)
            elif self.alpha_dist == 'beta':
                return np.random.beta(0.4, 0.4)

        def __call__(self, dataset, image, target):
            rnd_image, rnd_target = get_random_sample(dataset)

            alpha = self.sample_alpha()
            image = (1 - alpha) * image + alpha * rnd_image
            target = (1 - alpha) * target + alpha * rnd_target
            target = torch.clip(target, 0.0, 1.0)
            return image, target


    class SigmoidConcatMixer:
        def __init__(self, sigmoid_range=(3, 12)):
            self.sigmoid_range = sigmoid_range

        def sample_mask(self, size):
            x_radius = random.randint(*self.sigmoid_range)

            step = (x_radius * 2) / size[1]
            x = torch.arange(-x_radius, x_radius, step=step).float()
            y = torch.sigmoid(x)
            mix_mask = y.repeat(size[0], 1)
            return mix_mask

        def __call__(self, dataset, image, target):
            rnd_image, rnd_target = get_random_sample(dataset)

            mix_mask = self.sample_mask(image.shape[-2:])
            rnd_mix_mask = 1 - mix_mask

            image = mix_mask * image + rnd_mix_mask * rnd_image
            target = target + rnd_target
            target = torch.clip(target, 0.0, 1.0)
            return image, target


    class RandomMixer:
        def __init__(self, mixers, p=None):
            self.mixers = mixers
            self.p = p

        def __call__(self, dataset, image, target):
            mixer = np.random.choice(self.mixers, p=self.p)
            image, target = mixer(dataset, image, target)
            return image, target


    class UseMixerWithProb:
        def __init__(self, mixer, prob=.5):
            self.mixer = mixer
            self.prob = prob

        def __call__(self, dataset, image, target):
            if random.random() < self.prob:
                return self.mixer(dataset, image, target)
            return image, target

    '''
    Dataset Transforms
    '''
    def image_crop(image, bbox):
        return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]


    # Source: https://www.kaggle.com/davids1992/specaugment-quick-implementation
    def spec_augment(spec: np.ndarray,
                     num_mask=2,
                     freq_masking=0.15,
                     time_masking=0.20,
                     value=0):
        spec = spec.copy()
        num_mask = random.randint(1, num_mask)
        for i in range(num_mask):
            all_freqs_num, all_frames_num  = spec.shape
            freq_percentage = random.uniform(0.0, freq_masking)

            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[f0:f0 + num_freqs_to_mask, :] = value

            time_percentage = random.uniform(0.0, time_masking)

            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[:, t0:t0 + num_frames_to_mask] = value
        return spec


    class SpecAugment:
        def __init__(self,
                     num_mask=2,
                     freq_masking=0.15,
                     time_masking=0.20):
            self.num_mask = num_mask
            self.freq_masking = freq_masking
            self.time_masking = time_masking

        def __call__(self, image):
            return spec_augment(image,
                                self.num_mask,
                                self.freq_masking,
                                self.time_masking,
                                image.min())


    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, image, trg=None):
            if trg is None:
                for t in self.transforms:
                    image = t(image)
                return image
            else:
                for t in self.transforms:
                    image, trg = t(image, trg)
                return image, trg


    class UseWithProb:
        def __init__(self, transform, prob=.5):
            self.transform = transform
            self.prob = prob

        def __call__(self, image, trg=None):
            if trg is None:
                if random.random() < self.prob:
                    image = self.transform(image)
                return image
            else:
                if random.random() < self.prob:
                    image, trg = self.transform(image, trg)
                return image, trg


    class OneOf:
        def __init__(self, transforms, p=None):
            self.transforms = transforms
            self.p = p

        def __call__(self, image, trg=None):
            transform = np.random.choice(self.transforms, p=self.p)
            if trg is None:
                image = transform(image)
                return image
            else:
                image, trg = transform(image, trg)
                return image, trg


    class ToTensor:
        def __call__(self, array):
            return torch.from_numpy(array).float()


    class RandomCrop:
        def __init__(self, size):
            self.size = size

        def __call__(self, signal):
            start = random.randint(0, signal.shape[1] - self.size)
            return signal[:, start: start + self.size]


    class CenterCrop:
        def __init__(self, size):
            self.size = size

        def __call__(self, signal):

            if signal.shape[1] > self.size:
                start = (signal.shape[1] - self.size) // 2
                return signal[:, start: start + self.size]
            else:
                return signal


    class PadToSize:
        def __init__(self, size, mode='constant'):
            assert mode in ['constant', 'wrap']
            self.size = size
            self.mode = mode

        def __call__(self, signal):
            if signal.shape[1] < self.size:
                padding = self.size - signal.shape[1]
                offset = padding // 2
                pad_width = ((0, 0), (offset, padding - offset))
                if self.mode == 'constant':
                    signal = np.pad(signal, pad_width,
                                    'constant', constant_values=signal.min())
                else:
                    signal = np.pad(signal, pad_width, 'wrap')
            return signal


    def get_transforms_fsd_chunks(train, size,
                                  wrap_pad_prob=0.5,
                                  spec_num_mask=2,
                                  spec_freq_masking=0.15,
                                  spec_time_masking=0.20,
                                  spec_prob=0.5):
        if train:
            transforms = Compose([
                OneOf([
                    PadToSize(size, mode='wrap'),
                    PadToSize(size, mode='constant'),
                ], p=[wrap_pad_prob, 1 - wrap_pad_prob]),
                UseWithProb(SpecAugment(num_mask=spec_num_mask,
                                        freq_masking=spec_freq_masking,
                                        time_masking=spec_time_masking), spec_prob),
                RandomCrop(size),       # it's okay, our chunks are of exact `size` timesteps anyway
                ToTensor()
            ])
        else:
            transforms = Compose([
                PadToSize(size),
                # CenterCrop(size),
                ToTensor()
            ])
        return transforms



    '''
    Dataloader Collate Functions
    '''
    def _collate_fn(bilevel_batch):
        batch1 = [(x[0],x[1]) for x in bilevel_batch]
        batch2 = [(x[2],x[3]) for x in bilevel_batch]
        batch3 = [(x[4],x[5]) for x in bilevel_batch]
        batch4 = [(x[6],x[7]) for x in bilevel_batch]
        batch5 = [(x[8],x[9]) for x in bilevel_batch]

        t1, l1 = _collate_fn_part(batch1)
        t2, l2 = _collate_fn_part(batch2)
        t3, l3 = _collate_fn_part(batch3)
        t4, l4 = _collate_fn_part(batch4)
        t5, l5 = _collate_fn_part(batch5)

        return t1, l1, t2, l2, t3, l3, t4, l4, t5, l5

    def _collate_fn_part(batch):
        def func(p):
            return p[0].size(1)

        batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
        longest_sample = max(batch, key=func)[0]
        freq_size = longest_sample.size(0)
        minibatch_size = len(batch)
        max_seqlength = longest_sample.size(1)
        inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
        inputs_complex = torch.zeros((minibatch_size, 1, freq_size, max_seqlength), dtype=torch.complex64)
        targets = []
        for x in range(minibatch_size):
            sample = batch[x]
            real_tensor = sample[0]
            target = sample[1]
            seq_length = real_tensor.size(1)
            inputs[x][0].narrow(1, 0, seq_length).copy_(real_tensor)
            targets.append(target.unsqueeze(0))
        targets = torch.cat(targets)
        return inputs, targets


    def _collate_fn_multiclass(batch):
        def func(p):
            return p[0].size(1)

        batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
        longest_sample = max(batch, key=func)[0]
        freq_size = longest_sample.size(0)
        minibatch_size = len(batch)
        max_seqlength = longest_sample.size(1)
        inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
        inputs_complex = torch.zeros((minibatch_size, 1, freq_size, max_seqlength), dtype=torch.complex64)
        targets = torch.LongTensor(minibatch_size)
        for x in range(minibatch_size):
            sample = batch[x]
            real_tensor = sample[0]
            target = sample[1]
            seq_length = real_tensor.size(1)
            inputs[x][0].narrow(1, 0, seq_length).copy_(real_tensor)
            targets[x] = target
        return inputs, inputs_complex, targets

    '''
    Audio Parsing
    '''
    def load_audio(f, sr, min_duration: float = 5.):
        if min_duration is not None:
            min_samples = int(sr * min_duration)
        else:
            min_samples = None
        x, clip_sr = sf.read(f)
        x = x.astype('float32')
        assert clip_sr == sr

        # min filtering and padding if needed
        if min_samples is not None:
            if len(x) < min_samples:
                tile_size = (min_samples // x.shape[0]) + 1
                x = np.tile(x, tile_size)[:min_samples]
        return x

    class AudioParser(object):
        def __init__(self, n_fft=511, win_length=None, hop_length=None, sample_rate=22050,
                     feature="spectrogram", top_db=150):
            super(AudioParser, self).__init__()
            self.n_fft = n_fft
            self.win_length = self.n_fft if win_length is None else win_length
            self.hop_length = self.n_fft//2 if hop_length is None else hop_length
            assert feature in ['melspectrogram', 'spectrogram']
            self.feature = feature
            self.top_db = top_db
            if feature == "melspectrogram":
                self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=96 * 20,
                                                                    win_length=int(sample_rate * 0.03),
                                                                    hop_length=int(sample_rate * 0.01),
                                                                    n_mels=96)
            else:
                self.melspec = None

        def __call__(self, audio):
            if self.feature == 'spectrogram':
                # TOP_DB = 150
                comp = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
                                    win_length=self.win_length)
                real = np.abs(comp)
                real = librosa.amplitude_to_db(real, top_db=self.top_db)
                real += self.top_db / 2

                mean = real.mean()
                real -= mean        # per sample Zero Centering
                return real, comp

            elif self.feature == 'melspectrogram':
                # melspectrogram features, as per FSD50k paper
                x = torch.from_numpy(audio).unsqueeze(0)
                specgram = self.melspec(x)[0].numpy()
                specgram = librosa.power_to_db(specgram)
                specgram = specgram.astype('float32')
                specgram += self.top_db / 2
                mean = specgram.mean()
                specgram -= mean
                return specgram, specgram

    '''
    Dataset Classes
    '''

    class SpectrogramDataset(Dataset):
        def __init__(self, manifest_path: str, labels_map: str,
                     audio_config: dict, mode: Optional[str] = "multilabel",
                     augment: Optional[bool] = False,
                     labels_delimiter: Optional[str] = ",",
                     mixer: Optional = None,
                     transform: Optional = None) -> None:
            super(SpectrogramDataset, self).__init__()
            assert os.path.isfile(labels_map)
            assert os.path.splitext(labels_map)[-1] == ".json"
            assert audio_config is not None
            with open(labels_map, 'r') as fd:
                self.labels_map = json.load(fd)

            self.len = None
            self.labels_delim = labels_delimiter
            df = pd.read_csv(manifest_path)
            self.files = df['files'].values
            self.labels = df['labels'].values
            self.exts = df['ext'].values
            self.unique_exts = np.unique(self.exts)
            assert len(self.files) == len(self.labels) == len(self.exts)
            self.len = len(self.unique_exts)
            print(self.len)
            self.sr = audio_config.get("sample_rate", "22050")
            self.n_fft = audio_config.get("n_fft", 511)
            win_len = audio_config.get("win_len", None)
            if not win_len:
                self.win_len = self.n_fft
            else:
                self.win_len = win_len
            hop_len = audio_config.get("hop_len", None)
            if not hop_len:
                self.hop_len = self.n_fft // 2
            else:
                self.hop_len = hop_len

            self.normalize = audio_config.get("normalize", True)
            self.augment = augment
            self.min_duration = audio_config.get("min_duration", None)
            self.background_noise_path = audio_config.get("bg_files", None)
            if self.background_noise_path is not None:
                if os.path.exists(self.background_noise_path):
                    self.bg_files = glob.glob(os.path.join(self.background_noise_path, "*.wav"))
            else:
                self.bg_files = None

            self.mode = mode
            feature = audio_config.get("feature", "spectrogram")
            self.spec_parser = AudioParser(n_fft=self.n_fft, win_length=self.win_len,
                                           hop_length=self.hop_len, feature=feature)
            self.mixer = mixer
            self.transform = transform

            if self.bg_files is not None:
                print("prepping bg_features")
                self.bg_features = []
                for f in tqdm.tqdm(self.bg_files):
                    preprocessed_audio = self.__get_audio__(f)
                    real, comp = self.__get_feature__(preprocessed_audio)
                    self.bg_features.append(real)
            else:
                self.bg_features = None
            self.prefetched_labels = None
            if self.mode == "multilabel":
                self.fetch_labels()

        def fetch_labels(self):
            self.prefetched_labels = []
            for lbl in tqdm.tqdm(self.labels):
                proc_lbl = self.__parse_labels__(lbl)
                self.prefetched_labels.append(proc_lbl.unsqueeze(0))
            self.prefetched_labels = torch.cat(self.prefetched_labels, dim=0)
            print(self.prefetched_labels.shape)

        def __get_audio__(self, f):
            audio = load_audio(f, self.sr, self.min_duration)
            return audio

        def __get_feature__(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            real, comp = self.spec_parser(audio)
            return real, comp

        def get_bg_feature(self, index: int) -> torch.Tensor:
            if self.bg_files is None:
                return None
            real = self.bg_features[index]
            if self.transform is not None:
                real = self.transform(real)
            return real

        def __get_item_helper__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            f = self.files[index]
            f = '/datasets' + f[4:]
            lbls = self.labels[index]
            label_tensor = self.__parse_labels__(lbls)
            preprocessed_audio = self.__get_audio__(f)
            real, comp = self.__get_feature__(preprocessed_audio)
            if self.transform is not None:
                real = self.transform(real)
            return real, comp, label_tensor

        def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
            tgt_ext = self.unique_exts[index]
            idxs = np.where(self.exts == tgt_ext)[0]
            rand_index = np.random.choice(idxs)
            
            real, comp, label_tensor = self.__get_item_helper__(rand_index)
            if self.mixer is not None:
                real, final_label = self.mixer(self, real, label_tensor)
                if self.mode != "multiclass":
                    real = F.pad(real, (0,1))
                    return real.unsqueeze(0), final_label
            real = F.pad(real, (0,1))
            return real.unsqueeze(0), label_tensor

        def __parse_labels__(self, lbls: str) -> torch.Tensor:
            if self.mode == "multilabel":
                label_tensor = torch.zeros(len(self.labels_map)).float()
                for lbl in lbls.split(self.labels_delim):
                    label_tensor[self.labels_map[lbl]] = 1

                return label_tensor
            elif self.mode == "multiclass":
                return self.labels_map[lbls]

        def __len__(self):
            return self.len

        def get_bg_len(self):
            return len(self.bg_files)

    '''
    Audio Eval 
    '''
    class FSD50kEvalDataset(Dataset):
        def __init__(self, manifest_path: str, labels_map: str,
                     audio_config: dict,
                     labels_delimiter: Optional[str] = ",",
                     transform: Optional = None) -> None:
            super(FSD50kEvalDataset, self).__init__()
            assert os.path.isfile(labels_map)
            assert os.path.splitext(labels_map)[-1] == ".json"
            assert audio_config is not None
            with open(labels_map, 'r') as fd:
                self.labels_map = json.load(fd)

            self.len = None
            self.labels_delim = labels_delimiter
            df = pd.read_csv(manifest_path)
            self.files = df['files'].values
            self.labels = df['labels'].values
            self.exts = df['ext'].values
            self.unique_exts = np.unique(self.exts)

            assert len(self.files) == len(self.labels) == len(self.exts)
            self.len = len(self.unique_exts)
            print("LENGTH OF VAL SET:", self.len)
            self.sr = audio_config.get("sample_rate", "22050")
            self.n_fft = audio_config.get("n_fft", 511)
            win_len = audio_config.get("win_len", None)
            if not win_len:
                self.win_len = self.n_fft
            else:
                self.win_len = win_len
            hop_len = audio_config.get("hop_len", None)
            if not hop_len:
                self.hop_len = self.n_fft // 2
            else:
                self.hop_len = hop_len

            self.normalize = audio_config.get("normalize", False)
            self.min_duration = audio_config.get("min_duration", None)

            feature = audio_config.get("feature", "spectrogram")
            self.spec_parser = AudioParser(n_fft=self.n_fft, win_length=self.win_len,
                                           hop_length=self.hop_len, feature=feature)
            self.transform = transform

        def __get_audio__(self, f):
            audio = load_audio(f, self.sr, self.min_duration)
            return audio

        def __get_feature__(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            real, comp = self.spec_parser(audio)
            return real, comp

        def __get_item_helper__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            f = self.files[index]
            f = '/datasets' + f[4:]
            lbls = self.labels[index]
            label_tensor = self.__parse_labels__(lbls)
            preprocessed_audio = self.__get_audio__(f)
            real, comp = self.__get_feature__(preprocessed_audio)
            if self.transform is not None:
                real = self.transform(real)

            real = F.pad(real, (0,1))
            return real, comp, label_tensor

        def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
            tgt_ext = self.unique_exts[index]
            idxs = np.where(self.exts == tgt_ext)[0]
            tensors = []
            label_tensors = []
            for idx in idxs:
                real, comp, label_tensor = self.__get_item_helper__(idx)
                tensors.append(real.unsqueeze(0).unsqueeze(0))
                label_tensors.append(label_tensor.unsqueeze(0))
                
            tensors = torch.cat(tensors)
            return tensors, label_tensors[0]

        def __parse_labels__(self, lbls: str) -> torch.Tensor:
            label_tensor = torch.zeros(len(self.labels_map)).float()
            for lbl in lbls.split(self.labels_delim):
                label_tensor[self.labels_map[lbl]] = 1

            return label_tensor

        def __len__(self):
            return self.len


    def _collate_fn_eval(batch):
        return batch[0][0], batch[0][1]
except:
    pass
