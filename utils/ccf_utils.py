from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import itertools

def _softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def batch_ccf_analysis(heatmaps, clssmaps_target, clssmaps_predict):
    heatmaps = heatmaps.detach().cpu().numpy()
    clssmaps_target = clssmaps_target.detach().cpu().numpy()
    clssmaps_predict = clssmaps_predict.detach().cpu().numpy()
    B, C, H, W = heatmaps.shape
    records = []
    for heatmap, clssmap_target, clssmap_predict in zip(heatmaps, clssmaps_target, clssmaps_predict):
        heatmap_reshaped = heatmap.reshape((C, -1))
        idxs = heatmap_reshaped.argsort(1)[:, -1:]
        for i in range(0, 11):
            grid_y = int(idxs[i, -1] // W)
            grid_x = int(idxs[i, -1] % W)
            clss_target = clssmap_target[i, grid_y, grid_x]
            if i < 5:
                clss_predict = np.argmax(clssmap_predict[i * 2:(i + 1) * 2, grid_y, grid_x], axis=0)
                clss_score = _softmax(clssmap_predict[i * 2:(i + 1) * 2, grid_y, grid_x])
            else:
                clss_predict = np.argmax(clssmap_predict[10 + (i - 5) * 4:10 + (i + 1 - 5) * 4, grid_y, grid_x], axis=0)
                clss_score = _softmax(clssmap_predict[10 + (i - 5) * 4:10 + (i + 1 - 5) * 4, grid_y, grid_x])
            records.append([i, clss_target, clss_predict, clss_target == clss_predict, list(clss_score)])
    return records

def _gaussian(x, sigma):
    return np.exp(-x / (2 * sigma ** 2))

def ccf_encoder(config, gt_coords, gt_classes, tensor=True):
    if not isinstance(gt_coords, np.ndarray):
        gt_coords = gt_coords.detach().cpu().numpy()
    heatmap = np.zeros((config.num_classes, *config.predict_size))
    offymap = np.zeros((config.num_classes, *config.predict_size))
    offxmap = np.zeros((config.num_classes, *config.predict_size))
    maskmap = np.zeros((config.num_classes, *config.predict_size))
    clssmap = np.zeros((config.num_classes + 6, *config.predict_size))

    gridmap = np.array(list(itertools.product(range(1, config.predict_size[0] + 1),
                                              range(1, config.predict_size[1] + 1))))
    gridmap = (gridmap - 0.5) * config.stride

    for i, (gt_coord, gt_class) in enumerate(zip(gt_coords, gt_classes[:gt_coords.shape[0]])):
        distance = np.square((gt_coord[::-1] - gridmap)).sum(axis=-1)
        heatmap[i] = (_gaussian(distance, config.sigma).reshape(*config.predict_size))
        heatmap[i] = heatmap[i] / heatmap[i].max()
        offset = ((gt_coord[::-1] - gridmap) / config.stride).reshape(*config.predict_size, -1)
        offymap[i] = offset[:, :, 0]
        offxmap[i] = offset[:, :, 1]
        clssmap[i] = gt_class
        
        if heatmap[i].max() >= config.threshold:
            maskmap[i] = 1

    for i, gt_class in enumerate(gt_classes[gt_coords.shape[0]:]):
        clssmap[gt_coords.shape[0] + i] = int(gt_class > 0)

    if tensor:
        heatmap = torch.from_numpy(heatmap)
        offymap = torch.from_numpy(offymap)
        offxmap = torch.from_numpy(offxmap)
        maskmap = torch.from_numpy(maskmap)
        clssmap = torch.from_numpy(clssmap)

    return heatmap, offymap, offxmap, maskmap, clssmap

def ccf_decoder(config, heatmap, offymap, offxmap, clssmap, maskmap=None, tensor=True):
    assert len(heatmap.shape) == 3
    assert len(offymap.shape) == 3
    assert len(offxmap.shape) == 3
    if not isinstance(heatmap, np.ndarray):
        heatmap = heatmap.detach().cpu().numpy()
    if not isinstance(offymap, np.ndarray):
        offymap = offymap.detach().cpu().numpy()
    if not isinstance(offxmap, np.ndarray):
        offxmap = offxmap.detach().cpu().numpy()
    if (maskmap is not None) and (not isinstance(maskmap, np.ndarray)):
        maskmap = maskmap.detach().cpu().numpy()
    if (clssmap is not None) and (not isinstance(clssmap, np.ndarray)):
        clssmap = clssmap.detach().cpu().numpy()  # (num_classes * 7, H, W)
    C, H, W = heatmap.shape
    assert C == config.num_classes
    heatmap_reshaped = heatmap.reshape((C, -1))
    idxs = heatmap_reshaped.argsort(1)[:, -config.top_k:]
    scores = np.zeros((C, 1))
    predictions = np.zeros((C, 2))
    cls_predictions = np.array([1, 1, 1, 1, 1, 0, 0, 0, 1, 2, 2])
    d_v5_cls_predictions = np.array([0, 0, 0, 0, 0, 0])
    
#     v_thresholds = torch.Tensor([value[0] / (value[0] + value[1])for key, value in config.v_numbers.items()])
#     d_v5_thresholds = torch.Tensor([value[0] / (value[0] + value[1])for key, value in config.d_v5_numbers.items()])

    for i in range(C):
        if (maskmap is not None) and (maskmap[i].sum() <= 0):
            continue
        weight = 1 if config.vote == 'average' else heatmap_reshaped[i, idxs[i, -1]]
        grid_y = int(idxs[i, -1] // W)
        grid_x = int(idxs[i, -1] % W)
        predictions[i, 1] = (grid_y + offymap[i, grid_y, grid_x] + 0.5) * config.stride * weight
        predictions[i, 0] = (grid_x + offxmap[i, grid_y, grid_x] + 0.5) * config.stride * weight
        scores[i, 0] = heatmap_reshaped[i, idxs[i, -1]]
        num = weight
        if clssmap.shape[0]:
            if i < 5:
                cls_predictions[i] = (_sigmoid(clssmap[i, grid_y, grid_x]) >= 0.5)
            else:
                cls_predictions[i] = np.argmax(clssmap[5 + (i - 5) * 4:5 + (i + 1 - 5) * 4, grid_y, grid_x], axis=0)
                d_v5_cls_predictions[i - 5] = (_sigmoid(clssmap[29 + (i - 5), grid_y, grid_x]) >= 0.5)
                
        for j in range(config.top_k - 1):
            if heatmap_reshaped[i, idxs[i, -(j + 2)]] <= config.threshold:
                continue
            weight = 1 if config.vote == 'average' else heatmap_reshaped[i, idxs[i, -(j + 2)]]
            grid_y = int(idxs[i, -(j + 2)] // W)
            grid_x = int(idxs[i, -(j + 2)] % W)
            predictions[i, 1] += (grid_y + offymap[i, grid_y, grid_x] + 0.5) * config.stride * weight
            predictions[i, 0] += (grid_x + offxmap[i, grid_y, grid_x] + 0.5) * config.stride * weight
            scores[i, 0] += heatmap_reshaped[i, idxs[i, -(j + 2)]]
            num += weight
        predictions[i, 0] /= num
        predictions[i, 1] /= num
        scores[i, 0] /= num

    if tensor:
        predictions = torch.from_numpy(predictions)  # (C, 2)
        scores = torch.from_numpy(scores)  # (C, 1)
        cls_predictions = torch.from_numpy(cls_predictions)  # (C,)
        d_v5_cls_predictions = torch.from_numpy(d_v5_cls_predictions)  # (6,)

    return predictions, scores, cls_predictions, d_v5_cls_predictions

def batch_ccf_decoder(config, heatmaps, offymaps, offxmaps, clssmaps, maskmaps=None, tensor=True):
    results = list(
        map(ccf_decoder, [config] * heatmaps.shape[0], heatmaps, offymaps, offxmaps, clssmaps,
            maskmaps if maskmaps is not None else [None] * heatmaps.shape[0], [tensor] * heatmaps.shape[0]))
    predictions, cls_predictions, d_v5_cls_predictions = [], [], []
    for prediction, _, cls_prediction, d_v5_cls_prediction in results:
        predictions.append(prediction)
        cls_predictions.append(cls_prediction)
        d_v5_cls_predictions.append(d_v5_cls_prediction)
    predictions = torch.stack(predictions, dim=0)  # (B, 11, 2)
    cls_predictions = torch.stack(cls_predictions, dim=0)  # (B, 11)
    d_v5_cls_predictions = torch.stack(d_v5_cls_predictions, dim=0)  # (B, 6)
    return predictions, cls_predictions, d_v5_cls_predictions


def ccf_decoder_prob(config, heatmap, offymap, offxmap, clssmap, maskmap=None, tensor=True):
    assert len(heatmap.shape) == 3
    assert len(offymap.shape) == 3
    assert len(offxmap.shape) == 3
    if not isinstance(heatmap, np.ndarray):
        heatmap = heatmap.detach().cpu().numpy()
    if not isinstance(offymap, np.ndarray):
        offymap = offymap.detach().cpu().numpy()
    if not isinstance(offxmap, np.ndarray):
        offxmap = offxmap.detach().cpu().numpy()
    if (maskmap is not None) and (not isinstance(maskmap, np.ndarray)):
        maskmap = maskmap.detach().cpu().numpy()
    if (clssmap is not None) and (not isinstance(clssmap, np.ndarray)):
        clssmap = clssmap.detach().cpu().numpy()  # (num_classes * 7, H, W)
    C, H, W = heatmap.shape
    assert C == config.num_classes
    heatmap_reshaped = heatmap.reshape((C, -1))
    idxs = heatmap_reshaped.argsort(1)[:, -config.top_k:]
    scores = np.zeros((C, 1))
    predictions = np.zeros((C, 2))
    cls_predictions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).astype(np.float)
    
    for i in range(C):
        if (maskmap is not None) and (maskmap[i].sum() <= 0):
            continue
        weight = 1 if config.vote == 'average' else heatmap_reshaped[i, idxs[i, -1]]
        grid_y = int(idxs[i, -1] // W)
        grid_x = int(idxs[i, -1] % W)
        predictions[i, 1] = (grid_y + offymap[i, grid_y, grid_x] + 0.5) * config.stride * weight
        predictions[i, 0] = (grid_x + offxmap[i, grid_y, grid_x] + 0.5) * config.stride * weight
        scores[i, 0] = heatmap_reshaped[i, idxs[i, -1]]
        num = weight
        if clssmap.shape[0]:
            cls_predictions[i] = _sigmoid(clssmap[i, grid_y, grid_x])
                
        for j in range(config.top_k - 1):
            if heatmap_reshaped[i, idxs[i, -(j + 2)]] <= config.threshold:
                continue
            weight = 1 if config.vote == 'average' else heatmap_reshaped[i, idxs[i, -(j + 2)]]
            grid_y = int(idxs[i, -(j + 2)] // W)
            grid_x = int(idxs[i, -(j + 2)] % W)
            predictions[i, 1] += (grid_y + offymap[i, grid_y, grid_x] + 0.5) * config.stride * weight
            predictions[i, 0] += (grid_x + offxmap[i, grid_y, grid_x] + 0.5) * config.stride * weight
            scores[i, 0] += heatmap_reshaped[i, idxs[i, -(j + 2)]]
            num += weight
        predictions[i, 0] /= num
        predictions[i, 1] /= num
        scores[i, 0] /= num

    if tensor:
        predictions = torch.from_numpy(predictions)  # (C, 2)
        scores = torch.from_numpy(scores)  # (C, 1)
        cls_predictions = torch.from_numpy(cls_predictions)  # (C,)

    return predictions, scores, cls_predictions