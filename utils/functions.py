from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import torch
import numpy as np
from math import log
from utils.cl_utils import (
    cl_decoder, batch_cl_decoder, batch_cl_analysis
)
import torchvision.transforms.functional as tf
from utils.utils import (
    gen_annotation,
    confusion_matrix,
    confusion_matrix_with_image,
    cal_metrics,
    format_annotation
)
from tqdm import tqdm
import time

def bagging_train_valid(config, epoch, model, loader, criterion, params, optimizer=None, train=True):
    model.train() if train else model.eval()
    debug_records = []
    epoch_records = {'time': []}
    num_batchs = len(loader)
    v_loss_weight, d_loss_weight = params['v_loss_weight'], params['d_loss_weight']
    print('v loss weight: {} | d loss weight: {}'.format(v_loss_weight, d_loss_weight))
    for batch_idx, (sagittal_images, heatmaps, offymaps, offxmaps, maskmaps, clssmaps, _, _) in enumerate(loader):
        start_time = time.time()
        images = sagittal_images.float().to(config.device)
        heat_targets = heatmaps.float().to(config.device)
        offy_targets = offymaps.float().to(config.device)
        offx_targets = offxmaps.float().to(config.device)
        masks = maskmaps.float().to(config.device)
        clss_targets = clssmaps.long().to(config.device)

        if train:
            predictions = model(images, heatmaps=heat_targets, offymaps=offy_targets, offxmaps=offx_targets)
            heatmap, offymap, offxmap, clssmap = predictions['result']
            loss, info = criterion(heatmap, offymap, offxmap, clssmap,
                    heat_targets, offy_targets, offx_targets, clss_targets, masks, 
                    v_loss_weight, d_loss_weight, name='predict')
            loss = torch.abs(loss - config.flooding_d) + config.flooding_d

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                predictions = model(images, heatmaps=heat_targets, offymaps=offy_targets, offxmaps=offx_targets)
            heatmap, offymap, offxmap, clssmap = predictions['result']
            loss, info = criterion(heatmap, offymap, offxmap, clssmap,
                    heat_targets, offy_targets, offx_targets, clss_targets, masks, 
                    v_loss_weight, d_loss_weight, name='predict')

        for key, value in info.items():
            if key not in epoch_records: epoch_records[key] = []
            epoch_records[key].append(value)
        epoch_records['time'].append(time.time() - start_time)

        batch_records = batch_cl_analysis(heat_targets, clss_targets, clssmap)
        debug_records.extend(batch_records)

        if (batch_idx and batch_idx % config.display == 0) or (batch_idx == num_batchs - 1):
            context = '[{}] EP:{:03d}\tTI:{:03d}/{:03d}\t'.format('T' if train else 'V', epoch, batch_idx, num_batchs)
            context_predict = '\t'
            context_class = '\t'
            for key, value in epoch_records.items():
                if 'cls' in key:
                    context_class += '{}:{:.6f}({:.6f})\t'.format(key, value[-1], np.mean(value))
                elif 'predict' in key:
                    context_predict += '{}:{:.6f}({:.6f})\t'.format(key, value[-1], np.mean(value))
                else:
                    context += '{}:{:.6f}({:.6f})\t'.format(key, value[-1], np.mean(value))
            print(context)
            print(context_predict)
            print(context_class)
            
    if not train:
        params['v_loss'].append(np.mean(epoch_records['predict_v_cls_loss']))
        params['d_loss'].append(np.mean(epoch_records['predict_d_cls_loss']))

    return np.mean(epoch_records['loss']), params, epoch_records



def train(config, epoch, model, train_loader, criterion, optimizer):
    model.train()
    debug_records = []
    epoch_records = {'time': []}
    num_batchs = len(train_loader)
    for batch_idx, (sagittal_images, heatmaps, offymaps, offxmaps, maskmaps, clssmaps, _, _) in enumerate(
            train_loader):
        start_time = time.time()
        images = sagittal_images.float().to(config.device)
        heat_targets = heatmaps.float().to(config.device)
        offy_targets = offymaps.float().to(config.device)
        offx_targets = offxmaps.float().to(config.device)
        masks = maskmaps.float().to(config.device)
        clss_targets = clssmaps.long().to(config.device)

        predictions = model(images, heatmaps=heat_targets, offymaps=offy_targets, offxmaps=offx_targets)

        heatmap, offymap, offxmap, clssmap = predictions['result']
        loss, info = criterion(heatmap, offymap, offxmap, clssmap,
                               heat_targets, offy_targets, offx_targets, clss_targets, masks, name='predict')

        loss = torch.abs(loss - config.flooding_d) + config.flooding_d

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for key, value in info.items():
            if key not in epoch_records: epoch_records[key] = []
            epoch_records[key].append(value)
        epoch_records['time'].append(time.time() - start_time)

        batch_records = batch_cl_analysis(heat_targets, clss_targets, clssmap)
        debug_records.extend(batch_records)

        if (batch_idx and batch_idx % config.display == 0) or (batch_idx == num_batchs - 1):
            context = 'EP:{:03d}\tTI:{:03d}/{:03d}\t'.format(epoch, batch_idx, num_batchs)
            context_predict = '\t'
            context_class = '\t'
            for key, value in epoch_records.items():
                if 'cls' in key:
                    context_class += '{}:{:.6f}({:.6f})\t'.format(key, value[-1], np.mean(value))
                elif 'predict' in key:
                    context_predict += '{}:{:.6f}({:.6f})\t'.format(key, value[-1], np.mean(value))
                else:
                    context += '{}:{:.6f}({:.6f})\t'.format(key, value[-1], np.mean(value))
            print(context)
            print(context_predict)
            print(context_class)
            
#     with open('debug/{:03d}.txt'.format(epoch), 'w') as f:
#         for record in debug_records:
#             f.write(str(record)+'\n')

    return epoch_records

def compute_distance(P, Q, mode):
    if mode == 'kl':
        def _asymmetricKL(P_, Q_):
            return sum(P_ * log(P_ / Q_))
        return (_asymmetricKL(P, Q) + _asymmetricKL(Q, P)) / 2.
    elif mode == 'mse':
        return np.sum((P - Q) ** 2) ** 0.2

def evaluate(config, model, studies, annotation_path=None, trans_model=None, class_tensors=None):
    model.eval()
    annotations = []
    for study in studies.values():
        kp_frame = study.t2_sagittal_middle_frame
        sagittal_image = tf.resize(kp_frame.image, config.sagittal_size)
        sagittal_image = tf.to_tensor(sagittal_image).unsqueeze(dim=0).float().to(config.device)
        with torch.no_grad():
            predictions = model(sagittal_image)
            heatmaps, offymaps, offxmaps, clssmaps = predictions['result']
            if config.test_flip:
                sagittal_image_flipped = np.flip(sagittal_image.detach().cpu().numpy(), 3).copy()
                sagittal_image_flipped = torch.from_numpy(sagittal_image_flipped).to(sagittal_image.device)
                predictions_flipped = model(sagittal_image_flipped)
                heatmaps_flipped, _, _, clssmaps_flipped = predictions_flipped['result']
                heatmaps = (heatmaps + torch.from_numpy(np.flip(heatmaps_flipped.detach().cpu().numpy(), 3).copy()).to(
                    heatmaps.device)) / 2.
                clssmaps = (clssmaps + torch.from_numpy(np.flip(clssmaps_flipped.detach().cpu().numpy(), 3).copy()).to(
                    clssmaps.device)) / 2.
            heatmaps = torch.mean(heatmaps, dim=0, keepdim=True)
            offymaps = torch.mean(offymaps, dim=0, keepdim=True)
            offxmaps = torch.mean(offxmaps, dim=0, keepdim=True)
            clssmaps = torch.mean(clssmaps, dim=0, keepdim=True)
        prediction, _, cls_prediction, d_v5_cls_prediction = cl_decoder(
            config, heatmaps[0], offymaps[0], offxmaps[0], clssmaps[0], maskmap=None, tensor=True)
        if (trans_model is not None) and (class_tensors is not None):
            try:
                with torch.no_grad():
                    trans_images, _ = study.t2_transverse_k_nearest(config, False, prediction[5:, :2].float(), 1) 
                    trans_images = trans_images[:, 0, :].repeat(1, 3, 1, 1).float().to(config.device) # (6, 3, 96, 96)
                    trans_outputs = model(trans_images, True)
                    for d_index, (trans_image, trans_output) in enumerate(zip(trans_images, trans_outputs)):
                        trans_image = trans_image.detach().cpu().numpy()
                        if trans_image.sum() == 0: continue
                        trans_output = trans_output.detach().cpu().numpy()
                        distance = np.array([compute_distance(class_tensor, output, mode='mse') for class_tensor in class_tensors])
                        cls_prediction[5+d_index] = np.argmin(distance)
            except:
                pass
        height_ratio = config.sagittal_size[0] / kp_frame.size[1]
        width_ratio = config.sagittal_size[1] / kp_frame.size[0]
        ratio = torch.tensor([width_ratio, height_ratio], device=prediction.device)
        prediction = (prediction / ratio).round().float()
        annotation = gen_annotation(config, study, prediction, cls_prediction, d_v5_cls_prediction)
        annotations.append(annotation)

    if annotation_path is None:
        return annotations
    else:
        predictions = annotations
        with open(annotation_path, 'r') as file:
            annotations = json.load(file)

        annotations = format_annotation(annotations)

    matrix = confusion_matrix(config, studies, predictions, annotations)
    outputs = cal_metrics(matrix)

    i = 0
    while i < len(outputs) and outputs[i][0] != config.metric:
        i += 1
    if i < len(outputs):
        outputs = [outputs[i]] + outputs[:i] + outputs[i + 1:]

    return outputs


def valid(config, model, valid_studies, annotation_path, trans_model, class_tensors):
    metrics_values = evaluate(config, model, valid_studies, annotation_path=annotation_path, trans_model=trans_model, class_tensors=class_tensors)
    metrics_value = metrics_values[1][1]
    for data in metrics_values:
        print('valid {}: '.format(data[0]), *data[1:])
    return metrics_value


def testA(config, model, testA_studies, submission_file, trans_model, class_tensors):
    predictions = evaluate(config, model, testA_studies, annotation_path=None, trans_model=trans_model, class_tensors=class_tensors)
    with open(submission_file, 'w') as file:
        json.dump(predictions, file)
    print('=' * 100)
    print('Generated submission file: {}'.format(submission_file))


def testB(config, model, testA_studies, submission_file, trans_model, class_tensors):
    return testA(config, model, testA_studies, submission_file, trans_model=trans_model, class_tensors=class_tensors)

def vote_by_models(models_prediction, models_cls_prediction, models_d_v5_cls_prediction, tensor=True):
    prediction = np.mean(np.stack(models_prediction, axis=0), axis=0)
    models_cls_prediction = np.transpose(np.stack(models_cls_prediction, axis=0), [1, 0]) # (11, N)
    cls_prediction = np.array([np.argmax(np.bincount(line)) for line in models_cls_prediction])
    models_d_v5_cls_prediction = np.transpose(np.stack(models_d_v5_cls_prediction, axis=0), [1, 0]) # (6, N)
    d_v5_cls_prediction = np.array([np.argmax(np.bincount(line)) for line in models_d_v5_cls_prediction])
    if tensor:
        prediction = torch.from_numpy(prediction)
        cls_prediction = torch.from_numpy(cls_prediction)
        d_v5_cls_prediction = torch.from_numpy(d_v5_cls_prediction)
    return prediction, cls_prediction, d_v5_cls_prediction

def evaluate_cross(config, models, studies, annotation_path=None, trans_model=None, class_tensors=None):
    for model in models:
        model.eval()
    annotations = []
    for study in studies.values():
        with torch.no_grad():
            kp_frame = study.t2_sagittal_middle_frame
            sagittal_image = tf.resize(kp_frame.image, config.sagittal_size)
            sagittal_image = tf.to_tensor(sagittal_image).unsqueeze(dim=0).float().to(config.device)
            models_prediction, models_cls_prediction, models_d_v5_cls_prediction = [], [], []
            models_heatmaps, models_offymaps, models_offxmaps, models_clssmaps = [], [], [], []
            for model in models:
                predictions = model(sagittal_image)
                heatmaps, offymaps, offxmaps, clssmaps = predictions['result']
                if config.test_flip:
                    sagittal_image_flipped = np.flip(sagittal_image.detach().cpu().numpy(), 3).copy()
                    sagittal_image_flipped = torch.from_numpy(sagittal_image_flipped).to(sagittal_image.device)
                    predictions_flipped = model(sagittal_image_flipped)
                    heatmaps_flipped, _, _, clssmaps_flipped = predictions_flipped['result']
                    heatmaps = (heatmaps + torch.from_numpy(np.flip(heatmaps_flipped.detach().cpu().numpy(), 3).copy()).to(
                        heatmaps.device)) / 2.
                    clssmaps = (clssmaps + torch.from_numpy(np.flip(clssmaps_flipped.detach().cpu().numpy(), 3).copy()).to(
                        clssmaps.device)) / 2.
                models_heatmaps.append(heatmaps[0])
                models_offymaps.append(offymaps[0])
                models_offxmaps.append(offxmaps[0])
                models_clssmaps.append(clssmaps[0])
                prediction, _, cls_prediction, d_v5_cls_prediction = cl_decoder(
                    config, heatmaps[0], offymaps[0], offxmaps[0], clssmaps[0], maskmap=None, tensor=False)
                models_prediction.append(prediction)
                models_cls_prediction.append(cls_prediction)
                models_d_v5_cls_prediction.append(d_v5_cls_prediction)
            models_heatmap = torch.mean(torch.stack(models_heatmaps, dim=0), dim=0)
            models_offymap = torch.mean(torch.stack(models_offymaps, dim=0), dim=0)
            models_offxmap = torch.mean(torch.stack(models_offxmaps, dim=0), dim=0)
            models_clssmap = torch.mean(torch.stack(models_clssmaps, dim=0), dim=0)
            _, cls_prediction, d_v5_cls_prediction = vote_by_models(models_prediction, models_cls_prediction,
                                                        models_d_v5_cls_prediction, tensor=True)
            prediction, _, _, _ = cl_decoder(config, models_heatmap, models_offymap, models_offxmap, 
                                             models_clssmap, maskmap=None, tensor=True)
        height_ratio = config.sagittal_size[0] / kp_frame.size[1]
        width_ratio = config.sagittal_size[1] / kp_frame.size[0]
        ratio = torch.tensor([width_ratio, height_ratio], device=prediction.device)
        prediction = (prediction / ratio).round().float()
        annotation = gen_annotation(config, study, prediction, cls_prediction, d_v5_cls_prediction)
        annotations.append(annotation)

    if annotation_path is None:
        return annotations
    else:
        predictions = annotations
        with open(annotation_path, 'r') as file:
            annotations = json.load(file)

        annotations = format_annotation(annotations)

    matrix = confusion_matrix(config, studies, predictions, annotations)
    outputs = cal_metrics(matrix)

    i = 0
    while i < len(outputs) and outputs[i][0] != config.metric:
        i += 1
    if i < len(outputs):
        outputs = [outputs[i]] + outputs[:i] + outputs[i + 1:]

    return outputs

def testB_cross(config, models, testB_studies, submission_file, trans_model, class_tensors):
    predictions = evaluate_cross(config, models, testB_studies, annotation_path=None, trans_model=trans_model, class_tensors=class_tensors)
    with open(submission_file, 'w') as file:
        json.dump(predictions, file)
    print('=' * 100)
    print('Generated submission file: {}'.format(submission_file))

