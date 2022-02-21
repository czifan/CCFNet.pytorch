from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import torch
import torch.nn as nn
import time
import numpy as np
from configs.config_ours import config
from networks.ccfnet import get_ccfnet
from networks.ccfloss import CCFLoss
from utils.functions import (
    train, valid, testA, testB, bagging_train_valid
)
from torch.utils.data import DataLoader
from utils.utils import build_logging, distance
from utils.study_utils import construct_studies
from utils.datasets import SpineDataSet
import time
from sklearn.model_selection import train_test_split
import random
import torch.backends.cudnn as cudnn
from sklearn.model_selection import KFold
import torchvision.transforms.functional as tf
from utils.ccf_utils import ccf_decoder_prob
from utils.utils import confusion_matrix, format_annotation, gen_annotation, cal_metrics, compute_my_metric
from sklearn import metrics
from sklearn.model_selection import KFold
from thop import profile

def compute_params_flops(config, model):
    flops, params = profile(model, inputs=(torch.randn(1, 3, *config.sagittal_size).to(config.device),))
    config.logger.info('FLOPs = ' + str(flops/1000**3) + 'G')
    config.logger.info('Params = ' + str(params/1000**2) + 'M')

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

import argparse

parser = argparse.ArgumentParser(description='CCFNet')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--train_data_dir', default='', type=str)
parser.add_argument('--train_json_file', default='', type=str)
parser.add_argument('--valid_data_dir', default='', type=str)
parser.add_argument('--valid_json_file', default='', type=str)
parser.add_argument('--num_folds', default=4, type=int)
parser.add_argument('--fold_id', default=1, type=int)
args = parser.parse_args()

def calc_weights(annotation):
    v_numbers = {i: np.zeros(2) for i in range(5)}
    d_numbers = {i: np.zeros(4) for i in range(6)}
    d_v5_numbers = {i: np.zeros(2) for i in range(6)}
    for key, value in annotation.items():
        for i, (_, _, c) in enumerate(value[0]):
            v_numbers[i][int(c.item())] += 1
        for i, (_, _, c) in enumerate(value[1]):
            if c.item() >= 4: continue
            d_numbers[i][int(c.item())] += 1
        for i, (_, _, c) in enumerate(value[2]):
            if c.item() > 0:
                d_v5_numbers[i][1] += 1
            else:
                d_v5_numbers[i][0] += 1
    v_weights = 1. / np.sum(np.stack([value for value in v_numbers.values()], axis=0), axis=0)
    d_weights = 1. / np.sum(np.stack([value for value in d_numbers.values()], axis=0), axis=0)
    d_v5_weights = 1. / np.sum(np.stack([value for value in d_v5_numbers.values()], axis=0), axis=0)
    return v_numbers, d_numbers, d_v5_numbers, v_weights, d_weights, d_v5_weights

def prepare_data():
    train_studies, train_annotation, train_counter = construct_studies(
        config, config.train_data_dir, config.train_json_file, multi_processing=config.multi_processing)
    valid_studies, valid_annotation, valid_counter = construct_studies(
        config, config.valid_data_dir, config.valid_json_file, multi_processing=config.multi_processing)
    train_studies.update(valid_studies)
    train_annotation.update(valid_annotation)

    studies = [(k, v) for k, v in train_studies.items()]
    random.seed(221)
    random.shuffle(studies)
    studies = [studies[0:50], studies[50:100], studies[100:150], studies[150:]]

    tmp_studies = []
    for j in range(args.num_folds):
        if j == args.fold_id: continue 
        tmp_studies.extend(studies[j])
    train_k_studies = {k: v for i, (k, v) in enumerate(tmp_studies)}
    valid_k_studies = {k: v for i, (k, v) in enumerate(studies[args.fold_id])}
    train_k_annotation = {k: v for k, v in train_annotation.items() if k[0] in train_k_studies.keys()}
    valid_k_annotation = {k: v for k, v in train_annotation.items() if k[0] in valid_k_studies.keys()}


    print('Split dataset: {} train | {} valid'.format(len(train_k_studies), len(valid_k_studies)))
    valid_k_annotation = [anno for anno in
                        json.load(open(config.train_json_file, 'r')) + json.load(open(config.valid_json_file, 'r'))
                        if anno['studyUid'] in valid_k_studies.keys()]
    # json.dump(valid_k_annotation, open('spinaldiease_dataset/valid.json', 'w'))
    # valid_json_file = 'spinaldiease_dataset/valid.json'

    return (train_k_studies, train_k_annotation), (valid_k_studies, valid_k_annotation)

def gen_annotation_prob(config, study, prediction, cls_prediction):
    z_index = study.t2_sagittal.instance_uids[study.t2_sagittal_middle_frame.instance_uid]
    point = []
    for i, (coord, identification, cls, is_disc) in enumerate(zip(
            prediction, config.identifications, cls_prediction, config.is_disc)):
        point.append({
            'coord': coord.cpu().int().numpy().tolist(),
            'tag': {
                'identification': identification,
                'disc' if is_disc else 'vertebra': cls,
            },
            'zIndex': z_index
        })
    annotation = {
        'studyUid': study.study_uid,
        'data': [
            {
                'instanceUid': study.t2_sagittal_middle_frame.instance_uid,
                'seriesUid': study.t2_sagittal_middle_frame.series_uid,
                'annotation': [
                    {
                        'data': {
                            'point': point,
                        }
                    }
                ]
            }
        ]
    }
    return annotation

def format_annotation_prob(annotations):
    output = {}
    for annotation in annotations:
        study_uid = annotation['studyUid']
        series_uid = annotation['data'][0]['seriesUid']
        instance_uid = annotation['data'][0]['instanceUid']
        temp = {}
        for point in annotation['data'][0]['annotation'][0]['data']['point']:
            identification = point['tag']['identification']
            coord = point['coord']
            if 'disc' in point['tag']:
                disease = point['tag']['disc']
            else:
                disease = point['tag']['vertebra']
            if isinstance(disease, str):
                if 'v1' in disease: disease = 0.0
                else: disease = 1.0
            temp[identification] = {
                'coord': coord,
                'disease': disease,
            }
        output[study_uid] = {
            'seriesUid': series_uid,
            'instanceUid': instance_uid,
            'annotation': temp
        }
    return output

def compute_auc(y_true, y_pred):
    if len(y_pred) == 0: return 0.0
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    return metrics.auc(fpr, tpr)

def compute_my_metric_prob(config, studies, predictions, annotations):
    disc_point_pred, disc_point_true = [], []
    vertebra_point_pred, vertebra_point_true = [], []

    disc_cls_pred, disc_cls_true = [], []
    vertebra_cls_pred, vertebra_cls_true = [], []
    
    for study_uid in studies.keys():
        if study_uid not in annotations:
            print(study_uid, '++++++')
            continue
        annotation = annotations[study_uid]
        study = studies[study_uid]
        pixel_spacing = study.t2_sagittal_middle_frame.pixel_spacing
        pred_points = predictions[study_uid]['annotation']
        for identification, gt_point in annotation['annotation'].items():
            gt_coord = gt_point['coord']
            gt_disease = gt_point['disease']
            
            if identification not in pred_points:
                print(identification, pred_points, '-----------------')
                continue 
                
            pred_coord = pred_points[identification]['coord']
            pred_disease = pred_points[identification]['disease']
            
            if '-' in identification:
                disc_point_true.append(1)
            else:
                vertebra_point_true.append(1)

            if distance(gt_coord, pred_coord, pixel_spacing) <= config.max_dist:
                if '-' in identification: # disc
                    disc_point_pred.append(1)

                    disc_cls_pred.append(pred_disease.item())
                    disc_cls_true.append(gt_disease)
                else: # vertebra
                    vertebra_point_pred.append(1)

                    vertebra_cls_pred.append(pred_disease.item())
                    vertebra_cls_true.append(gt_disease)
            else:
                if '-' in identification:
                    disc_point_pred.append(0)
                else:
                    vertebra_point_pred.append(0)
                            
    metric_dict = {
        'disc_kp_recall': 1.0 * sum(disc_point_pred) / sum(disc_point_true),
        'vertebra_kp_recall': 1.0 * sum(vertebra_point_pred) / sum(vertebra_point_true),
        'disc_cls_auc': compute_auc(disc_cls_true, disc_cls_pred) if len(set(disc_cls_true)) == 2 else 0.0,
        'vertebra_cls_auc': compute_auc(vertebra_cls_true, vertebra_cls_pred) if len(set(vertebra_cls_true)) == 2 else 0.0,
    }    
    metric_dict['kp_recall'] = (metric_dict['disc_kp_recall'] + metric_dict['vertebra_kp_recall']) / 2.0
    metric_dict['cls_auc'] = (metric_dict['disc_cls_auc'] + metric_dict['vertebra_cls_auc']) / 2.0
    metric_dict['score'] = metric_dict['kp_recall'] * metric_dict['cls_auc']
    return metric_dict           

def evaluate(config, model, studies, eval_annotations, trans_model=None, class_tensors=None):
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
        prediction, _, cls_prediction = ccf_decoder_prob(
            config, heatmaps[0], offymaps[0], offxmaps[0], clssmaps[0], maskmap=None, tensor=True)
        height_ratio = config.sagittal_size[0] / kp_frame.size[1]
        width_ratio = config.sagittal_size[1] / kp_frame.size[0]
        ratio = torch.tensor([width_ratio, height_ratio], device=prediction.device)
        prediction = (prediction / ratio).round().float()
        annotation = gen_annotation_prob(config, study, prediction, cls_prediction)
        annotations.append(annotation)

    predictions = annotations
    predictions = format_annotation_prob(predictions)
    
    # with open(annotation_path, 'r') as file:
    #     annotations = json.load(file)
    annotations = format_annotation_prob(eval_annotations)

    metric_dict = compute_my_metric_prob(config, studies, predictions, annotations)
    
    return metric_dict

def train(config, epoch, model, loader, criterion, params, optimizer=None, train=True):
    model.train() if train else model.eval()
    epoch_records = {'time': []}
    num_batchs = len(loader)
    v_loss_weight, d_loss_weight, d_v5_loss_weight = params['v_loss_weight'], params['d_loss_weight'], params['d_v5_loss_weight']
    print('v loss weight: {} | d loss weight: {} | d v5 loss weight: {}'.format(v_loss_weight, d_loss_weight, d_v5_loss_weight))
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
                    v_loss_weight, d_loss_weight, d_v5_loss_weight, name='predict')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                predictions = model(images, heatmaps=heat_targets, offymaps=offy_targets, offxmaps=offx_targets)
            heatmap, offymap, offxmap, clssmap = predictions['result']
            loss, info = criterion(heatmap, offymap, offxmap, clssmap,
                    heat_targets, offy_targets, offx_targets, clss_targets, masks, 
                    v_loss_weight, d_loss_weight, d_v5_loss_weight, name='predict')

        for key, value in info.items():
            if key not in epoch_records: epoch_records[key] = []
            epoch_records[key].append(value)
        epoch_records['time'].append(time.time() - start_time)

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
        params['d_v5_loss'].append(np.mean(epoch_records['predict_d_v5_cls_loss']))

    return np.mean(epoch_records['loss']), params, epoch_records

def train_valid(config, k, train_k_studies, train_k_annotation, valid_k_studies, valid_k_annotation):
    print('Training Fold: {}'.format(k))
    save_model_file = os.path.join(config.kfold_model_dir, '{}.pth.tar'.format(k))
    v_numbers, d_numbers, d_v5_numbers, v_weights, d_weights, d_v5_weights = calc_weights(train_k_annotation)
    config.v_numbers = v_numbers
    config.d_numbers = d_numbers
    config.d_v5_numbers = d_v5_numbers
    config.v_weights = v_weights
    config.d_weights = d_weights
    config.d_v5_weights = d_v5_weights

    model = get_ccfnet(config).to(config.device)
    compute_params_flops(config, model)
    criterion = CCFLoss(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=5e-5)
    train_k_dataset = SpineDataSet(config, train_k_studies, train_k_annotation)
    train_k_loader = DataLoader(train_k_dataset, num_workers=config.num_workers,
                              batch_size=config.batch_size, shuffle=True, pin_memory=True,
                              collate_fn=train_k_dataset.collate_fn)

    best_metric_dict = None
    params = {'v_loss': [], 'd_loss': [], 'd_v5_loss': [],
           'v_loss_weight': config.class_weight, 'd_loss_weight': config.class_weight, 'd_v5_loss_weight': config.class_weight}
    for epoch in range(config.epochs):
        start_time = time.time()
        _, _, epoch_records = train(config, epoch, model, train_k_loader, criterion, params, optimizer=optimizer, train=True)
        metric_dict = evaluate(config, model, valid_k_studies, valid_k_annotation)
        if best_metric_dict is None or np.isnan(metric_dict['score']) or metric_dict['score'] >= best_metric_dict['score']:
            best_metric_dict = metric_dict

        config.logger.info(f'Epoch={epoch}')
        config.logger.info('=' * 50 + 'valid' + '=' * 50)
        for key, value in metric_dict.items():
            config.logger.info('valid_'+key+': '+str(round(value, 4)))
        config.logger.info('=' * 50 + 'best' + '=' * 51)
        for key, value in best_metric_dict.items():
            config.logger.info('best_'+key+': '+str(round(value, 4)))
        config.logger.info('=' * 105)
        config.logger.info('')

        lr_scheduler.step()
        print('=' * 60 + ' {:.4f} '.format(time.time() - start_time) + '=' * 60)
    torch.save(model.state_dict(), save_model_file)
    print('saved model: {}'.format(save_model_file))

    torch.cuda.empty_cache()

    return save_model_file, epoch_records

def main():
    set_seed(args.seed)
    config.train_data_dir = args.train_data_dir
    config.train_json_file = args.train_json_file
    config.valid_data_dir = args.valid_data_dir
    config.valid_json_file = args.valid_json_file

    config.output_dir = os.path.join('output', 'ours_resnet18_msc', str(args.fold_id))
    os.makedirs(config.output_dir, exist_ok=True)
    config.logger = build_logging(os.path.join(config.output_dir, 'log.log'))
    
    (train_studies, train_annotation), \
    (valid_studies, valid_annotation) = prepare_data()
    
    train_valid(config, 0, train_studies, train_annotation, valid_studies, valid_annotation)

if __name__ == '__main__':
    main()
