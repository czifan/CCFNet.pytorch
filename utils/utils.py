from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import json
import torch
from typing import Dict, Tuple
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 5000)
import os
import logging
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from copy import deepcopy

def build_logging(filename):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=filename,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logger = logging.getLogger('')
    logger.addHandler(console)
    return logger

def str2tensor(s: str) -> torch.Tensor:
    return torch.Tensor(list(map(float, s.split('\\'))))

def unit_vector(tensor: torch.Tensor, dim=-1):
    norm = (tensor ** 2).sum(dim=dim, keepdim=True).sqrt()
    return tensor / norm

def unit_normal_vector(orientation: torch.Tensor):
    temp1 = orientation[:, [1, 2, 0]]
    temp2 = orientation[:, [2, 0, 1]]
    output = temp1 * temp2[[1, 0]]
    output = output[0] - output[1]
    return unit_vector(output, dim=-1)

def lazy_property(func):
    attr_name = "_lazy_" + func.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return _lazy_property

def read_annotation(config, path) -> Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]]:
    with open(path, 'r') as annotation_file:
        non_hit_count = {}
        annotation = {}
        for x in json.load(annotation_file):
            study_uid = x['studyUid']

            # assert len(x['data']) == 1, (study_uid, len(x['data']))
            data = x['data'][0]
            instance_uid = data['instanceUid']
            series_uid = data['seriesUid']

            # assert len(data['annotation']) == 1, (study_uid, len(data['annotation']))
            points = data['annotation'][0]['data']['point']

            vertebra_label = torch.full([len(config.SPINAL_VERTEBRA_ID), 3],
                                        config.PADDING_VALUE, dtype=torch.long)
            disc_label = torch.full([len(config.SPINAL_DISC_ID), 3],
                                    config.PADDING_VALUE, dtype=torch.long)
            disc_v5_label = torch.full([len(config.SPINAL_DISC_ID), 3],
                                       0, dtype=torch.long)
            for point in points:
                identification = point['tag']['identification']
                if identification in config.SPINAL_VERTEBRA_ID:
                    position = config.SPINAL_VERTEBRA_ID[identification]
                    diseases = point['tag']['vertebra']

                    vertebra_label[position, :2] = torch.tensor(point['coord'])
                    for disease in diseases.split(','):
                        if disease in config.SPINAL_VERTEBRA_DISEASE_ID:
                            disease = config.SPINAL_VERTEBRA_DISEASE_ID[disease]
                            vertebra_label[position, 2] = disease
                elif identification in config.SPINAL_DISC_ID:
                    position = config.SPINAL_DISC_ID[identification]
                    diseases = point['tag']['disc']

                    disc_label[position, :2] = torch.tensor(point['coord'])
                    disc_v5_label[position, :2] = torch.tensor(point['coord'])
                    for disease in diseases.split(','):
                        if disease != 'v5' and disease in config.SPINAL_DISC_DISEASE_ID:
                            disease = config.SPINAL_DISC_DISEASE_ID[disease]
                            disc_label[position, 2] = disease
                        elif disease == 'v5' and disease in config.SPINAL_DISC_DISEASE_ID:
                            disease = config.SPINAL_DISC_DISEASE_ID[disease]
                            disc_v5_label[position, 2] = disease
                elif identification in non_hit_count:
                    non_hit_count[identification] += 1
                else:
                    non_hit_count[identification] = 1

            annotation[study_uid, series_uid, instance_uid] = vertebra_label, disc_label, disc_v5_label
    if len(non_hit_count) > 0:
        print(non_hit_count)
    return annotation

def cal_metrics(confusion_matrix: pd.DataFrame):
    key_point_recall = confusion_matrix.iloc[:-2].sum().sum() / confusion_matrix.sum().sum()
    precision = {col: confusion_matrix.loc[col, col] / confusion_matrix.loc[col].sum() for col in confusion_matrix}
    recall = {col: confusion_matrix.loc[col, col] / confusion_matrix[col].sum() for col in confusion_matrix}
    f1 = {col: 2 * precision[col] * recall[col] / (precision[col] + recall[col]) for col in confusion_matrix}
    macro_f1 = sum(f1.values()) / len(f1)

    columns = confusion_matrix.columns
    recall_true_point = {col: confusion_matrix.loc[col, col] / confusion_matrix.loc[columns, col].sum()
                         for col in confusion_matrix}
    f1_true_point = {col: 2 * precision[col] * recall_true_point[col] / (precision[col] + recall_true_point[col])
                     for col in confusion_matrix}
    macro_f1_true_point = sum(f1_true_point.values()) / len(f1)
    output = [('macro f1', macro_f1), ('key point recall', key_point_recall),
              ('macro f1 (true point)', macro_f1_true_point)]
    output += sorted([(k + ' f1 (true point)', v, precision[k], recall[k]) for k, v in f1_true_point.items()],
                     key=lambda x: x[0])
    return output

def gen_annotation(config, study, prediction, cls_prediction, d_v5_cls_prediction):
    z_index = study.t2_sagittal.instance_uids[study.t2_sagittal_middle_frame.instance_uid]
    point = []
    for i, (coord, identification, cls, is_disc) in enumerate(zip(
            prediction, config.identifications, cls_prediction, config.is_disc)):
        point.append({
            'coord': coord.cpu().int().numpy().tolist(),
            'tag': {
                'identification': identification,
                'disc' if is_disc else 'vertebra': 'v' + str(cls.item() + 1)
                # 'disc' if is_disc else 'vertebra': _parse_number(cls.item())
            },
            'zIndex': z_index
        })
        if is_disc and d_v5_cls_prediction[i - 5].item() > 0:
            point[-1]['tag']['disc'] = point[-1]['tag']['disc'] + ',v5'
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


def format_annotation(annotations):
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
            if disease == '':
                disease = 'v1'
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

def compute_my_metric(config, studies, predictions, annotations):
    total_points, true_points = 0, 0
    disc_4cls_dict = {'v1': 0, 'v2': 1, 'v3': 2, 'v4': 3}
    disc_4cls_pred, disc_4cls_true = [], [] # v1 v2 v3 v4
    disc_2cls_pred, disc_2cls_true = [], [] # v5
    vertebra_2cls_pred, vertebra_2cls_true = [], [] 
    
    for study_uid, annotation in annotations.items():
        study = studies[study_uid]
        pixel_spacing = study.t2_sagittal_middle_frame.pixel_spacing
        pred_points = predictions[study_uid]['annotation']
        for identification, gt_point in annotation['annotation'].items():
            gt_coord = gt_point['coord']
            gt_disease = gt_point['disease'].split(',')
            
            if identification not in pred_points:
                continue 
                
            pred_coord = pred_points[identification]['coord']
            pred_disease = pred_points[identification]['disease'].split(',')
            
            total_points += 1
            
            if distance(gt_coord, pred_coord, pixel_spacing) <= config.max_dist:
                true_points += 1
                if '-' in identification: # disc
                    tmp_pred_disease = deepcopy(pred_disease)
                    if 'v5' in pred_disease:
                        tmp_pred_disease.remove('v5')
                    tmp_gt_disease = deepcopy(gt_disease)
                    if 'v5' in gt_disease:
                        tmp_gt_disease.remove('v5')
                    if len(tmp_gt_disease) == 1:
                        disc_4cls_pred.append(disc_4cls_dict[tmp_pred_disease[0]])
                        disc_4cls_true.append(disc_4cls_dict[tmp_gt_disease[0]])
                    if 'v5' in pred_disease:
                        disc_2cls_pred.append(1)
                    else:
                        disc_2cls_pred.append(0)
                    if 'v5' in gt_disease:
                        disc_2cls_true.append(1)
                    else:
                        disc_2cls_true.append(0)
                else: # vertebra
                    if 'v1' in pred_disease:
                        vertebra_2cls_pred.append(0)
                    else:
                        vertebra_2cls_pred.append(1)
                    if 'v1' in gt_disease:
                        vertebra_2cls_true.append(0)
                    else:
                        vertebra_2cls_true.append(1)
                            
    metric_dict = {
        'keypoint_accuracy': 1.0 * true_points / total_points,
        'vertebra_f1_score': f1_score(vertebra_2cls_true, vertebra_2cls_pred),
        'disc_cls4_f1_score': f1_score(disc_4cls_true, disc_4cls_pred, average='macro'),
        'disc_cls2_f1_score': f1_score(disc_2cls_true, disc_2cls_pred),
    }
    
    metric_dict['class_avg_score'] = (metric_dict['vertebra_f1_score'] + metric_dict['disc_cls4_f1_score'] + metric_dict['disc_cls2_f1_score']) / 3.0
    metric_dict['score'] = metric_dict['keypoint_accuracy'] * metric_dict['class_avg_score']
    return metric_dict           

def confusion_matrix_with_image(config, studies, predictions, annotations) -> pd.DataFrame:
    columns = ['disc_' + k for k in config.SPINAL_DISC_DISEASE_ID]
    columns += ['vertebra_' + k for k in config.SPINAL_VERTEBRA_DISEASE_ID]
    output = pd.DataFrame(config.epsilon, columns=columns, index=columns + ['wrong', 'not_hit'])

    predictions = format_annotation(predictions)
    for study_uid, annotation in annotations.items():
        study = studies[study_uid]
        pixel_spacing = study.t2_sagittal_middle_frame.pixel_spacing
        pred_points = predictions[study_uid]['annotation']
        image = study.t2_sagittal_middle_frame.image
        plt.imshow(image, cmap='gray')
        for identification, gt_point in annotation['annotation'].items():
            gt_coord = gt_point['coord']
            gt_disease = gt_point['disease']

            if '-' in identification:
                _type = 'disc_'
            else:
                _type = 'vertebra_'

            if identification not in pred_points:
                for d in gt_disease.split(','):
                    output.loc['not_hit', _type + d] += 1
                continue

            pred_coord = pred_points[identification]['coord']
            pred_disease = pred_points[identification]['disease']
            
            #plt.text(gt_coord[0]-40, gt_coord[1], gt_disease, c='b', fontsize=10)
            #plt.text(pred_coord[0]+10, pred_coord[1], pred_disease, c='r', fontsize=10)

            if distance(gt_coord, pred_coord, pixel_spacing) >= config.max_dist:
                plt.scatter(*gt_coord, c='b')
                plt.scatter(*pred_coord, c='r')
                for d in gt_disease.split(','):
                    output.loc['wrong', _type + d] += 1
            else:
                plt.scatter(*gt_coord, c='y')
                plt.scatter(*pred_coord, c='w')
                for d in gt_disease.split(','):
                    for dp in pred_disease.split(','):
                        output.loc[_type + dp, _type + d] += 1
        plt.savefig('visualize/{}.jpg'.format(study.study_uid))
        plt.close()

    print(output)
    return output


def confusion_matrix(config, studies, predictions, annotations) -> pd.DataFrame:
    columns = ['disc_' + k for k in config.SPINAL_DISC_DISEASE_ID]
    columns += ['vertebra_' + k for k in config.SPINAL_VERTEBRA_DISEASE_ID]
    output = pd.DataFrame(config.epsilon, columns=columns, index=columns + ['wrong', 'not_hit'])

    predictions = format_annotation(predictions)
    for study_uid, annotation in annotations.items():
        study = studies[study_uid]
        pixel_spacing = study.t2_sagittal_middle_frame.pixel_spacing
        pred_points = predictions[study_uid]['annotation']
        for identification, gt_point in annotation['annotation'].items():
            gt_coord = gt_point['coord']
            gt_disease = gt_point['disease']

            if '-' in identification:
                _type = 'disc_'
            else:
                _type = 'vertebra_'

            if identification not in pred_points:
                for d in gt_disease.split(','):
                    output.loc['not_hit', _type + d] += 1
                continue

            pred_coord = pred_points[identification]['coord']
            pred_disease = pred_points[identification]['disease']
            if distance(gt_coord, pred_coord, pixel_spacing) >= config.max_dist:
                for d in gt_disease.split(','):
                    output.loc['wrong', _type + d] += 1
            else:
                for d in gt_disease.split(','):
                    for dp in pred_disease.split(','):
                        output.loc[_type + dp, _type + d] += 1
#                 for dp in pred_disease.split(','):
#                     if dp in gt_disease:
#                         output.loc[_type + dp, _type + dp] += 1
#                     else:
#                         output.loc[_type + dp, _type + gt_disease.split(',')[0]] += 1

    print(output)
    return output

def distance(coord0, coord1, pixel_spacing):
    x = (coord0[0] - coord1[0]) * pixel_spacing[0]
    y = (coord0[1] - coord1[1]) * pixel_spacing[1]
    output = math.sqrt(x ** 2 + y ** 2)
    return output