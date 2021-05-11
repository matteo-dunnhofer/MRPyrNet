import sys
sys.path.append('..')

import argparse
import os
import random
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from mrnetdataset import MRNetDataset
import tqdm
import csv
import json
import modules.utils as ut

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--path_to_data', type=str, required=True)
parser.add_argument('--path_to_models', type=str, required=True)
args = parser.parse_args()


random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def extract_predictions(task, plane, path_to_data, path_to_models, train=True):
    assert task in ['acl', 'meniscus', 'abnormal']
    assert plane in ['axial', 'coronal', 'sagittal']
    
    models = os.listdir(path_to_models)
    model_name = list(filter(lambda name: task in name and plane in name, models))[0]
    model_path = f'{path_to_models}/{model_name}'

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = torch.load(model_path)
    model = model.to(device)
    model.eval()
    
    dataset = MRNetDataset(path_to_data, 
                              task, 
                              plane, 
                              train=train,
                              transform=False)
    
    data_loader = torch.utils.data.DataLoader(dataset, 
                                               batch_size=1, 
                                               shuffle=False, 
                                               num_workers=4, 
                                               drop_last=False)
    predictions = []
    labels = []
    with torch.no_grad():
        for image, label, _ in tqdm.tqdm(data_loader):

            prediction0, prediction1, prediction2, prediction3, prediction4 = model(image.to(device))

            probas0 = torch.sigmoid(prediction0)
            probas1 = torch.sigmoid(prediction1)
            probas2 = torch.sigmoid(prediction2)
            probas3 = torch.sigmoid(prediction3)
            probas4 = torch.sigmoid(prediction4)

            probas = torch.cat([probas0[0], probas1[0], probas2[0], probas3[0], probas4[0]])
      
            labels.append(label[0].item())

            predictions.append(probas.max().item())

    return predictions, labels


final_results_val = {}

for task in ['acl', 'meniscus']:
    results = {}

    for plane in ['axial', 'coronal', 'sagittal']:
        predictions, labels = extract_predictions(task, plane, args.path_to_data, args.path_to_models)
        results['labels'] = labels
        results[plane] = predictions
      
    X = np.zeros((len(predictions), 3))
    X[:, 0] = results['axial']
    X[:, 1] = results['coronal']
    X[:, 2] = results['sagittal']

    y = np.array(labels)

    # train logistic regressor
    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(X, y)

    # test logistic regressor
    results_val = {}

    for plane in ['axial', 'coronal', 'sagittal']:
        predictions, labels = extract_predictions(task, plane, args.path_to_data, args.path_to_models, train=False)
        results_val['labels'] = labels
        results_val[plane] = predictions

    X_val = np.zeros((len(predictions), 3))
    X_val[:, 0] = results_val['axial']
    X_val[:, 1] = results_val['coronal']
    X_val[:, 2] = results_val['sagittal']

    y_val = np.array(labels)

    y_pred = logreg.predict_proba(X_val)[:, 1]
    y_class_preds = (y_pred > 0.5).astype(np.float32)
    auc = metrics.roc_auc_score(y_val, y_pred)
    print(f'{task} tear - AUC: {auc}')

    accuracy, sensitivity, specificity = ut.accuracy_sensitivity_specificity(y_val, y_class_preds)
    final_results_val[task] = [auc, accuracy, sensitivity, specificity]


exp_dir = args.path_to_models.split('/')[:-2]

with open(os.path.join(*exp_dir, 'results', f'model-results.csv'), 'w') as res_file:
    fw = csv.writer(res_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    fw.writerow(['Task', 'AUC', 'Accuracy', 'Sensitivity', 'Specificity'])
    for ck in final_results_val.keys():
        fw.writerow([f'{ck}'] + [str(val) for val in final_results_val[ck]])
