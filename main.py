# Author: Ning Li, contact: edwardln@stu.xjtu.edu.cn or l.ning@itv.rwth-aachen.de

import os
import time
import math
import torch
from torch import nn
from torch.utils import data
import numpy as np
import optuna
import random
from sklearn.model_selection import StratifiedKFold, KFold
from collections import OrderedDict
from utils.tools import *

R = 8.31446261815324  # J/(mol*K) from the 26th General Conference on Weights and Measures (CGPM)
R_ = 82.057338  # atm*cm3/(mol*K). CHEMKIN uses cgs unit.
cal = 4.184  # CHEMKIN uses the thermal calorie, 1 cal = 4.184 Joules
lnA_min, lnA_max = -10, 130
n_min, n_max = -11, 5
E_min, E_max = -5000, 105000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(self, drop_out=0.4, num_hidden_layers=3, num_neurons=256, num_input=256, num_out=3, active='ReLu'):
        super().__init__()
        if active == 'Sigmoid':
            act_f = nn.Sigmoid()
        elif active == 'Tanh':
            act_f = nn.Tanh()
        elif active == 'ReLu':
            act_f = nn.ReLU()
        else:
            act_f = nn.ReLU()
        # The order of the four is fixed
        self.net = nn.Sequential(OrderedDict([
            ('0_fc', nn.Linear(num_input, num_neurons)),
            ('0_BN', nn.BatchNorm1d(num_neurons)),
            ('0_ReLu', act_f),
            ('0_dropout', nn.Dropout(drop_out))
        ]))
        for i in range(num_hidden_layers - 1):
            self.net.add_module('{}_fc'.format(i + 1), nn.Linear(num_neurons, num_neurons))
            self.net.add_module('{}_BN'.format(i + 1), nn.BatchNorm1d(num_neurons))
            self.net.add_module('{}_ReLu'.format(i + 1), act_f)
            self.net.add_module('{}_dropout'.format(i + 1), nn.Dropout(drop_out))
        self.net.add_module('out_layer', nn.Linear(num_neurons, num_out))
        # self.net.apply(self.init_weights)

    # define forward function
    def forward(self, X):
        return self.net(X)


def weight_init(m):
    # for [lnA, n, E], best std=0.01
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


class JointLoss(nn.Module):
    # reduction='mean'
    def __init__(self, w_lnA=1., w_n=1., w_E=1., w_lnk=1., trans=False):
        super(JointLoss, self).__init__()
        self.beta = self.cal_beta()  # used to cal a series of lnk
        # If replacing the data set, please recalculate the next four parameters
        self.avg_lnA = 25.84
        self.avg_n = 1.97
        self.avg_E = 19860
        self.avg_lnk = 19.15
        self.w_lnk = w_lnk
        self.ane = torch.tensor([w_lnA / self.avg_lnA ** 2, w_n / self.avg_n ** 2, w_E / self.avg_E ** 2],
                                device=DEVICE)
        self.transpose = trans
        self.max_min = torch.tensor([[lnA_max - lnA_min, 0, 0],
                                     [0, n_max - n_min, 0],
                                     [0, 0, E_max - E_min]], dtype=torch.float32, device=DEVICE)
        self.min_ = torch.tensor([lnA_min, n_min, E_min], dtype=torch.float32, device=DEVICE)

    def cal_beta(self):
        beta = torch.tensor([[], [], []], dtype=torch.float32, device=DEVICE)
        for T_r in range(5, 21):  # temperature range of 500-2000 K
            T = int(10000 / T_r)
            beta = torch.cat(
                [beta, torch.tensor([[1.], [math.log(T)], [-cal / (R * T)]], dtype=torch.float32, device=DEVICE)],
                dim=1)
        return beta

    def cal_lnk(self, d):
        return torch.matmul(d, self.beta)

    def trans(self, xx):
        return torch.matmul(xx, self.max_min) + self.min_

    def forward(self, x, y):
        if self.transpose:
            x = self.trans(x)
            y = self.trans(y)
        ane_loss = torch.matmul(torch.mean((x - y) ** 2, dim=0), self.ane)
        lnk_loss = torch.mean((self.cal_lnk(x) - self.cal_lnk(y)) ** 2) / (self.avg_lnk ** 2)
        Joint_Loss = ane_loss + self.w_lnk * lnk_loss
        return Joint_Loss


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train(net, data_iter, loss_func, updater):
    """train a model for one epoch"""
    if isinstance(net, torch.nn.Module):
        net.train()  # set to train model
    metric = Accumulator(2)

    for X, y in data_iter:
        if len(X) == 1:  # 'BatchNorm' is applied, which requires more than 1 sample per iteration
            continue
        # calculate gradients and update params
        y_hat = net(X)
        loss = loss_func(y_hat, y)
        updater.zero_grad()
        loss.backward()
        updater.step()
        metric.add(float(loss) * y.numel(), y.numel())

    # return average train loss
    return metric[0] / metric[1]


def evaluate(net, data_iter, loss_func):
    """evaluate test loss"""
    net.eval()  # set to evaluate model
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(float(loss_func(net(X), y)) * y.numel(), y.numel())
    # return average test loss
    return metric[0] / metric[1]


def validate(net, data_iter, rxn, rxnfp):
    # set to evaluate model
    net.eval()

    # reproduce reaction from rxnfp
    fp2rxn_dic = {}
    for i in range(len(rxn)):
        temp = ''
        for r in rxnfp[i]:
            temp += '{:.4f}'.format(r)
        fp2rxn_dic[temp] = rxn[i]

    # evaluate model
    reactions, true_values, pred_values = [], [], []
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            y_hat = de_normalize_lnA_n_E(y_hat)
            y = de_normalize_lnA_n_E(y)
            for j in X.tolist():
                temp = ''
                for jj in j:
                    temp += '{:.4f}'.format(jj)
                reactions.append(fp2rxn_dic[temp])
            true_values.extend(y.tolist())
            pred_values.extend(y_hat.tolist())
    return reactions, true_values, pred_values


def validate_model_B(net, data_iter, rxn, rxnfp):
    # set to evaluate model
    net.eval()

    # reproduce reaction from rxnfp
    fp2rxn_dic = {}
    for i in range(len(rxn)):
        temp = ''
        for r in rxnfp[i]:
            temp += '{:.4f}'.format(r)
        fp2rxn_dic[temp] = rxn[i]

    rxns, temps, truth, pred = [], [], [], []
    # evaluate model
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = net(X)
            for j in X.tolist():
                temp = ''
                for jj in j:
                    temp += '{:.4f}'.format(jj)
                rxns.append(fp2rxn_dic[temp])
                temps.append(1000 / float(j[-1]))
            truth.extend(y.tolist())
            pred.extend(y_hat.tolist())
    truth = [x[0] for x in truth]  # torch.reshape
    pred = [x[0] for x in pred]  # torch.reshape
    return rxns, temps, truth, pred


def fitting_model_B(rxns, temps, truth, pred):
    ans = {}
    for i in range(len(rxns)):
        if rxns[i] not in ans:
            ans[rxns[i]] = {'T': [temps[i]], 'true': [truth[i]], 'pred': [pred[i]]}
        else:
            ans[rxns[i]]['T'].append(temps[i])
            ans[rxns[i]]['true'].append(truth[i])
            ans[rxns[i]]['pred'].append(pred[i])

    reactions, true_values, pred_values = [], [], []
    for rxn in ans:
        lnA, n, E, _ = fitting_Arrhenius(ans[rxn]['T'], ans[rxn]['true'])
        lnA_, n_, E_, _ = fitting_Arrhenius(ans[rxn]['T'], ans[rxn]['pred'])
        reactions.append(rxn)
        true_values.append([lnA, n, E])
        pred_values.append([lnA_, n_, E_])

    return reactions, true_values, pred_values


def normalize_lnA_n_E(database, uf=None):
    # uf: uncertainty factor of k, max(pred/true, true/pred)
    database['lnA_n_E'] = []
    for i in range(len(database['A'])):
        A, n, E = database['A'][i], database['n'][i], database['E'][i]
        if uf:
            A *= random.uniform(1 / uf, uf)
        norm_lnA = (math.log(A) - lnA_min) / (lnA_max - lnA_min)
        norm_n = (n - n_min) / (n_max - n_min)
        norm_E = (E - E_min) / (E_max - E_min)
        database['lnA_n_E'].append([norm_lnA, norm_n, norm_E])
    database['lnA_n_E'] = np.array(database['lnA_n_E'])
    return database


def de_normalize_lnA_n_E(x):
    max_min = torch.tensor([[lnA_max - lnA_min, 0, 0],
                            [0, n_max - n_min, 0],
                            [0, 0, E_max - E_min]], dtype=torch.float32, device=DEVICE)
    min_ = torch.tensor([lnA_min, n_min, E_min], dtype=torch.float32, device=DEVICE)
    return torch.matmul(x, max_min) + min_


def generate_T_points(database):
    rxn, rxnfp, rate, rc = [], [], [], []
    for i in range(len(database['rxnfp'])):
        A, n, E = database['A'][i], database['n'][i], database['E'][i]
        for T_r in range(5, 21):  # temperature range of 500-2000 K
            T_r /= 10
            rxn.append(database['rxn'][i])
            new_rxnfp = database['rxnfp'][i].tolist()
            new_rxnfp.append(T_r)
            rxnfp.append(new_rxnfp)
            rate.append([cal_lnk(math.log(A), n, E, 1000/T_r)])
            rc.append(database['rc'][i])
    database['rxn'] = np.array(rxn)
    database['rxnfp'] = np.array(rxnfp)
    database['rc'] = np.array(rc)
    database['rate'] = np.array(rate)
    return database


def split_sp(np_data):
    # return index for each species
    ans = {}
    for i in range(len(np_data['sub_mech'])):
        if np_data['sub_mech'][i] not in ans:
            ans[np_data['sub_mech'][i]] = [i]
        else:
            ans[np_data['sub_mech'][i]].append(i)
    return ans


def model_A_train(drop_out=0.0465, num_hl=5, num_neurons=91, active_func='ReLu',
                  batch_size=128, lr=9.25e-05, weight_decay=0.05, data_input=None):
    features, targets, labels = data_input['rxnfp'], data_input['lnA_n_E'], data_input['rc']
    ans = {'rxn': [], 'true': [], 'pred': [], 'loss': []}  # record predictions in each fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)  # Stratified K-Fold split
    for i, (train_index, test_index) in enumerate(skf.split(features, labels)):
        train_features = torch.tensor(features[train_index]).to(torch.float32).to(DEVICE)
        train_labels = torch.tensor(targets[train_index]).to(torch.float32).to(DEVICE)
        test_features = torch.tensor(features[test_index]).to(torch.float32).to(DEVICE)
        test_labels = torch.tensor(targets[test_index]).to(torch.float32).to(DEVICE)

        train_iter = data.DataLoader(data.TensorDataset(train_features, train_labels), batch_size, shuffle=True)
        test_iter = data.DataLoader(data.TensorDataset(test_features, test_labels), batch_size, shuffle=False)

        # define model
        # The model must be redefined in each fold
        model = MLP(drop_out=drop_out, num_hidden_layers=num_hl, num_neurons=num_neurons, active=active_func)
        model.to(DEVICE)
        model.apply(weight_init)

        # define hyperparameters
        num_epochs = 1000
        loss = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # train model
        best_loss = 1.0
        for epoch in range(num_epochs):
            train_loss = train(model, train_iter, loss, optimizer)
            test_loss = evaluate(model, test_iter, loss)
            print('{:<2d}\t{:<4d}\t{:.4f}\t{:.4f}\t{:.2e}'.format(i+1, epoch + 1, train_loss, test_loss, lr))
            if (epoch > 200) and (test_loss < best_loss):
                torch.save(model.state_dict(), r'.\models\model_A\fold_{}.pth'.format(i+1))
                best_loss = test_loss
        ans['loss'].append(best_loss)
        # print('\nFold {}: best test loss is {:.5f}\n'.format(i+1, best_loss))

        # validate
        model.load_state_dict(torch.load(r'.\models\model_A\fold_{}.pth'.format(i+1)))
        rxn, xx, yy = validate(model, test_iter, data_input['rxn'], features)
        ans['rxn'].extend(rxn)
        ans['true'].extend(xx)
        ans['pred'].extend(yy)
    # score(ans['rxn'], ans['true'], ans['pred'], print_info=True, save_file=os.path.abspath(r'.\docs\score_A.txt'))
    print(ans['loss'])
    return ans


def model_B_train(drop_out=0.0465, num_hl=5, num_neurons=91, active_func='ReLu',
                  batch_size=128, lr=9.25e-05, weight_decay=0.05, data_input=None):

    ans = {'rxn': [], 'T': [], 'true': [], 'pred': [], 'loss': []}  # record predictions in each fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)  # Stratified K-Fold split
    for i, (train_index, test_index) in enumerate(skf.split(data_input['rxnfp'], data_input['rc'])):
        # The reaction must first be split and then generate sampling points at different temperatures
        OME_train, OME_test = {}, {}
        for k in data_input:
            OME_train[k] = data_input[k][train_index]
            OME_test[k] = data_input[k][test_index]
        OME_train = generate_T_points(OME_train)
        OME_test = generate_T_points(OME_test)
        train_features, train_labels = OME_train['rxnfp'], OME_train['rate']
        test_features, test_labels = OME_test['rxnfp'], OME_test['rate']

        train_features = torch.tensor(train_features).to(torch.float32).to(DEVICE)
        train_labels = torch.tensor(train_labels).to(torch.float32).to(DEVICE)
        test_features = torch.tensor(test_features).to(torch.float32).to(DEVICE)
        test_labels = torch.tensor(test_labels).to(torch.float32).to(DEVICE)

        train_iter = data.DataLoader(data.TensorDataset(train_features, train_labels), batch_size, shuffle=True)
        test_iter = data.DataLoader(data.TensorDataset(test_features, test_labels), batch_size, shuffle=False)

        # define model
        # The model must be redefined in each fold
        model = MLP(drop_out=drop_out, num_hidden_layers=num_hl, num_neurons=num_neurons, active=active_func,
                    num_input=257, num_out=1)
        model.to(DEVICE)
        model.apply(weight_init)

        # define hyperparameters
        num_epochs = 1000
        loss = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # train model
        best_loss = 2000.
        for epoch in range(num_epochs):
            train_loss = train(model, train_iter, loss, optimizer)
            test_loss = evaluate(model, test_iter, loss)
            print('{:<2d}\t{:<4d}\t{:.4f}\t{:.4f}\t{:.2e}'.format(i+1, epoch + 1, train_loss, test_loss, lr))
            if (epoch > 200) and (test_loss < best_loss):
                torch.save(model.state_dict(), r'.\models\model_B\fold_{}.pth'.format(i+1))
                best_loss = test_loss
        ans['loss'].append(best_loss)
        # print('\nFold {}: best test loss is {:.5f}\n'.format(i+1, best_loss))

        # validate
        model.load_state_dict(torch.load(r'.\models\model_B\fold_{}.pth'.format(i+1)))
        rxns, temps, xx, yy = validate_model_B(model, test_iter, OME_test['rxn'], OME_test['rxnfp'])
        ans['T'].extend(temps)
        ans['rxn'].extend(rxns)
        ans['true'].extend(xx)
        ans['pred'].extend(yy)

    ans['rxn'], ans['true'], ans['pred'] = fitting_model_B(ans['rxn'], ans['T'], ans['true'], ans['pred'])
    # score(ans['rxn'], ans['true'], ans['pred'], print_info=True, save_file=os.path.abspath(r'.\docs\score_B.txt'))
    print(ans['loss'])
    return ans


def model_C_train(drop_out=0.0465, num_hl=5, num_neurons=91, active_func='ReLu',
                  batch_size=128, lr=9.25e-05, weight_decay=0.05, data_input=None):
    features, targets, labels = data_input['rxnfp'], data_input['lnA_n_E'], data_input['rc']
    ans = {'rxn': [], 'true': [], 'pred': [], 'loss': []}  # record predictions in each fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)  # Stratified K-Fold split
    for i, (train_index, test_index) in enumerate(skf.split(features, labels)):
        train_features = torch.tensor(features[train_index]).to(torch.float32).to(DEVICE)
        train_labels = torch.tensor(targets[train_index]).to(torch.float32).to(DEVICE)
        test_features = torch.tensor(features[test_index]).to(torch.float32).to(DEVICE)
        test_labels = torch.tensor(targets[test_index]).to(torch.float32).to(DEVICE)

        train_iter = data.DataLoader(data.TensorDataset(train_features, train_labels), batch_size, shuffle=True)
        test_iter = data.DataLoader(data.TensorDataset(test_features, test_labels), batch_size, shuffle=False)

        # define model
        # The model must be redefined in each fold
        model = MLP(drop_out=drop_out, num_hidden_layers=num_hl, num_neurons=num_neurons, active=active_func)
        model.to(DEVICE)
        model.apply(weight_init)
        # model.load_state_dict(torch.load(os.path.abspath(r'.\models\lnk_1000.pth')))

        # define hyperparameters
        num_epochs = 1000
        # loss = nn.MSELoss(reduction='mean')
        loss = JointLoss(w_lnA=0.05, w_n=0.05, w_E=0.2, w_lnk=0.7, trans=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # train model
        best_loss = 10.0
        for epoch in range(num_epochs):
            train_loss = train(model, train_iter, loss, optimizer)
            test_loss = evaluate(model, test_iter, loss)
            print('{:<2d}\t{:<4d}\t{:.4f}\t{:.4f}\t{:.2e}'.format(i+1, epoch + 1, train_loss, test_loss, lr))
            if (epoch > 200) and (test_loss < best_loss):
                torch.save(model.state_dict(), r'.\models\model_C\fold_{}.pth'.format(i+1))
                best_loss = test_loss
        ans['loss'].append(best_loss)
        # print('\nFold {}: best test loss is {:.5f}\n'.format(i+1, best_loss))

        # validate
        model.load_state_dict(torch.load(r'.\models\model_C\fold_{}.pth'.format(i+1)))
        rxn, xx, yy = validate(model, test_iter, data_input['rxn'], features)
        ans['rxn'].extend(rxn)
        ans['true'].extend(xx)
        ans['pred'].extend(yy)
    # score(ans['rxn'], ans['true'], ans['pred'], print_info=True, save_file=os.path.abspath(r'.\docs\score_C_test.txt'))
    print(ans['loss'])
    return ans


def model_C_species(drop_out=0.0465, num_hl=5, num_neurons=91, active_func='ReLu',
                    batch_size=128, lr=9.25e-05, weight_decay=0.05, data_input=None):
    species_index = split_sp(data_input)
    features, targets, labels = data_input['rxnfp'], data_input['lnA_n_E'], data_input['rc']
    ans = {'rxn': [], 'true': [], 'pred': [], 'loss': []}  # record predictions in each fold
    for spec in species_index:
        train_index, test_index = [], np.array(species_index[spec])
        for s in species_index:
            if s != spec:
                train_index += species_index[s]
        train_index = np.array(train_index)

        train_features = torch.tensor(features[train_index]).to(torch.float32).to(DEVICE)
        train_labels = torch.tensor(targets[train_index]).to(torch.float32).to(DEVICE)
        test_features = torch.tensor(features[test_index]).to(torch.float32).to(DEVICE)
        test_labels = torch.tensor(targets[test_index]).to(torch.float32).to(DEVICE)

        train_iter = data.DataLoader(data.TensorDataset(train_features, train_labels), batch_size, shuffle=True)
        test_iter = data.DataLoader(data.TensorDataset(test_features, test_labels), batch_size, shuffle=False)

        # define model
        model = MLP(drop_out=drop_out, num_hidden_layers=num_hl, num_neurons=num_neurons, active=active_func)
        model.to(DEVICE)
        model.apply(weight_init)
        # model.load_state_dict(torch.load(os.path.abspath(r'.\models\lnk_1000.pth')))

        # define hyperparameters
        num_epochs = 1000
        # loss = nn.MSELoss(reduction='mean')
        loss = JointLoss(w_lnA=0.05, w_n=0.05, w_E=0.2, w_lnk=0.7, trans=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # train model
        best_loss = 1.0
        for epoch in range(num_epochs):
            train_loss = train(model, train_iter, loss, optimizer)
            test_loss = evaluate(model, test_iter, loss)
            print('{:<7s}\t{:<4d}\t{:.4f}\t{:.4f}\t{:.2e}'.format(spec, epoch + 1, train_loss, test_loss, lr))
            if (epoch > 200) and (test_loss < best_loss):
                torch.save(model.state_dict(), r'.\models\model_C\fold_{}.pth'.format(spec))
                best_loss = test_loss
        ans['loss'].append(best_loss)
        # print('\nFold {}: best test loss is {:.5f}\n'.format(i+1, best_loss))

        # validate
        model.load_state_dict(torch.load(r'.\models\model_C\fold_{}.pth'.format(spec)))
        rxn, xx, yy = validate(model, test_iter, data_input['rxn'], features)
        ans['rxn'].extend(rxn)
        ans['true'].extend(xx)
        ans['pred'].extend(yy)
        # score(rxn, xx, yy, print_info=False, save_file=os.path.abspath(r'.\docs\score_C_{}.txt'.format(spec)))
    # score(ans['rxn'], ans['true'], ans['pred'], print_info=True, save_file=os.path.abspath(r'.\docs\score_C_spec.txt'))
    print(ans['loss'])
    return ans


def objective(trail):
    num_hl = trail.suggest_int('hidden_layers', 2, 6)
    num_neurons = trail.suggest_int('neurons', 64, 256)
    # active = trail.suggest_categorical('active_func', ['ReLu', 'Sigmoid', 'Tanh'])
    active = 'ReLu'
    drop_out = trail.suggest_float('drop_out', 0.0, 0.8, step=0.01)
    lr = trail.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
    batch_size = trail.suggest_categorical('batch_size', [32, 64, 128, 256])
    # optima = trail.suggest_categorical('optimizer', ['Adam', 'SGD'])
    weight_decay = trail.suggest_float('weight_decay', 0, 1, step=0.01)

    # load database and normalization
    dataset_source = torch.load(r'.\data\database.npy')
    dataset_normalized = normalize_lnA_n_E(dataset_source)
    # ans = model_A_train(drop_out, num_hl, num_neurons, active, batch_size, lr, weight_decay, dataset_normalized)
    # ans = model_B_train(drop_out, num_hl, num_neurons, active, batch_size, lr, weight_decay, dataset_source)
    ans = model_C_train(drop_out, num_hl, num_neurons, active, batch_size, lr, weight_decay, dataset_normalized)
    return sum(ans['loss']) / len(ans['loss'])


def train_model(model='C', ensemble_learning=10, save_evaluation=None, save_predictions=None):
    """
    :param model: A, B or C
    :param ensemble_learning: number of ensemble models. if = 1, no ensemble learning.
    :param save_evaluation: file path to save the evaluation result
    :param save_predictions: save the source data of model predictions
    :return:
    """
    # load database and normalization
    dataset_source = torch.load(r'.\data\database.npy')
    dataset_normalized = normalize_lnA_n_E(dataset_source)
    ans = {}
    for i in range(ensemble_learning):
        print('Running resemble learning on model {}, {:0>2d}'.format(model, i + 1))
        if model == 'A':
            tmp = model_A_train(drop_out=0.0465, num_hl=5, num_neurons=91, active_func='ReLu', batch_size=128,
                                lr=9.25e-05, weight_decay=0.05, data_input=dataset_normalized)
        elif model == 'B':
            tmp = model_B_train(drop_out=0.1, num_hl=4, num_neurons=196, active_func='ReLu', batch_size=64, lr=3.2e-04,
                                weight_decay=0.18, data_input=dataset_source)  # cross-validation for B
        elif model == 'C':
            tmp = model_C_train(drop_out=0.13, num_hl=6, num_neurons=241, active_func='ReLu', batch_size=128,
                                lr=1.065e-04, weight_decay=0.63, data_input=dataset_normalized)
        elif model == 'species':
            tmp = model_C_species(drop_out=0.13, num_hl=6, num_neurons=241, active_func='ReLu', batch_size=128,
                                  lr=1.065e-04, weight_decay=0.63, data_input=dataset_normalized)
        else:
            print('Incorrect model. Please choose from A, B or C')
            return 0
        score(tmp['rxn'], tmp['true'], tmp['pred'], save_file=r'.\evaluation\score_{}_{:0>2d}-spec.txt'.format(model, i + 1))
        torch.save(tmp, r'.\evaluation\model_{}_tmp_{:0>2d}-spec.pth'.format(model, i + 1))
        ans = merge_socre(tmp, ans)
    ans['true'] /= ensemble_learning
    ans['pred'] /= ensemble_learning

    if not save_evaluation:
        save_evaluation = os.path.abspath(r'.\evaluation\Model_{}_evaluation-species.txt'.format(model))
    score(ans['rxn'], ans['true'], ans['pred'], print_info=True, save_file=save_evaluation)

    if not save_predictions:
        save_predictions = os.path.abspath(r'.\evaluation\Model_{}_source-species.pth'.format(model))
    torch.save(ans, save_predictions)

    return ans


if __name__ == '__main__':
    """
    # using Optuna to optimize hyperparameters
    study = optuna.create_study(study_name='Model_A', direction='minimize')
    study.optimize(objective, n_trials=100)

    print('Optimization completed, result:')
    print(study.best_params)
    print(study.best_trial.value)

    optuna.visualization.plot_optimization_history(study).show()
    # optuna.visualization.plot_intermediate_values(study).show()
    optuna.visualization.plot_slice(study).show()
    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_contour(study).show()
    optuna.visualization.plot_parallel_coordinate(study).show()

    """
    train_model(model='C', ensemble_learning=10)
