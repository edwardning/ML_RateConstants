import os
import time
from rdkit import Chem
from rdkit.Chem import Draw
import math
import numpy as np


R = 8.31446261815324  # J/(mol*K) from the 26th General Conference on Weights and Measures (CGPM)
R_ = 82.057338  # atm*cm3/(mol*K). CHEMKIN uses cgs unit.
cal = 4.184  # CHEMKIN uses the thermal calorie, 1 cal = 4.184 Joules


def generate_lnk(file, out_file):
    ans = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            if i <= 18:
                continue
            l = l.strip().split()
            rxn, lnA, n, E, lnA0, n0, E0 = l[0], float(l[1]), float(l[2]), float(l[3]), float(l[7]), float(l[8]), float(
                l[9])
            ans[rxn] = [lnA, n, E, lnA0, n0, E0]

    with open(out_file, 'w') as f:
        for T_r in range(5, 21):  # temperature range of 500-2000 K
            T = int(10000 / T_r)
            for rxn in ans:
                true_lnk = cal_lnk(ans[rxn][0], ans[rxn][1], ans[rxn][2], T)
                pred_lnk = cal_lnk(ans[rxn][3], ans[rxn][4], ans[rxn][5], T)
                f.writelines('{}\t{:.4f}\t{:.2f}\t{}\n'.format(rxn, true_lnk, pred_lnk, T))
    return ans


def draw_molecular(species):
    """
    species = {'DMM': 'InChI=1S/C3H8O2/c1-4-3-5-2/h3H2,1-2H3',
               'DME': 'InChI=1S/C2H6O/c1-3-2/h1-2H3',
               'OME2': 'InChI=1S/C4H10O3/c1-5-3-7-4-6-2/h3-4H2,1-2H3',
               'DEE': 'InChI=1S/C4H10O/c1-3-5-4-2/h3-4H2,1-2H3',
               'DEM': 'InChI=1S/C5H12O2/c1-3-6-5-7-4-2/h3-5H2,1-2H3',
               'C3H8': 'InChI=1S/C3H8/c1-3-2/h3H2,1-2H3',
               'C4H10': 'InChI=1S/C4H10/c1-3-4-2/h3-4H2,1-2H3',
               'NC5H12': 'InChI=1S/C5H12/c1-3-5-4-2/h3-5H2,1-2H3'}
    """
    for s in species:
        m = Chem.MolFromInchi(species[s])
        Draw.MolToFile(m, r'.\docs\{}.png'.format(s))
    return


def val_rc_spec(file):
    rc = {}
    sp = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            rxn = line[0]
            species = line[1]
            rxn_class = line[2]
            tr = [float(x) for x in line[3:6]]
            pr = [float(x) for x in line[6:9]]

            if species not in sp:
                sp[species] = {'truth': [tr], 'pred': [pr]}
            else:
                sp[species]['truth'].append(tr)
                sp[species]['pred'].append(pr)

            if rxn_class not in rc:
                rc[rxn_class] = {'truth': [tr], 'pred': [pr]}
            else:
                rc[rxn_class]['truth'].append(tr)
                rc[rxn_class]['pred'].append(pr)

        for s in sp:
            lnA_true, n_true, E_true = [x[0] for x in sp[s]['truth']], [x[1] for x in sp[s]['truth']], [x[2] for x in sp[s]['truth']]
            lnA_pred, n_pred, E_pred = [x[0] for x in sp[s]['pred']], [x[1] for x in sp[s]['pred']], [x[2] for x in sp[s]['pred']]
            print('Fuel {:<3d} {}: {:.4f}\t{:.2%}\t{:.4f}\t{:.2%}\t{:.4f}\t{:.2%}'.
                  format(len(lnA_true), s, RMSE(lnA_true, lnA_pred), NRMSE(lnA_true, lnA_pred),
                            RMSE(n_true, n_pred), NRMSE(n_true, n_pred),
                            RMSE(E_true, E_pred), NRMSE(E_true, E_pred),))
        print('\n')
        for s in rc:
            lnA_true, n_true, E_true = [x[0] for x in rc[s]['truth']], [x[1] for x in rc[s]['truth']], [x[2] for x in rc[s]['truth']]
            lnA_pred, n_pred, E_pred = [x[0] for x in rc[s]['pred']], [x[1] for x in rc[s]['pred']], [x[2] for x in rc[s]['pred']]
            print('Reaction Class {:<3d} {}: {:.4f}\t{:.2%}\t{:.4f}\t{:.2%}\t{:.4f}\t{:.2%}'.
                  format(len(lnA_true), s, RMSE(lnA_true, lnA_pred), NRMSE(lnA_true, lnA_pred),
                            RMSE(n_true, n_pred), NRMSE(n_true, n_pred),
                            RMSE(E_true, E_pred), NRMSE(E_true, E_pred),))

    return


def fitting_Arrhenius(temperature, lnk):
    """
    Mathematical principles referenced from CHEMRev (https://doi.org/10.1002/kin.20049)
    :param temperature:
    :param lnk:
    :return:
    """
    X = np.array([[1, math.log(t), 1/t] for t in temperature])
    Y = np.array(lnk)
    beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    lnA, n, E = list(beta)
    E = -E*R/cal  # cal/mol

    avg_err = 0
    for i, temp in enumerate(temperature):
        lnk_fitting = cal_lnk(lnA, n, E, temp)
        lnk_truth = lnk[i]
        avg_err += abs((lnk_truth - lnk_fitting) / lnk_truth)
    avg_err /= len(temperature)
    return lnA, n, E, avg_err


def r2_score(y_true, y_pred):
    y_mean = sum(y_true) / len(y_true)
    sse, sst = 0, 0
    for i in range(len(y_true)):
        sse += (y_true[i] - y_pred[i]) ** 2
        sst += (y_true[i] - y_mean) ** 2
    return 1 - sse/sst


def NRMSE(y_true, y_pred):
    abs_y = [abs(x) for x in y_true]
    return RMSE(y_true, y_pred) / (sum(abs_y) / len(abs_y))


def RMSE(y_true, y_pred):
    return MSE(y_true, y_pred) ** 0.5


def MSE(y_true, y_pred):
    ans = 0
    for i in range(len(y_true)):
        ans += (y_true[i] - y_pred[i]) ** 2
    ans /= len(y_true)
    return ans


def MPE(y_true, y_pred):
    # mean percentage error
    ans, valid_num = 0, 0
    for i in range(len(y_true)):
        if y_true[i] != 0:
            ans += abs((y_true[i] - y_pred[i]) / y_true[i])
            valid_num += 1
    ans /= valid_num
    return ans


def cal_lnk_range(lnA, n, E):
    ans = []
    for T_r in range(5, 21):  # temperature range of 500-2000 K
        T = int(10000 / T_r)
        ans.append(cal_lnk(lnA, n, E, T))
    return ans


def cal_lnk(lnA, n, E, T):
    return lnA + n * math.log(T) - (E * cal) / (R * T)


def cal_k(A, n, E, T):
    return A * T ** n * math.exp(- (E * cal) / (R * T))


def score(rxn, y_true, y_pred, save_file=r'validation.txt', print_info=True):
    """
    Final Evaluation Criteria
    :param rxn: reactions
    :param y_true: [[lnA, n, E], ] of truth
    :param y_pred: [[lnA, n, E], ] of prediction
    :param T_eg: given temperature for plotting
    :param save_file:
    :param print_info:
    :return:
    """
    lnA_true, n_true, E_true = [x[0] for x in y_true], [x[1] for x in y_true], [x[2] for x in y_true]
    lnA_pred, n_pred, E_pred = [x[0] for x in y_pred], [x[1] for x in y_pred], [x[2] for x in y_pred]

    lnk_true, lnk_pred, rmse, nrmse = {}, {}, {}, {}
    lnk_all_true, lnk_all_pred = [], []
    for T_r in range(5, 21):  # temperature range of 500-2000 K
        T = 10000 / T_r
        T_ = str(int(T))
        lnk_true[T_], lnk_pred[T_] = [], []
        for i in range(len(rxn)):
            lnk_true[T_].append(cal_lnk(lnA_true[i], n_true[i], E_true[i], T))
            lnk_pred[T_].append(cal_lnk(lnA_pred[i], n_pred[i], E_pred[i], T))
        rmse[T_] = RMSE(lnk_true[T_], lnk_pred[T_])
        nrmse[T_] = NRMSE(lnk_true[T_], lnk_pred[T_])
        lnk_all_true.extend(lnk_true[T_])
        lnk_all_pred.extend(lnk_pred[T_])
    T_eg = [500, 1000, 2000]

    with open(save_file, 'w') as f:
        info = 'Validation results on test dataset:\n'
        info += 'R2 of lnk within 500-2000K is {:.4f}\n'.format(r2_score(lnk_all_true, lnk_all_pred))
        info += 'Average RMSE of lnk within 500-2000K is {:.4f}\n'.format(sum(rmse.values()) / len(rmse))
        info += 'Average NRMSE of lnk within 500-2000K is {:.2%}\n\n'.format(sum(nrmse.values()) / len(nrmse))

        info += '\tR2 score\tRMSE\tMPE\tNRMSE\n'
        info += 'lnA:\t{:.4f}\t{:.4f}\t{:.2%}\t{:.2%}\n'.format(r2_score(lnA_true, lnA_pred),
                                                                RMSE(lnA_true, lnA_pred),
                                                                MPE(lnA_true, lnA_pred),
                                                                NRMSE(lnA_true, lnA_pred))
        info += 'n:\t{:.4f}\t{:.4f}\t{:.2%}\t{:.2%}\n'.format(r2_score(n_true, n_pred),
                                                              RMSE(n_true, n_pred),
                                                              MPE(n_true, n_pred),
                                                              NRMSE(n_true, n_pred))
        info += 'E:\t{:.4f}\t{:.4f}\t{:.2%}\t{:.2%}\n'.format(r2_score(E_true, E_pred),
                                                              RMSE(E_true, E_pred),
                                                              MPE(E_true, E_pred),
                                                              NRMSE(E_true, E_pred))
        for temp in T_eg:
            info += 'lnk({}K):\t{:.4f}\t{:.4f}\t{:.2%}\t{:.2%}\n'.format(
                temp,
                r2_score(lnk_true[str(temp)], lnk_pred[str(temp)]),
                RMSE(lnk_true[str(temp)], lnk_pred[str(temp)]),
                MPE(lnk_true[str(temp)], lnk_pred[str(temp)]),
                NRMSE(lnk_true[str(temp)], lnk_pred[str(temp)]))
        info += '\nDetails:\n'
        info += 'Temp.:'
        for i in rmse:
            info += '\t{}'.format(int(i))
        info += '\nRMSE:'
        for i in rmse:
            info += '\t{:.4f}'.format(rmse[i])
        info += '\nNRMSE:'
        for i in nrmse:
            info += '\t{:.2%}'.format(nrmse[i])

        info += '\n\tTruth\t\t\t\t\t\tPrediction\n'
        info += 'Rxn.\tlnA\tn\tE\tlnk({}K)\tlnk({}K)\tlnk({}K)\tlnA\tn\tE\tlnk({}K)\tlnk({}K)\tlnk({}K)\n'.format(
            T_eg[0], T_eg[1], T_eg[2], T_eg[0], T_eg[1], T_eg[2])
        for i in range(len(rxn)):
            info += '{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t'.format(
                rxn[i], lnA_true[i], n_true[i], E_true[i],
                lnk_true[str(T_eg[0])][i], lnk_true[str(T_eg[1])][i], lnk_true[str(T_eg[2])][i])
            info += '{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
                lnA_pred[i], n_pred[i], E_pred[i],
                lnk_pred[str(T_eg[0])][i], lnk_pred[str(T_eg[1])][i], lnk_pred[str(T_eg[2])][i])
        f.writelines(info)
        if print_info:
            print(info)
    return rmse


if __name__ == '__main__':
    pass


