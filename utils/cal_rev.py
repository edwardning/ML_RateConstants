# Author: Ning Li, contact: edwardln@stu.xjtu.edu.cn or l.ning@itv.rwth-aachen.de
# Mathematical principles referenced from CHEMRev (https://doi.org/10.1002/kin.20049)
import os
import time
import math
import numpy as np

R = 8.31446261815324  # J/(mol*K) from the 26th General Conference on Weights and Measures (CGPM)
R_ = 82.057338  # atm*cm3/(mol*K). CHEMKIN uses cgs unit.
cal = 4.184  # CHEMKIN uses the thermal calorie, 1 cal = 4.184 Joules


class Therm:
    def __init__(self):
        self.data = {}
        self.spec = ''
        self.formula = {}
        self.phase = 'G'
        self.phase_ = {'G': 'gas', 'L': 'liquid', 'S': 'solid'}
        self.T_low = 300.  # K
        self.T_common = 1000  # K
        self.T_high = 5000  # K
        self.a_lowT = [0.]*7
        self.a_highT = [0.]*7
        self.source = ''

    def addLine(self, line, indicator):
        self.source += line+'\n'
        if indicator == '1':
            self.add_l1(line)
        elif indicator == '2':
            line = [line[i:i + 15] for i in range(0, 79, 15)]
            self.a_highT[:5] = [float(x) for x in line]
        elif indicator == '3':
            line = [line[i:i + 15] for i in range(0, 79, 15)]
            self.a_highT[5:] = [float(x) for x in line[:2]]
            self.a_lowT[:3] = [float(x) for x in line[2:]]
        elif indicator == '4':
            line = [line[i:i + 15] for i in range(0, 60, 15)]
            self.a_lowT[3:] = [float(x) for x in line]
        return

    def add_l1(self, l1):
        self.spec = l1[:16].strip()
        # self.spec = l1.split()[0]
        self.get_formula(l1[24:44])
        self.phase = self.phase_[self.phase] if l1[44] == ' ' else self.phase_[l1[44].upper()]
        self.T_low = float(l1[45:55])
        self.T_high = float(l1[55:65])
        self.T_common = float(l1[65:73])
        self.get_formula(l1[73:78])
        return

    def get_formula(self, ss):
        ss = [ss[i:i+5] for i in range(0, 20, 5)]
        for s in ss:
            elem = s[:2].strip()
            if len(elem) != 0 and (not elem.isdigit()):
                self.formula[elem] = int(s[2:5])
        return

    def print_data(self):
        print(self.source)
        return

    def info(self):
        print('Therm data for {}:'.format(self.spec))
        print('Formula: {}'.format(self.formula))
        print('Phase: {}'.format(self.phase))
        print('Temperature segmentation: {}K {}K {}K'.format(self.T_low, self.T_high, self.T_common))
        print('Coefficients for high-T: {}'.format(self.a_highT))
        print('Coefficients for low-T : {}'.format(self.a_lowT))
        return

    def Cp(self, T):  # Unit: J
        return R * self.Cp_R(T)

    def H(self, T):  # Unit: J
        return R * T * self.H_RT(T)

    def S(self, T):  # Unit: J
        return R * self.S_R(T)

    def Cp_R(self, T):  # Unit: J
        if T < self.T_common:
            [a1, a2, a3, a4, a5, a6, a7] = self.a_lowT
        else:
            [a1, a2, a3, a4, a5, a6, a7] = self.a_highT
        return a1 + a2*T + a3*T**2 + a4*T**3 + a5*T**4

    def H_RT(self, T):  # Unit: J
        if T < self.T_common:
            [a1, a2, a3, a4, a5, a6, a7] = self.a_lowT
        else:
            [a1, a2, a3, a4, a5, a6, a7] = self.a_highT
        return a1 + a2/2*T + a3/3*T**2 + a4/4*T**3 + a5/5*T**4 + a6/T

    def S_R(self, T):  # Unit: J
        if T < self.T_common:
            [a1, a2, a3, a4, a5, a6, a7] = self.a_lowT
        else:
            [a1, a2, a3, a4, a5, a6, a7] = self.a_highT
        return a1*math.log(T) + a2*T + a3/2*T**2 + a4/3*T**3 + a5/4*T**4 + a7

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return print('Therm data for {}. More information using .info()'.format(self.spec))


def load_therm_file(file_path):
    therm = {}
    spec = ''
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.lstrip().startswith('!') or len(line.rstrip()) < 80:
                continue
            int_ = line[79]
            if int_ == '1':
                spec = line[:16].strip()
                # spec = line.split()[0]
                if spec in therm:
                    print('duplicate species: ', spec)
                therm[spec] = Therm()
                therm[spec].addLine(line, int_)
            elif int_ in ['2', '3', '4']:
                therm[spec].addLine(line, int_)
    return therm


def get_rev_rate(rxn, A, n, E, therm, T_low=500, T_high=2000, num=50):
    # rxn: split by '=' and '+', no extra info like (+M)
    # num: griding numbers
    # therm: Therm object
    # Fitting Theory: beta = (XT·X)^(-1)·XT·Y
    T_space = 1000 / np.linspace(1000/T_low, 1000/T_high, num)
    ln_k_rev = []
    reac, prod = rxn.split('=')
    reac = reac.split('+')
    prod = prod.split('+')

    for T in T_space:
        del_H_RT, del_S_R = 0., 0.
        for p in prod:
            del_H_RT += therm[p].H_RT(T)
            del_S_R += therm[p].S_R(T)
        for r in reac:
            del_H_RT -= therm[r].H_RT(T)
            del_S_R -= therm[r].S_R(T)
        Kp = math.exp(del_S_R-del_H_RT)
        Kc = Kp*(R_*T)**(len(reac)-len(prod))
        k_f = A * T ** n * math.exp(-(E * cal) / (R * T))
        k_r = k_f / Kc
        print(T, '\t', del_H_RT, '\t', del_S_R, '\t', Kp, '\t', Kc, '\t', k_f, '\t', k_r)
        ln_k_rev.append(math.log(k_r))
    X = np.array([[1, math.log(t), 1/t] for t in T_space])
    Y = np.array(ln_k_rev)
    beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    A_r, n_r, E_r = list(beta)
    A_r = math.exp(A_r)  # k
    E_r = -E_r*R/cal  # cal/mol

    # calculate fitting error
    error_max = 0
    for i, T in enumerate(T_space):
        k_fitting = A_r * T ** n_r * math.exp(-(E_r * cal) / (R * T))
        k_fitting = math.log(k_fitting)
        k_rev = ln_k_rev[i]
        error = abs((k_fitting - k_rev) / k_rev)
        error_max = error if (error > error_max) else error_max

    return A_r, n_r, E_r, error_max


if __name__ == '__main__':
    file = r'C:\Users\edwardning\Desktop\RWTH-DE\NUIGTools\chemrev\thermo.dat'
    the = load_therm_file(file)
    # A, n, E, err = get_rev_rate('C4H9OaOO=C4H9Oa+O2', 6.970E+25,  -3.360,   36067.8, the)
    A, n, E, err = get_rev_rate('C4H9ObOO=C4H9Ob+O2', 2.450E+18, -0.990, 38595.5, the)

    print('\nReverse rate constants in Arrhenius form:\n{:.4e}\t{:.4f}\t{:.2f}\t{:.2%}'.format(A, n, E, err))

