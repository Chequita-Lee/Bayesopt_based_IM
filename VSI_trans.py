'''
identification of optimal Sv and VSI
'''


import numpy as np
import os
from bayes_opt import BayesianOptimization
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import time

path = os.getcwd()
print(path)
events_name = ['Set_1a_processed', 'Set_1b_processed', 'Set_2_processed', 'Set_4_processed']
events_path = [path + '/' + k for k in events_name]
Info_curve = pd.read_excel(path + '/response_cable_trans/Xtrain_cable_trans.xlsx', sheet_name='Sheet3')
if_converge = Info_curve['if_converge'].values
index_converge = np.where(if_converge == 'yes')[0]
MC_Curve1155 = 0.0008
MC_Curve1143 = 0.0009
MC_Curve1144 = 0.0008
MC_Curve4221 = 0.0009
MC_Curve4121 = 0.0009
MC_Curve4114 = 0.0009

Curve1155 = Info_curve['Curve1155'].values.reshape(-1, 1)[index_converge,:] / MC_Curve1155
Curve1143 = Info_curve['Curve1143'].values.reshape(-1, 1)[index_converge,:] / MC_Curve1143
Curve1144 = Info_curve['Curve1144'].values.reshape(-1, 1)[index_converge,:] / MC_Curve1144
Curve4221 = Info_curve['Curve4221'].values.reshape(-1, 1)[index_converge,:] / MC_Curve4221
Curve4121 = Info_curve['Curve4121'].values.reshape(-1, 1)[index_converge,:] / MC_Curve4121
Curve4114 = Info_curve['Curve4114'].values.reshape(-1, 1)[index_converge,:] / MC_Curve4114


#################################################################################################
#  ---------------------------- identification of optimal Sv(Topt)-------------------------------

# --------------------------Calculation of Sv at each Ti-------------------------------
def Sv_cal(t_start):
    """
    read Sv at t_start of each ground motion
    """
    intensity = []
    for k in range(len(events_name)):
        path_FN = path + '/' + events_name[k] + '/Spectrum_FN'
        FN_list = os.listdir(path_FN)
        Sv_FN_list = [a for a in FN_list if a[:2] == "Sv"]
        for j in range(len(Sv_FN_list)):
            Sv_FN = np.loadtxt(path_FN + '/Sv_' + events_name[k][4:-10] + '_' + str(j + 1) + '.txt')
            index_T1 = np.where(Sv_FN[:, 0] > t_start)
            Sv_FN_M = Sv_FN[index_T1[0][0], 1]
            intensity.append(round(Sv_FN_M, 6))

    intensity = np.array(intensity).reshape(-1, 1)
    return intensity

# -------------------------------- define objective functions --------------------------------------
def objective_Curve_pylon(t_start):
    """
    define the objective function of Bayesopt with SV as IM
    """
    intensity = Sv_cal(t_start)
    intensity = intensity[index_converge, :]
    X_IM = np.log(intensity).reshape(-1, 1)
    Y_Curve1155 = np.log(Curve1155)
    y_predict_Curve1155 = LinearRegression().fit(np.hstack((X_IM, np.power(X_IM, 2))), Y_Curve1155.reshape(-1, 1)).predict(np.hstack((X_IM, np.power(X_IM, 2))))
    r2_Curve1155 = r2_score(Y_Curve1155, y_predict_Curve1155)
    # print(r2_Curve1155)

    Y_Curve1143 = np.log(Curve1143)
    y_predict_Curve1143 = LinearRegression().fit(np.hstack((X_IM, np.power(X_IM, 2))), Y_Curve1143.reshape(-1, 1)).predict(np.hstack((X_IM, np.power(X_IM, 2))))
    r2_Curve1143 = r2_score(Y_Curve1143, y_predict_Curve1143)

    Y_Curve1144 = np.log(Curve1144)
    y_predict_Curve1144 = LinearRegression().fit(np.hstack((X_IM, np.power(X_IM, 2))), Y_Curve1144.reshape(-1, 1)).predict(np.hstack((X_IM, np.power(X_IM, 2))))
    r2_Curve1144 = r2_score(Y_Curve1144, y_predict_Curve1144)

    Y_Curve4221 = np.log(Curve4221)
    y_predict_Curve4221 = LinearRegression().fit(np.hstack((X_IM, np.power(X_IM, 2))),Y_Curve4221.reshape(-1, 1)).predict(np.hstack((X_IM, np.power(X_IM, 2))))
    r2_Curve4221 = r2_score(Y_Curve4221, y_predict_Curve4221)

    Y_Curve4121 = np.log(Curve4121)
    y_predict_Curve4121 = LinearRegression().fit(np.hstack((X_IM, np.power(X_IM, 2))),Y_Curve4121.reshape(-1, 1)).predict(np.hstack((X_IM, np.power(X_IM, 2))))
    r2_Curve4121 = r2_score(Y_Curve4121, y_predict_Curve4121)

    Y_Curve4114 = np.log(Curve4114)
    y_predict_Curve4114 = LinearRegression().fit(np.hstack((X_IM, np.power(X_IM, 2))),Y_Curve4114.reshape(-1, 1)).predict(np.hstack((X_IM, np.power(X_IM, 2))))
    r2_Curve4114 = r2_score(Y_Curve4114, y_predict_Curve4114)

#     print(r2_Curve1155, r2_Curve1143, r2_Curve1144, r2_Curve4221, r2_Curve4121, r2_Curve4114)

    r2 = round(np.power(r2_Curve1155*np.power(r2_Curve1143, 5)*r2_Curve1144*r2_Curve4221*np.power(r2_Curve4121, 3), 1/11), 6)

    return r2

def para_bayes_opt_curve_pylon(init_points, n_iter):
    opt = BayesianOptimization(objective_Curve_pylon, para_grid_sample, random_state=1416)
    opt.maximize(init_points=init_points, n_iter=n_iter)
    params_best = opt.max['params']
    score_best = opt.max['target']

    print('\n', '\n', 'params_best:', params_best,
          '\n', '\n', 'score_best:', score_best,)
    return params_best, score_best

start_time = time.time()
para_grid_sample = {'t_start':(0.03, 14.83)}  # range of independent variable, t_start.
params_best, score_best = para_bayes_opt_curve_pylon(10, 25)    # initial observation :10, following observations: 35
end_time = time.time()
print(end_time-start_time)
Topt = params_best['t_start']
print("optimal T for Sv :", Topt)

#################################################################################################
#  ---------------------------- identification of optimal VSI-------------------------------

def MVSI_cal(t_start, t_end):
    """
    calculate VSI with the integral from t_start to t_end
    """
    intensity = []
    for k in range(len(events_name)):
        path_FN = path + '/' + events_name[k] + '/Spectrum_FN'
        FN_list = os.listdir(path_FN)
        Sv_FN_list = [a for a in FN_list if a[:2] == "Sv"]
        for j in range(len(Sv_FN_list)):
            Sv_FN = np.loadtxt(path_FN + '/Sv_' + events_name[k][4:-10] + '_' + str(j + 1) + '.txt')
            index_T1 = np.where(Sv_FN[:, 0] > t_start)
            index_T2 = np.where(Sv_FN[:, 0] > t_end)
            Sv_FN_M = Sv_FN[index_T1[0][0]:index_T2[0][0], :]
            dT = round(Sv_FN_M[1, 0] - Sv_FN_M[0, 0], 4)
            MVSI = np.sum(dT * Sv_FN_M[:, 1])
            intensity.append(round(MVSI, 6))

    intensity = np.array(intensity).reshape(-1, 1)
    return intensity

# ---------------------------------------definition of objective function --------------------------------------
def objective_Curve_pylon(t_start, t_end):
    """
    define the objective function
    """
    intensity = MVSI_cal(t_start, t_end)
    intensity = intensity[index_converge, :]
    X_IM = np.log(intensity).reshape(-1, 1)
    Y_Curve1155 = np.log(Curve1155)
    y_predict_Curve1155 = LinearRegression().fit(np.hstack((X_IM, np.power(X_IM, 2))), Y_Curve1155.reshape(-1, 1)).predict(np.hstack((X_IM, np.power(X_IM, 2))))
    r2_Curve1155 = r2_score(Y_Curve1155, y_predict_Curve1155)

    Y_Curve1143 = np.log(Curve1143)
    y_predict_Curve1143 = LinearRegression().fit(np.hstack((X_IM, np.power(X_IM, 2))), Y_Curve1143.reshape(-1, 1)).predict(np.hstack((X_IM, np.power(X_IM, 2))))
    r2_Curve1143 = r2_score(Y_Curve1143, y_predict_Curve1143)

    Y_Curve1144 = np.log(Curve1144)
    y_predict_Curve1144 = LinearRegression().fit(np.hstack((X_IM, np.power(X_IM, 2))), Y_Curve1144.reshape(-1, 1)).predict(np.hstack((X_IM, np.power(X_IM, 2))))
    r2_Curve1144 = r2_score(Y_Curve1144, y_predict_Curve1144)

    Y_Curve4221 = np.log(Curve4221)
    y_predict_Curve4221 = LinearRegression().fit(np.hstack((X_IM, np.power(X_IM, 2))), Y_Curve4221.reshape(-1, 1)).predict(np.hstack((X_IM, np.power(X_IM, 2))))
    r2_Curve4221 = r2_score(Y_Curve4221, y_predict_Curve4221)

    Y_Curve4121 = np.log(Curve4121)
    y_predict_Curve4121 = LinearRegression().fit(np.hstack((X_IM, np.power(X_IM, 2))), Y_Curve4121.reshape(-1, 1)).predict(np.hstack((X_IM, np.power(X_IM, 2))))
    r2_Curve4121 = r2_score(Y_Curve4121, y_predict_Curve4121)

    Y_Curve4114 = np.log(Curve4114)
    y_predict_Curve4114 = LinearRegression().fit(np.hstack((X_IM, np.power(X_IM, 2))),Y_Curve4114.reshape(-1, 1)).predict(np.hstack((X_IM, np.power(X_IM, 2))))
    r2_Curve4114 = r2_score(Y_Curve4114, y_predict_Curve4114)

    r2 = round(np.power(r2_Curve1155*np.power(r2_Curve1143, 5)*r2_Curve1144*r2_Curve4221*np.power(r2_Curve4121, 3), 1/11), 6)

    return r2

def para_bayes_opt_curve_pylon(init_points, n_iter):
    opt = BayesianOptimization(objective_Curve_pylon, para_grid_sample, random_state=1416)
    opt.maximize(init_points=init_points, n_iter=n_iter)
    params_best = opt.max['params']
    score_best = opt.max['target']

    print('\n', '\n', 'params_best:', params_best,
          '\n', '\n', 'score_best:', score_best,)
    return params_best, score_best

start_time = time.time()
para_grid_sample = {'t_start':(0.03, Topt-0.01), 't_end':(Topt+0.01, 14.5)}    # range of independent variable, t_start and t_end
params_best, score_best = para_bayes_opt_curve_pylon(10, 35)     # initial observation :15, following observations: 35
end_time = time.time()
print(end_time-start_time)
T1_opt = params_best['t_start']
T2_opt = params_best['t_end']
print("optimal T1 and T2 for VSI :", T1_opt, ' , ', T2_opt)