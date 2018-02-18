
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import pypower.api as P
#import os


PD = 2

#Example of writing path

#filename = os.path.join(os.path.abspath(os.curdir), 'matpower6.0/case2869pegase.m')



class first_simulation():
    def __init__(self, J = 360, tau = 6):
        self.J = J
        self.tau = tau
        self.right_answer = np.exp( - self.tau**2 / 2)
    def __call__(self):
        coef = np.arange(1, self.J + 1) / self.J
        w_1 = np.sin(2* np.pi * coef)
        w_2 = np.cos(2* np.pi * coef)
        self.W = np.hstack((w_1.reshape((-1,1)), w_2.reshape((-1,1))))
        return self.W, self.tau * np.ones((self.J))
    
class second_simulation():
    def __init__(self, J = 360, tau = 6):
        self.J = J
        self.tau = tau
    def __call__(self):
        def prime_numbers(num):
            coef = np.arange(1, num + 1)
            for elem in coef[1:]:
                filt = (coef % elem != 0) | (coef == elem)
                coef = coef[filt]
            return coef
        coef = prime_numbers(self.J) / self.J
        w_1 = np.sin(2* np.pi * coef)
        w_2 = np.cos(2* np.pi * coef)
        self.W = np.hstack((w_1.reshape((-1,1)), w_2.reshape((-1,1))))
        return self.W, self.tau * np.ones((self.W.shape[0]))
        


def preprocess(filename):
    
    def initial_preprocess(filename):
        data = []
        with open(filename, 'r') as f:
            for line in f:
                data.append(line.split('\t'))
        length = list(map(len, data))
        length_arr = np.array(length)
        indices = np.argwhere(length_arr == 1)
        for i in range(len(indices)):
            if 'baseMVA' in data[i][0]:
                s = data[i][0]
                s = s.strip(';\n')
                s = s[s.find('=') + 1:]
                baseMVA = int(s)
                break
        new_data = []
        for i in range(len(indices)):
            if i != len(indices) - 1:
                new_data.append(data[indices[i,0] + 1:indices[i+1,0]])
            else:
                new_data.append(data[indices[i,0] + 1:])
        new_indices = np.argwhere(np.array(list(map(len, new_data))) > 2)[:,0]
        bus_data = new_data[new_indices[0]]
        generator_data = new_data[new_indices[1]]
        branch_data = new_data[new_indices[2]]
        cost_data = new_data[new_indices[3]]
        return baseMVA, bus_data, generator_data, branch_data, cost_data

    def final_preprocess(data):
        data = np.array(data)
        data = data[:, 1:]
        data[:,-1] = np.array(list(map(lambda x: x.rstrip(';\n'), data[:, -1])))
        data = data.astype(float)
        return data
    
    baseMVA, bus_data, generator_data, branch_data, cost_data = initial_preprocess(filename)
    bus_data = final_preprocess(bus_data)
    generator_data = final_preprocess(generator_data)
    branch_data = final_preprocess(branch_data)
    cost_data = final_preprocess(cost_data)
    return baseMVA, bus_data, generator_data, branch_data, cost_data

def DF_converter(bus_data, generator_data, branch_data):
    col_bus = ['bus number', 'bus type', 'Pd', 'Qd', 'Gs', 'Bs', 'area number', 'Vm', 'Va',                'baseKV', 'zone, loss zone', 'maxVm', 'minVm']
    col_gen = ['bus number', 'Pg', 'Qg', 'Qmax', 'Qmin', 'Vg', 'mBase', 'status', 'Pmax', 'Pmin', 'Pc1',                'Pc2', 'Qc1min', 'Qc1max', 'Qc2min', 'Qc2max', 'ramp rate for load following/AGC (MW/min)',                'ramp rate for 10 minute reserves (MW)', 'ramp rate for 30 minute reserves (MW)',                'ramp rate for reactive power (2 sec timescale) (MVAr/min)', 'APF']
    branch_col = ['f', 't', 'r', 'x', 'b', 'rateA', 'rateB', 'rateC', 'ratio', 'angle', 'initial branch status',                   'minimum angle difference', 'maximum angle difference']
    bus_df = pd.DataFrame(bus_data, columns= col_bus)
    bus_df[['bus number', 'bus type', 'area number', 'zone, loss zone']] = bus_df[['bus number',                                                                                    'bus type',                                                                                    'area number',                                                                                    'zone, loss zone']].astype(int) 
    gen_df = pd.DataFrame(generator_data, columns= col_gen)
    gen_df[['bus number', 'status']] = gen_df[['bus number', 'status']].astype(int) 
    branch_df = pd.DataFrame(branch_data, columns= branch_col)
    branch_df[['f', 't', 'initial branch status']] = branch_df[['f', 't', 'initial branch status']].astype(int)
    return bus_df, gen_df, branch_df


def reorder_busses(bus_dict, bus_df, branch_df, gen_df):
    bus_df1 = bus_df.copy()
    branch_df1 = branch_df.copy()
    gen_df1 = gen_df.copy()
    bus_df1['bus number alt'] = bus_dict.loc[bus_df1['bus number'].values].values[:,0]
    bus_df1 = bus_df1.sort_values(['bus number alt'])
    bus_df1['bus number'] = bus_df1['bus number alt']
    bus_df1 = bus_df1.drop(labels='bus number alt', axis = 1)
    branch_df1.f = bus_dict.loc[branch_df1.f.values].values
    branch_df1.t = bus_dict.loc[branch_df1.t.values].values
    gen_df1['bus number'] = bus_dict.loc[gen_df1['bus number'].values].values
    return bus_df1, branch_df1, gen_df1.sort_values(['bus number'])

def transform_bus_gen_data(bus_df, gen_df):
    ## it is assumed that the slack bus can't turn out to be in gen_df1
    assert len(gen_df[gen_df.status == 0]) == 0
    bus_df1 = bus_df.copy()
    gen_df1 = gen_df.copy()
    bus_df1 = bus_df1.sort_values(['bus number'])
    gen_df1 = gen_df1.sort_values(['bus number'])
    candidates = gen_df1[gen_df1.Pmax == gen_df1.Pmin]['bus number'].values
    idx = gen_df1[gen_df1.status == 0].index
    gen_df1 = gen_df1.drop(idx)
    C_g = np.zeros((bus_df1.shape[0], gen_df1.shape[0]))
    vec = bus_df1['bus number'].isin(gen_df1['bus number'].values).values
    C_g[np.argwhere(vec)[:, 0], np.arange(gen_df1.shape[0])] = 1
    diff = bus_df1.Pd.values + bus_df1.Gs.values
    S_r =  C_g.dot(gen_df1.Pg.values) - diff
    bus_df1.loc[(bus_df1['bus number'].isin(candidates)), 'bus type'] = 1
    idx = gen_df1[gen_df1.Pmax == gen_df1.Pmin].index
    gen_df1 = gen_df1.drop(idx)
    bus_df1.loc[:, 'Gs'] = 0
    idx = np.argwhere(bus_df1['bus type'] == 1)[:, 0]
    not_idx = np.argwhere(bus_df1['bus type'] != 1)[:, 0]
    bus_df1.iloc[idx, PD] = -S_r[idx]
    bus_df1.iloc[not_idx, PD] = 0
    gen_df1.loc[:, 'Pg'] = S_r[not_idx]
    gen_df1.loc[:, 'Pmax'] -= diff[not_idx]
    gen_df1.loc[:, 'Pmin'] -= diff[not_idx]
    return bus_df1, gen_df1
    
    
    
def D_constructor(bus_dict, branch_df):
    N = len(bus_dict)
    M = branch_df.shape[0]
    d = np.zeros((M, N))
    f = bus_dict.loc[branch_df.f.values].values.ravel()
    t = bus_dict.loc[branch_df.t.values].values.ravel()
    d[np.arange(M), f] = 1
    d[np.arange(M), t] = -1
    return d


def B_constructor_with_shift_correction(baseMVA, bus_dict, bus_df, gen_df, branch_df):
    bus_df1, branch_df1, gen_df1 = reorder_busses(bus_dict, bus_df, branch_df, gen_df)
    B, _, sh, _ = P.makeBdc(baseMVA , bus_df1.values, branch_df1.values)
    B = B.toarray()
    idx = np.argwhere(bus_df1['bus type'] == 1)[:, 0]
    not_idx = np.argwhere(bus_df1['bus type'] != 1)[:, 0]
    bus_df1.iloc[idx, PD] += sh[idx]*baseMVA
    gen_df1.loc[:, 'Pg'] -= sh[not_idx]*baseMVA
    gen_df1.loc[:, 'Pmax'] -= sh[not_idx]*baseMVA
    gen_df1.loc[:, 'Pmin'] -= sh[not_idx]*baseMVA
    return B, bus_df1, gen_df1, branch_df1

def p_bounds(baseMVA, bus_dict ,gen_df):
    N = len(bus_dict)
    #N_R = gen_df.shape[0] - 1
    slack_num = bus_dict.iloc[-1, 0]
    #bus_dict = dict(zip(bus_df['bus number'].values, range(bus_df.shape[0])))
    p_s = gen_df[gen_df['bus number'] == slack_num].Pmin.values[0]
    P_s = gen_df[gen_df['bus number'] == slack_num].Pmax.values[0]
    p_r = gen_df[gen_df['bus number'] != slack_num].Pmin.values
    P_r = gen_df[gen_df['bus number'] != slack_num].Pmax.values
    p_g = gen_df[gen_df['bus number'] != slack_num].Pg.values
    return p_s / baseMVA, P_s / baseMVA, p_r / baseMVA, P_r / baseMVA, p_g / baseMVA

def bus_dict_constructor(bus_df):
    assert len(bus_df[bus_df['bus type'] == 4]) == 0
    bus_dict1 = bus_df[bus_df['bus type'] == 2]
    N_R = len(bus_dict1)
    bus_dict1 = pd.DataFrame(data= np.arange(N_R), index = bus_dict1['bus number'].values)
    bus_dict2 = bus_df[bus_df['bus type'] == 1] 
    N_F = len(bus_dict2)
    bus_dict2 = pd.DataFrame(data= np.arange(N_R, N_R + N_F), index= bus_dict2['bus number'].values)
    bus_dict3 = bus_df[bus_df['bus type'] == 3]
    N_S = len(bus_dict3)
    assert N_S == 1
    bus_dict3 = pd.DataFrame(data= np.arange(N_R + N_F, N_R + N_F + N_S), index= bus_dict3['bus number'].values)
    bus_dict = pd.concat([bus_dict1, bus_dict2, bus_dict3])
    return bus_dict

def p_fixed(baseMVA, bus_df):
    bus_df = bus_df[bus_df['bus type'] == 1]
    p_f = bus_df.Pd.values
    return - p_f / baseMVA

def p_random(P_r, p_r, p_g, way = 'fourth', k = 20, rareness = 2e6):
    mu = (P_r + p_r + k*p_g) / (k + 2.0)
    dist = np.min(np.vstack((mu-p_r, P_r - mu)).T, axis= 1)
    if way == 'first':
        sigma = dist / 2.0
    elif way == 'second':
        sigma = dist/ (4.0 * scipy.stats.norm.ppf(0.75))
    elif way == 'third':
        sigma = dist
    elif way == 'fourth':
        sigma = dist / rareness
    else:
        raise NotImplementedError
    return mu, np.diag(sigma)
############################################
###TO CHECK!!!!!!!
############################################

def constraints_constructor(baseMVA, bus_df, gen_df, branch_df, theta_bound, random_params = None):
    bus_df1, gen_df1 = transform_bus_gen_data(bus_df, gen_df)
    bus_dict = bus_dict_constructor(bus_df1)
    D = D_constructor(bus_dict, branch_df)
    B, bus_df1, gen_df1, branch_df1 = B_constructor_with_shift_correction(baseMVA, bus_dict, bus_df1, gen_df1, branch_df)
    #print(B)
    B_inv = np.linalg.pinv(B)
    p_s, P_s, p_r, P_r, p_g = p_bounds(baseMVA, bus_dict, gen_df1)
    p_f = p_fixed(baseMVA, bus_df1)
    N_R = len(p_r)
    N_F = len(p_f)
    M = D.shape[0]
    B_R = B_inv[:, :N_R]
    B_F = B_inv[:, N_R:-1]
    B_S = B_inv[:, -1].reshape((-1, 1))
    Gamma = np.zeros((2*N_R + 2 + 2 * M , N_R))
    Gamma[:N_R, :] = np.identity(N_R)
    Gamma[N_R: 2*N_R, :] = - np.identity(N_R)
    Gamma[2*N_R, :] = np.ones((N_R))
    Gamma[2*N_R + 1, :] = - np.ones((N_R))
    Gamma[2*N_R + 2: 2*N_R + 2 + M, :] = D.dot(B_R) - D.dot(B_S.dot(np.ones((1,N_R))))
    Gamma[2*N_R + 2 + M:,:] = - D.dot(B_R) + D.dot(B_S.dot(np.ones((1,N_R))))
    Key = np.zeros((2*N_R + 2 + 2 * M))
    Key[:N_R] = P_r
    Key[N_R: 2*N_R] = -p_r
    Key[2*N_R] = - p_s - p_f.sum()
    Key[2*N_R + 1] = P_s + p_f.sum()
    Key[2*N_R + 2: 2*N_R + 2 + M] = theta_bound - (D.dot(B_F) - D.dot(B_S.dot(np.ones((1, N_F))))).dot(p_f)
    Key[2*N_R + 2 + M:] = theta_bound + (D.dot(B_F) - D.dot(B_S.dot(np.ones((1, N_F))))).dot(p_f)
    if random_params == None:
        mu, Sigma = p_random(P_r, p_r, p_g)
    else:
        mu, Sigma = p_random(P_r, p_r, p_g, *random_params)
    return Gamma, Key, mu, Sigma

def adjust_constraints(constraints, dist_params):
    Gamma, Key = constraints
    mu, Sigma = dist_params
    sig_sqrt = np.sqrt(Sigma)
    normalizer = np.diag(Gamma.dot(Sigma).dot(Gamma.T))
    normalizer = np.sqrt(normalizer)
    W = Gamma.dot(sig_sqrt) / normalizer.reshape((-1,1))
    tau = Key - Gamma.dot(mu)
    tau = tau / normalizer
    return W, tau

