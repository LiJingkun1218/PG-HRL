import random
import numpy as np
import torch

'''
This module contains job sequencing rules used in the experiment.
Sequencing agents may follow one of these rules or use trained parameters for decision-making.
'''

def _safe_tensor_to_numpy(data):
    """Safely convert tensor to numpy array."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data

def _get_argmin_with_random_tie(values):
    """Get index of min value, randomly choose if tie."""
    if isinstance(values, torch.Tensor):
        values_np = values.detach().cpu().numpy()
    else:
        values_np = values
    
    min_value = np.min(values_np)
    min_indices = np.where(values_np == min_value)[0]
    job_position = random.choice(min_indices)
    return job_position

def _get_argmax_with_random_tie(values):
    """Get index of max value, randomly choose if tie."""
    if isinstance(values, torch.Tensor):
        values_np = values.detach().cpu().numpy()
    else:
        values_np = values
    
    max_value = np.max(values_np)
    max_indices = np.where(values_np == max_value)[0]
    job_position = random.choice(max_indices)
    return job_position

# Benchmark, as the worst possible case

def CR(data): # Critical Ratio rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    time_till_due = data[5]
    CR = time_till_due / (data[0] + data[1])
    return _get_argmin_with_random_tie(CR) , 0

def MDD(data):  # Modified Due Date rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    due = data[2]
    finish = data[1] + data[3]    
    MDD = np.max([due, finish], axis=0)
    return _get_argmin_with_random_tie(MDD), 0

def MS(data): # Minimum Slack rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    return _get_argmin_with_random_tie(data[6]), 0

def ATC(data):  # Apparent Tardiness Cost rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    average_pt = data[0].mean()
    cost = (data[2] - data[3] - data[0]).clip(0, None)
    priority = np.exp(-cost / (0.05*average_pt)) / data[0]
    job_position = priority.argmax()
    return job_position, 0

def SPT(data): # Shortest Processing Time rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    return _get_argmin_with_random_tie(data[0]), 0

def LSPO(data): # Least Slack per Operation rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    sum_val = data[7]/(data[11]+0.001)
    return _get_argmin_with_random_tie(sum_val), 0

def EDD(data): # Earliest Due Date rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    return _get_argmin_with_random_tie(data[2]), 0

def random_sequencing(data): # Random sequencing
    data = [_safe_tensor_to_numpy(d) for d in data]
    job_position = np.random.randint(len(data[0]))
    return job_position, 0

def MOD(data): # Modified Operation Due Date rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    due = data[2]
    operational_finish = data[0] + data[3]
    MOD = np.max([due, operational_finish], axis=0)
    job_position = MOD.argmin()
    return job_position, 0

def CRSPT(data): # CR + SPT rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    CRSPT = data[5] / (data[0] + data[1]) + data[0]
    job_position = CRSPT.argmin()
    return job_position, 0

def LPT(data): # Longest Processing Time rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    return _get_argmax_with_random_tie(data[0]), 0

def LRO(data): # Least Remaining Operations / Highest Completion Rate rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    return _get_argmax_with_random_tie(data[10]), 0

def SRO(data): # Shortest Remaining Operations rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    job_position = data[10].argmin()
    return job_position, 0

def LWKR(data): # Least Work Remaining rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    job_position = (data[0] + data[1]).argmin()
    return job_position, 0

def LWKRSPT(data): # LWRK + SPT rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    job_position = (data[0]*2 + data[1]).argmin()
    return job_position, 0

def LWKRMOD(data): # LWRK + MOD rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    due = data[2]
    operational_finish = data[0] + data[3]
    MOD = np.max([due, operational_finish], axis=0)
    job_position = (data[0] + data[1] + MOD).argmin()
    return job_position, 0

def LDD(data): # Latest Due Date rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    return _get_argmax_with_random_tie(data[2]), 0

def COVERT(data): # Cost Over Time rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    average_pt = data[0].mean()
    cost = (data[2] - data[3] - data[0]).clip(0, None)
    priority = (1 - cost / (0.05*average_pt)).clip(0, None) / data[0]
    job_position = priority.argmax()
    return job_position, 0

def MON(data): # Montagne heuristic
    data = [_safe_tensor_to_numpy(d) for d in data]
    due_over_pt = np.array(data[2])/np.sum(data[0])
    priority = due_over_pt/np.array(data[0])
    job_position = priority.argmax()
    return job_position, 0

def NPT(data): # Next Processing Time rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    job_position = data[9].argmin()
    return job_position, 0

def AVPRO(data): # Average Processing Time per Operation rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    AVPRO = (data[0] + data[1]) / (data[10] + 1)
    job_position = AVPRO.argmin()
    return job_position, 0

def SRMWK(data): # Slack per Remaining Work rule (identical to CR)
    data = [_safe_tensor_to_numpy(d) for d in data]
    SRMWK = data[6] / (data[0] + data[1])
    job_position = SRMWK.argmin()
    return job_position, 0

def SRMWKSPT(data): # Slack per Remaining Work + SPT rule (identical to CR+SPT)
    data = [_safe_tensor_to_numpy(d) for d in data]
    SRMWKSPT = data[6] / (data[0] + data[1]) + data[0]
    job_position = SRMWKSPT.argmin()
    return job_position, 0

def WINQ(data): # WINQ (Work In Next Queue) rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    return _get_argmin_with_random_tie(data[7]), 0

def PTWINQ(data): # PT + WINQ rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    sum_val = data[0] + data[7]
    job_position = sum_val.argmin()
    return job_position, 0

def PTWINQS(data): # PT + WINQ + Slack rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    sum_val = data[0] + data[6] + data[7]
    job_position = sum_val.argmin()
    return job_position, 0

def DPTWINQNPT(data): # 2PT + WINQ + NPT rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    sum_val = data[0]*2 + data[7] + data[9]
    job_position = sum_val.argmin()
    return job_position, 0

def DPTLWKR(data): # 2PT + LWKR rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    sum_val = data[0]*2 + data[1]
    job_position = sum_val.argmin()
    return job_position, 0

def DPTLWKRS(data): # 2PT + LWKR + Slack rule
    data = [_safe_tensor_to_numpy(d) for d in data]
    sum_val = data[0]*2 + data[1] + data[6]
    job_position = sum_val.argmin()
    return job_position, 0

def FIFO(dummy): # First In, First Out rule (data not needed)
    job_position = 0
    return job_position, 0