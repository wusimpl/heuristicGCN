import numpy as np, sys, os, random, pdb, json, uuid, time, argparse
from pprint import pprint
import logging, logging.config
from collections import defaultdict as ddict
from ordered_set import OrderedSet
from random import sample
# PyTorch related imports
import torch
from torch.nn import functional as F
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader
from torch.nn import Parameter
from torch_scatter import scatter_add

np.set_printoptions(precision=4)

def set_gpu(gpus):
	"""
	Sets the GPU to be used for the run

	Parameters
	----------
	gpus:           List of GPUs to be used for the run
	
	Returns
	-------
		
	"""
	os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = gpus

def get_logger(name, log_dir, config_dir):
	"""
	Creates a logger object

	Parameters
	----------
	name:           Name of the logger file
	log_dir:        Directory where logger file needs to be stored
	config_dir:     Directory from where log_config.json needs to be read
	
	Returns
	-------
	A logger object which writes to both file and stdout
		
	"""
	config_dict = json.load(open( config_dir + 'log_config.json'))
	config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def get_combined_results(left_results, right_results):
	results = {}
	count   = float(left_results['count'])

	results['left_mr']	= round(left_results ['mr'] /count, 5)
	results['left_mrr']	= round(left_results ['mrr']/count, 5)
	results['right_mr']	= round(right_results['mr'] /count, 5)
	results['right_mrr']	= round(right_results['mrr']/count, 5)
	results['mr']		= round((left_results['mr']  + right_results['mr']) /(2*count), 5)
	results['mrr']		= round((left_results['mrr'] + right_results['mrr'])/(2*count), 5)

	for k in range(10):
		results['left_hits@{}'.format(k+1)]	= round(left_results ['hits@{}'.format(k+1)]/count, 5)
		results['right_hits@{}'.format(k+1)]	= round(right_results['hits@{}'.format(k+1)]/count, 5)
		results['hits@{}'.format(k+1)]		= round((left_results['hits@{}'.format(k+1)] + right_results['hits@{}'.format(k+1)])/(2*count), 5)
	return results

def get_param(shape):
	param = Parameter(torch.Tensor(*shape))
	xavier_normal_(param.data)
	return param

def com_mult(a, b):
	r1, i1 = a[..., 0], a[..., 1]
	r2, i2 = b[..., 0], b[..., 1]
	return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim = -1)

def conj(a):
	a[..., 1] = -a[..., 1]
	return a

def cconv(a, b):
	return torch.irfft(com_mult(torch.rfft(a, 1), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

def ccorr(a, b):
	return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

# 实现一个函数,输入一个长度为N的long tensor x,x:
# 如果元素出现次数 <=num,返回它在 x 中的所有索引
# 如果元素出现次数 >num,随机选取两个索引返回
# 对每个元素重复步骤1和2
# 例如,如果x中元素5出现3次,该函数会随机选择5的两个索引返回,其余<=2次的元素全部返回。
def sample_indices(x, num):
	# 对于x没有邻居的情况
	if x.shape[-1]==0:
		return None

	device = x.device
	values, counts = torch.unique(x, return_counts=True)
	indices = []
	for value, count in zip(values, counts):
		value_indices = torch.nonzero(x == value).squeeze()
		if count <= num:
			indices.append(value_indices)
		else:
			sampled_indices = sample(list(value_indices), num)
			indices.append(torch.tensor(sampled_indices,device=device))
	indices = [item.reshape(-1) for item in indices]

	return torch.cat(indices)