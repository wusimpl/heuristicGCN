import torch_geometric.nn
from torch_geometric.datasets import RelLinkPredDataset
from torch_geometric.nn import MessagePassing
from torch_geometric.transforms import RandomLinkSplit
from helper import *
from data_loader import *
from tqdm import tqdm
# sys.path.append('./')
from model.models import *

class Runner(object):
	def __init__(self, params):
		self.p = params
		self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		# self.logger.info(vars(self.p))
		# pprint(vars(self.p))

		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cpu')

		self.load_data()
		self.model = self.add_model(self.p.model, self.p.score_func)
		self.optimizer = self.add_optimizer(self.model.parameters())

	def collate_fn(self,data):
		data = data[0]
		masked_triples = torch.stack([data.edge_index[0], data.edge_type, data.edge_type.fill_(-1)], dim=0)
		labels = data.edge_index[1]
		return masked_triples, labels
		return data  # suppose to return triple array and their labels.

	def __get_data_loader(self, data, split, batch_size, shuffle=False):
		return  DataLoader(
				RLPDDataset(data, num_hops=2, split=split, device=self.device),
				batch_size      = 1, ## sorry, this can only be one, sad.
				shuffle         = shuffle,
				num_workers     = max(0, self.p.num_workers),
				collate_fn      = self.collate_fn
			)

	def load_data(self):
		self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

		# current_directory = os.getcwd()
		# data_path = osp.join(current_directory, 'data', 'RLPD')
		##@@
		data_path = osp.join(osp.dirname(osp.realpath(__file__)), '', 'data', 'RLPD')

		full_data = RelLinkPredDataset(data_path, 'FB15k-237')[0]
		self.p.num_ent		= full_data.num_nodes
		self.p.num_rel		= full_data.edge_type.max() + 1 # 增加一种自环关系

		self.edge_index, self.edge_type =full_data.edge_index, full_data.edge_type
		non_split_data = Data(edge_index=full_data.edge_index, edge_type=full_data.edge_type, num_nodes=full_data.num_nodes)

		cached_path = osp.join(osp.dirname(osp.realpath(__file__)), '', 'data', 'RLPD','FB15k-237','cached','train.pt')
		if not os.path.exists(cached_path):	
			transform = RandomLinkSplit(num_val=0.2, num_test=0.3,
										is_undirected=True, split_labels=True)
			train_data, val_data, test_data = transform(non_split_data)

			train_data.pos_edge_label_type = train_data.edge_type[:train_data.pos_edge_label_index.shape[1]]
			train_data.neg_edge_label_type = train_data.edge_type[train_data.pos_edge_label_index.shape[1]:]
			val_data.pos_edge_label_type = val_data.edge_type[:val_data.pos_edge_label_index.shape[1]]
			val_data.neg_edge_label_type = val_data.edge_type[val_data.pos_edge_label_index.shape[1]:]
			test_data.pos_edge_label_type = test_data.edge_type[:test_data.pos_edge_label_index.shape[1]]
			test_data.neg_edge_label_type = test_data.edge_type[test_data.pos_edge_label_index.shape[1]:]

			train_data = train_data.to(self.device)
			val_data = val_data.to(self.device)
			test_data = test_data.to(self.device)
		else:
			train_data = torch.load(cached_path)
			val_data = torch.load(cached_path)
			test_data = torch.load(cached_path)

		# 耗时操作
		self.data_iter = {
			'train':self.__get_data_loader(train_data, 'train', self.p.batch_size),
			# 'valid':self.__get_data_loader(val_data,   'valid', self.p.batch_size),
			# 'test': self.__get_data_loader(test_data,  'test',  self.p.batch_size)
		}

	def add_model(self, model, score_func):
		model_name = '{}_{}'.format(model, score_func)

		if   model_name.lower()	== 'compgcn_transe': 	model = CompGCN_TransE(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'compgcn_distmult': 	model = CompGCN_DistMult(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'compgcn_conve': 	model = CompGCN_ConvE(self.edge_index, self.edge_type, params=self.p)
		else: raise NotImplementedError

		model.to(self.device)

		return model

	def add_optimizer(self, parameters):
		return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)

	def read_batch(self, batch, split):
		"""
		Function to read a batch of data and move the tensors in batch to CPU/GPU

		Parameters
		----------
		batch: 		the batch to process
		split: (string) If split == 'train', 'valid' or 'test' split

		
		Returns
		-------
		Head, Relation, Tails, labels
		"""
		if split == 'train':
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label
		else:
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label

	def save_model(self, save_path):
		state = {
			'state_dict'	: self.model.state_dict(),
			'best_val'	: self.best_val,
			'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.p)
		}
		torch.save(state, save_path)

	def load_model(self, load_path):
		state			= torch.load(load_path)
		state_dict		= state['state_dict']
		self.best_val		= state['best_val']
		self.best_val_mrr	= self.best_val['mrr'] 

		self.model.load_state_dict(state_dict)
		self.optimizer.load_state_dict(state['optimizer'])

	def evaluate(self, split, epoch):
		"""
		Function to evaluate the model on validation or test set

		Parameters
		----------
		split: (string) If split == 'valid' then evaluate on the validation set, else the test set
		epoch: (int) Current epoch count
		
		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""
		left_results  = self.predict(split=split, mode='tail_batch')
		right_results = self.predict(split=split, mode='head_batch')
		results       = get_combined_results(left_results, right_results)
		# self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
		return results

	def predict(self, split='valid', mode='tail_batch'):
		"""
		Function to run model evaluation for a given mode

		Parameters
		----------
		split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
		mode: (string):		Can be 'head_batch' or 'tail_batch'
		
		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""
		self.model.eval()

		with torch.no_grad():
			results = {}
			train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

			for step, batch in enumerate(train_iter):
				sub, rel, obj, label	= self.read_batch(batch, split)
				pred			= self.model.forward(sub, rel)
				b_range			= torch.arange(pred.size()[0], device=self.device)
				target_pred		= pred[b_range, obj]
				pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
				pred[b_range, obj] 	= target_pred
				ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

				ranks 			= ranks.float()
				results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
				results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
				results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
				for k in range(10):
					results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

				# if step % 100 == 0:
				# 	self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

		return results

	def train(self, epoch):
		"""
		Function to run one epoch of training

		Parameters
		----------
		epoch: current epoch count
		
		Returns
		-------
		loss: The loss value after the completion of one epoch
		"""
		# self.model.train()
		losses = []
		# train_dataloader = iter(self.data_iter['train'])
		train_dataloader = self.data_iter['train']

		for step, batch in enumerate(train_dataloader):
			self.optimizer.zero_grad()
			sub, rel, obj, label = self.read_batch(batch, 'train')

			pred	= self.model.forward(sub, rel)
			loss	= self.model.loss(pred, label)

			loss.backward()
			self.optimizer.step()
			losses.append(loss.item())

		loss = np.mean(losses)
		# self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
		return loss

	def fit(self):
		self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
		save_path = os.path.join('./checkpoints', self.p.name)

		if self.p.restore:
			self.load_model(save_path)
			self.logger.info('Successfully Loaded previous model')

		kill_cnt = 0
		for epoch in range(self.p.max_epochs):
			train_loss  = self.train(epoch)
			val_results = self.evaluate('valid', epoch)
			if val_results['mrr'] > self.best_val_mrr:
				self.best_val	   = val_results
				self.best_val_mrr  = val_results['mrr']
				self.best_epoch	   = epoch
				self.save_model(save_path)
				kill_cnt = 0
			else:
				kill_cnt += 1
				if kill_cnt % 10 == 0 and self.p.gamma > 5:
					self.p.gamma -= 5 
					self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
				if kill_cnt > 25: 
					self.logger.info("Early Stopping!!")
					break

			self.logger.info('[Epoch {:2}]: Train loss: {:.4} Valid MRR: {:.3} Best MRR: {:.3} hits@1: {:.3} hits@3: {:.3} hits@10: {:.3}'
							 .format(epoch,train_loss,val_results['mrr'],self.best_val_mrr,val_results['hits@1'],val_results['hits@3'],val_results['hits@10']))

		self.logger.info('Loading best model, Evaluating on Test data')
		self.load_model(save_path)
		test_results = self.evaluate('test', epoch)

def load_args():
	parser = argparse.ArgumentParser(description='Parser For Arguments',
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('-name', default='testrun', help='Set run name for saving/restoring models')
	parser.add_argument('-data', dest='dataset', default='FB15k-237', help='Dataset to use, default: FB15k-237')
	parser.add_argument('-model', dest='model', default='compgcn', help='Model Name')
	parser.add_argument('-score_func', dest='score_func', default='conve', help='Score Function for Link prediction')
	parser.add_argument('-opn', dest='opn', default='corr', help='Composition Operation to be used in CompGCN')

	parser.add_argument('-batch', dest='batch_size', default=128, type=int, help='Batch size')
	parser.add_argument('-gamma', type=float, default=40.0, help='Margin')
	parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
	parser.add_argument('-epoch', dest='max_epochs', type=int, default=500, help='Number of epochs')
	parser.add_argument('-l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
	parser.add_argument('-lr', type=float, default=0.001, help='Starting Learning Rate')
	parser.add_argument('-lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
	parser.add_argument('-num_workers', type=int, default=10, help='Number of processes to construct batches')
	parser.add_argument('-seed', dest='seed', default=41504, type=int, help='Seed for randomization')

	parser.add_argument('-restore', dest='restore', action='store_true', help='Restore from the previously saved model')
	parser.add_argument('-bias', dest='bias', action='store_true', help='Whether to use bias in the model')

	parser.add_argument('-num_bases', dest='num_bases', default=-1, type=int,
						help='Number of basis relation vectors to use')
	parser.add_argument('-init_dim', dest='init_dim', default=100, type=int,
						help='Initial dimension size for entities and relations')
	parser.add_argument('-gcn_dim', dest='gcn_dim', default=200, type=int, help='Number of hidden units in GCN')
	parser.add_argument('-embed_dim', dest='embed_dim', default=None, type=int,
						help='Embedding dimension to give as input to score function')
	parser.add_argument('-gcn_layer', dest='gcn_layer', default=1, type=int, help='Number of GCN Layers to use')
	parser.add_argument('-gcn_drop', dest='dropout', default=0.1, type=float, help='Dropout to use in GCN Layer')
	parser.add_argument('-hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')

	# ConvE specific hyperparameters
	parser.add_argument('-hid_drop2', dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
	parser.add_argument('-feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
	parser.add_argument('-k_w', dest='k_w', default=10, type=int, help='ConvE: k_w')
	parser.add_argument('-k_h', dest='k_h', default=20, type=int, help='ConvE: k_h')
	parser.add_argument('-num_filt', dest='num_filt', default=200, type=int,
						help='ConvE: Number of filters in convolution')
	parser.add_argument('-ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')

	parser.add_argument('-logdir', dest='log_dir', default='./log/', help='Log directory')
	parser.add_argument('-config', dest='config_dir', default='./config/', help='Config directory')

	# arg_list = ['-score_func', 'distmult', '-opn', 'mult', '-gamma', '9', '-hid_drop', '0.2', '-init_dim', '200', '-lr', '1e-3', '-data', 'FB15k-237', '-batch', '100', '-epoch', '20']
	# args = parser.parse_args(args=arg_list)
	##@@
	args = parser.parse_args()

	return args

if __name__ == '__main__':
	args = load_args()

	if not args.restore: args.name = args.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')

	set_gpu(args.gpu)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	model = Runner(args)
	model.fit()
