from torch_geometric.utils import to_scipy_sparse_matrix
from helper import *
from torch.utils.data import Dataset
import os.path as osp
from torch_geometric.data import Data
from itertools import chain
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from scipy.sparse.csgraph import shortest_path

def edge_index_to_triple(edge_index,edge_type):
    pass

class RLPDDataset(Dataset):
    def __init__(self, data, num_hops, cached_path=None, split='train',device='cpu'):
        super().__init__()
        self.device = device
        self._max_z = None
        self.raw_data = data
        if cached_path is None:
            self.cached_path = osp.join(osp.dirname(osp.realpath(__file__)), '', 'data', 'RLPD','FB15k-237','cached',split+'.pt')
        self.processed_data = []
        self.num_hops = num_hops
        self.process_example()

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        subgraphs = self.processed_data[idx]
        return subgraphs

    def process_example(self):
        if os.path.exists(self.cached_path):
            print("cached files found in {0}".format(self.cached_path))
            self.processed_data = torch.load(self.cached_path)
            return

        self._max_z = 0

        # Collect a list of subgraphs for training, validation and testing:
        full_data = Data(edge_index=self.raw_data.edge_index, edge_type=self.raw_data.edge_type,num_nodes=self.raw_data.num_nodes)
        pos_data = Data(edge_index=self.raw_data.pos_edge_label_index,edge_type = self.raw_data.neg_edge_label_type,num_nodes=self.raw_data.num_nodes)
        neg_data = Data(edge_index=self.raw_data.neg_edge_label_index,edge_type = self.raw_data.pos_edge_label_type,num_nodes=self.raw_data.num_nodes)

        self.self_loop_edge_type_value = full_data.edge_type.max() + 1  # 添加一种新的关系类型：自环关系

        subgraphs = self.generate_node_subgraph(full_data)
        pos_data_list = self.extract_enclosing_subgraphs(subgraphs,pos_data, 1,device=self.device,data_type="pos")
        neg_data_list = self.extract_enclosing_subgraphs(subgraphs,neg_data, 0,device=self.device,data_type="neg")

        # Convert node labeling to one-hot features. [no need anymore ,already done in extract_enclosing_subgraphs]
        # for data in chain(pos_data_list,neg_data_list):
        #     # We solely learn links from structure, dropping any node features:
        #     data.x = F.one_hot(data.z, self._max_z + 1).to(torch.float)

        self.processed_data = pos_data_list + neg_data_list

        torch.save(self.processed_data, self.cached_path)

    def relabel_node_index(self,graph):
        # extract unique node indices from edge_index of current graph
        uni_nodes_indices = list(set(torch.cat([graph.edge_index[0], graph.edge_index[1]]).tolist()))
        uni_nodes_indices = torch.tensor(uni_nodes_indices).to(graph.edge_index.device)
        # graph.full_graph_num_nodes = graph.num_nodes
        graph.num_nodes = uni_nodes_indices.shape[-1]

        graph.original_node_index = uni_nodes_indices
        # replace new node indices with old node indices in edge_index
        new_edge_index = graph.edge_index.clone().detach()
        for new_node_index, old_node_index in enumerate(graph.original_node_index):
            location_need_to_be_replaced_mask1 = (new_edge_index[0] == old_node_index)
            location_need_to_be_replaced_mask2 = (new_edge_index[1] == old_node_index)

            new_edge_index[0][location_need_to_be_replaced_mask1] = new_node_index
            new_edge_index[1][location_need_to_be_replaced_mask2] = new_node_index
        graph.original_edge_index = graph.edge_index
        graph.edge_index = new_edge_index

        return graph,uni_nodes_indices

    def generate_node_subgraph(self,full_data):
        device = self.device
        # 预先对所有节点进行采样，获取其子图并根据edge_type剪枝
        subgraphs = []  # 子图按节点的索引排列，例如index为0的节点子图为subgraph[0]

        global_edge_type = full_data.edge_type.clone()
        full_data.edge_type = None  # 不能让full_data包含edge_type字段，否则NeighborLoader采样会依据edge_type进行采样
        num_neighbors = [-1, -1]  # 二阶邻居全采样
        num_edge_type = 3  # 每种边类型保留几条边

        # 节点和边的全局id
        full_data.n_id = torch.arange(full_data.num_nodes).to(device)  # check description of NeighborLoader
        full_data.e_id = torch.tensor(range(global_edge_type.shape[-1])).to(device)

        # input_nodes = torch.tensor(range(full_data.num_nodes))
        # loader = NeighborLoader(full_data, num_neighbors=[-1], batch_size=1, input_nodes=input_nodes)
        # count=0
        # for i, subgraph in tqdm(enumerate(loader), desc='extracting subgraphs for nodes', total=full_data.num_nodes):
        #     num_edges = subgraph.edge_index.shape[-1]
        #     if num_edges <=1:
        #         count+=1
        # print(count)

        input_nodes = torch.tensor(range(full_data.num_nodes))
        loader = NeighborLoader(full_data, num_neighbors=num_neighbors, batch_size=1, input_nodes=input_nodes)
        # @@@
        # input_nodes = torch.tensor([564, 4994]) # src=8188 dst=13449 src=1896 dst=5753 src=8598 dst=13470
        # loader = NeighborLoader(full_data, num_neighbors=num_neighbors, batch_size=1,input_nodes=input_nodes)

        # for i, subgraph in tqdm(enumerate(loader),desc='extracting subgraphs for nodes',total=2):
        # @@@
        for i, subgraph in tqdm(enumerate(loader), desc='extracting subgraphs for nodes', total=full_data.num_nodes):
            # re-fetch index
            subgraph.edge_index = full_data.edge_index[:, subgraph.e_id]
            subgraph.edge_type = global_edge_type[subgraph.e_id]  # 获取重建索引边在全图中对应的边

            # subgraph.center_node_index = (subgraph.n_id==input_nodes[i]).nonzero().squeeze().item() # 中心节点在子图的索引
            # subgraph.center_node_original_index = input_nodes[i].item() # 中心节点在全图的索引

            # 根据规则筛选边
            indices_of_chosen_edges = sample_indices(subgraph.edge_type, num_edge_type)
            if indices_of_chosen_edges is None:  # 对于孤立节点，加入自环
                subgraph.edge_type = torch.tensor([self.self_loop_edge_type_value], device=device)
                subgraph.edge_index = torch.tensor([[i], [i]], device=device)
            else:
                subgraph.edge_type = subgraph.edge_type[indices_of_chosen_edges]
                subgraph.edge_index = subgraph.edge_index[:, indices_of_chosen_edges]

            # remove redundant staff
            subgraph.num_nodes = None
            subgraph.e_id = None
            subgraph.n_id = None
            subgraph.batch_size = None

            subgraphs.append(subgraph)
        return subgraphs

    def extract_enclosing_subgraphs(self, subgraphs, data, y, device,data_type):
        data_list = []

        # 合并每条边的子图，然后打标签并计算Z
        i = 0
        edge_bar = tqdm(data.edge_index.t().tolist(),desc='merging subgraphs for node pairs '+ data_type)
        for src, dst in edge_bar:

            # src = 564
            # dst = 4994
            # src_subgraph = subgraphs[0]
            # dst_subgraph = subgraphs[1]
            # @@@
            src_subgraph = subgraphs[src]
            dst_subgraph = subgraphs[dst]

            # 合并两个节点的子图并添加自环关系
            sub_edge_index = torch.cat([src_subgraph.edge_index, dst_subgraph.edge_index], dim=1)
            sub_edge_type = torch.cat([src_subgraph.edge_type, dst_subgraph.edge_type], dim=0)
            sub_node_index = torch.unique(sub_edge_index)
            self_loop_edge_index = torch.tensor([[src,dst],[src,dst]],device=device)
            self_loop_edge_type = torch.tensor([self.self_loop_edge_type_value.item()]*2,device=device)
            sub_edge_index = torch.cat([sub_edge_index, self_loop_edge_index], dim=1)
            sub_edge_type = torch.cat([sub_edge_type, self_loop_edge_type], dim=0)

            # 移除当前节点对, 是为了在这个子图上预测源节点和目标节点是否应该连接。如果保留原有的边, 等于数据泄露, 模型可以直接利用这个信息做预测, 失去模型训练的意义
            mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
            mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
            sub_edge_index = sub_edge_index[:, mask1 & mask2]
            sub_edge_type = sub_edge_type[mask2 & mask1]

            # 移除重复边
            triples = torch.vstack([sub_edge_index, sub_edge_type.reshape(1,-1)]) # 将其堆叠成三元组
            unique_triples = torch.from_numpy(np.unique(triples.cpu().numpy(), axis=1)).to(device) ## sorted
            subgraph = Data(edge_index=unique_triples[[0,1],:],edge_type=unique_triples[2,:])
            unique_node_index = subgraph.edge_index.unique()
            subgraph.num_nodes = unique_node_index.shape[-1]

            if src not in unique_node_index or dst not in unique_node_index:
                print("seed nodes info missed")

            # 将子图中的节点重新标记索引, 即子图节点的索引将重新按0到 N-1 排序, 而不是保留原始图的节点索引
            # 一些图学习算法要求节点从0开始排序, 这样可以重新标记节点, 避免原索引不从0开始的情况。
            subgraph,sub_node_index = self.relabel_node_index(subgraph)

            src1 = src
            dst1 = dst
            try:
                src = (sub_node_index==src).nonzero().item()
                dst = (sub_node_index==dst).nonzero().item()
                # Calculate node labeling for each node in the subgraph.
                z = self.drnl_node_labeling(subgraph.edge_index, src, dst,
                                        num_nodes=subgraph.num_nodes)
                subgraph.z = z.to(device)
                subgraph.label = y
                # subgraph.z = F.one_hot(z, self._max_z + 1).to(torch.float).to(device) # 极其耗时，算逑
                data_list.append(subgraph)
            except Exception:
                print("error! src={0} dst={1}".format(src1,dst1))
            if i==1000: # for debugging
                return data_list
            i+=1
        return data_list

    def drnl_node_labeling(self, edge_index, src, dst, num_nodes=None):
        # Double-radius node labeling (DRNL).
        src, dst = (dst, src) if src > dst else (src, dst)
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

        idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
        adj_wo_src = adj[idx, :][:, idx]

        idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
        adj_wo_dst = adj[idx, :][:, idx]

        dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True,
                                 indices=src)
        dist2src = np.insert(dist2src, dst, 0, axis=0)
        dist2src = torch.from_numpy(dist2src)

        dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True,
                                 indices=dst - 1)
        dist2dst = np.insert(dist2dst, src, 0, axis=0)
        dist2dst = torch.from_numpy(dist2dst)

        dist = dist2src + dist2dst
        dist_over_2, dist_mod_2 = torch.div(dist, 2, rounding_mode='floor'), dist % 2
        # dist_over_2, dist_mod_2 = dist // 2, dist % 2

        z = 1 + torch.min(dist2src, dist2dst)
        z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
        z[src] = 1.
        z[dst] = 1.
        z[torch.isnan(z)] = 0.

        self._max_z = max(int(z.max()), self._max_z)
        return z.to(torch.long)

class TrainDataset(Dataset):
	"""
	Training Dataset class.

	Parameters
	----------
	triples:	The triples used for training the model
	params:		Parameters for the experiments
	
	Returns
	-------
	A training Dataset class instance used by DataLoader
	"""
	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params
		self.entities	= np.arange(self.p.num_ent, dtype=np.int32)

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele			= self.triples[idx]
		triple, label, sub_samp	= torch.LongTensor(ele['triple']), np.int32(ele['label']), np.float32(ele['sub_samp'])
		trp_label		= self.get_label(label) # what the hell is 'sub_samp'

		if self.p.lbl_smooth != 0.0:
			trp_label = (1.0 - self.p.lbl_smooth)*trp_label + (1.0/self.p.num_ent)

		return triple, trp_label, None, None

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		trp_label	= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, trp_label
	
	def get_neg_ent(self, triple, label):
		def get(triple, label):
			pos_obj		= label
			mask		= np.ones([self.p.num_ent], dtype=np.bool)
			mask[label]	= 0
			neg_ent		= np.int32(np.random.choice(self.entities[mask], self.p.neg_num - len(label), replace=False)).reshape([-1])
			neg_ent		= np.concatenate((pos_obj.reshape([-1]), neg_ent))

			return neg_ent

		neg_ent = get(triple, label)
		return neg_ent

	def get_label(self, label):
		y = np.zeros([self.p.num_ent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)

class TestDataset(Dataset):
	"""
	Evaluation Dataset class.

	Parameters
	----------
	triples:	The triples used for evaluating the model
	params:		Parameters for the experiments
	
	Returns
	-------
	An evaluation Dataset class instance used by DataLoader for model evaluation
	"""
	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele		= self.triples[idx]
		triple, label	= torch.LongTensor(ele['triple']), np.int32(ele['label'])
		label		= self.get_label(label)

		return triple, label

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		label		= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, label
	
	def get_label(self, label):
		y = np.zeros([self.p.num_ent], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)