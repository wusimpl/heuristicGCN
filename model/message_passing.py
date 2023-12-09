import inspect, torch
from torch_scatter import scatter

def scatter_(name, src, index, dim_size=None):
	if name == 'add': name = 'sum'
	assert name in ['sum', 'mean', 'max']
	out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=name)
	return out[0] if isinstance(out, tuple) else out


class MessagePassing(torch.nn.Module):
	r"""Base class for creating message passing layers

	.. math::
		\mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
		\square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
		\left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

	where :math:`\square` denotes a differentiable, permutation invariant
	function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
	and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
	MLPs.
	See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
	create_gnn.html>`__ for the accompanying tutorial.

	"""

	def __init__(self, aggr='add'):
		super(MessagePassing, self).__init__()

		# fetch the name of arguments of these functions
		self.message_args = inspect.getargspec(self.message)[0][1:]	# In the defined message function: get the list of arguments as list of string| For eg. in rgcn this will be ['x_j', 'edge_type', 'edge_norm'] (arguments of message function)
		self.update_args  = inspect.getargspec(self.update)[0][2:]	# Same for update function starting from 3rd argument | first=self, second=out

	# extra args receive by kwargs are [x,edge_type,relation features, edge_norm, relation mode]
	def propagate(self, aggr, edge_index, **kwargs):
		assert aggr in ['add', 'mean', 'max']
		kwargs['edge_index'] = edge_index
		size = None
		message_args = []
		for arg in self.message_args:
			if arg[-2:] == '_i':					# If arguments ends with _i then include indic
				tmp  = kwargs[arg[:-2]]				# Take the front part of the variable | Mostly it will be 'x', 
				size = tmp.size(0)
				message_args.append(tmp[edge_index[0]])		# Lookup for head entities in edges # fetch features of head entities
			elif arg[-2:] == '_j':
				tmp  = kwargs[arg[:-2]]				# tmp = kwargs['x'] # fetch x
				size = tmp.size(0)
				message_args.append(tmp[edge_index[1]])		# Lookup for tail entities in edges 取出每条边的尾实体
			else:
				message_args.append(kwargs[arg])		# Take things from kwargs

		update_args = [kwargs[arg] for arg in self.update_args]		# Take update args from kwargs

		out = self.message(*message_args) # *: unpack a list or tuple into separate arguments when calling a function.
		out = scatter_(aggr, out, edge_index[0], dim_size=size)		# Aggregated neighbors for each vertex
		out = self.update(out, *update_args)

		return out

	def message(self, x_j):  # pragma: no cover
		r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
		for each edge in :math:`(i,j) \in \mathcal{E}`.
		Can take any argument which was initially passed to :meth:`propagate`.
		In addition, features can be lifted to the source node :math:`i` and
		target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
		variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

		return x_j

	def update(self, aggr_out):  # pragma: no cover
		r"""Updates node embeddings in analogy to
		:math:`\gamma_{\mathbf{\Theta}}` for each node
		:math:`i \in \mathcal{V}`.
		Takes in the output of aggregation as first argument and any argument
		which was initially passed to :meth:`propagate`."""

		return aggr_out
