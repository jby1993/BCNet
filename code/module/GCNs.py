import torch
import torch.nn as nn
from torch.nn import Parameter, Module
import torch.nn.functional as F
import torch_geometric.nn as geo_nn
import torch_geometric.utils as geo_utils
from torch_geometric.utils import remove_self_loops, add_self_loops
import math

class ResidualAdd(Module):
	def __init__(self,conv1,conv2,act1,act2):
		super(ResidualAdd,self).__init__()
		self.conv1=conv1
		self.conv2=conv2
		self.act1=act1
		self.act2=act2
	def forward(self,x,edge_index=None,**kwargs):
		if edge_index is not None:
			h=self.conv2(self.act1(self.conv1(x,edge_index)),edge_index)
		else:
			h=self.conv2(self.act1(self.conv1(x,**kwargs)),**kwargs)
		assert(h.shape==x.shape)
		return self.act2(h+x)

class Transform(Module):
	def __init__(self,infeatures,outfeatures,bias=True):
		super(Transform,self).__init__()
		self.weight=Parameter(torch.FloatTensor(infeatures,outfeatures))
		nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
		# geo_nn.inits.glorot(self.weight)
		if bias:
			self.bias=Parameter(torch.FloatTensor(1,outfeatures))
			self.bias.data.zero_()
		else:
			self.register_parameter('bias', None)
	#x:N,infeatures
	def forward(self,x):
		x = x.unsqueeze(0) if x.dim() == 1 else x
		out=torch.matmul(x,self.weight)
		if self.bias is not None:
			out=out+self.bias
		return out

class MultiPerceptro(Module):
	def __init__(self,dims,bias=True):
		super(MultiPerceptro,self).__init__()
		assert(len(dims)>=2)
		self.Tlist=nn.ModuleList()
		for ind in range(len(dims)-1):
			dim1=dims[ind]
			dim2=dims[ind+1]
			self.Tlist.append(Transform(dim1,dim2,bias))
	def forward(self,x):
		x = x.unsqueeze(0) if x.dim() == 1 else x
		for ind,trans in enumerate(self.Tlist):
			if ind < len(self.Tlist)-1:
				x=F.relu(trans(x))
			else:
				x=trans(x)
		return x


#spiral conv, x should pading one dim with zero to make spiral conv correct
class SpiralConv(Module):
	def __init__(self,in_c,out_c,spiral_indices,activation='relu',bias=True):
		super(SpiralConv,self).__init__()
		self.in_c = in_c
		self.out_c = out_c
		self.spiral_size = spiral_indices.shape[-1]

		self.conv = nn.Linear(in_c*self.spiral_size,out_c,bias=bias)
		self.register_buffer('spiral_adj',torch.from_numpy(spiral_indices))
		if activation == 'relu':
			self.activation = nn.ReLU()
		elif activation == 'elu':
			self.activation = nn.ELU()
		elif activation == 'leaky_relu':
			self.activation = nn.LeakyReLU(0.02)
		elif activation == 'sigmoid':
			self.activation = nn.Sigmoid()
		elif activation == 'tanh':
			self.activation = nn.Tanh()
		elif activation == 'identity':
			self.activation = lambda x: x
		else:
			raise NotImplementedError()
	def forward(self,x,**kwargs):
		bsize, num_pts, feats = x.size()
		_, _, spiral_size = self.spiral_adj.size()
		spiral_adj=self.spiral_adj.expand(bsize,num_pts,spiral_size)

		spirals_index = spiral_adj.reshape(bsize*num_pts*spiral_size) # [1d array of batch,vertx,vertx-adj]
		batch_index = torch.arange(bsize, device=x.device).view(-1,1).repeat([1,num_pts*spiral_size]).view(-1).long() # [0*numpt,1*numpt,etc.]
		spirals = x[batch_index,spirals_index,:].view(bsize*num_pts,spiral_size*feats) # [bsize*numpt, spiral*feats]


		out_feat = self.conv(spirals)
		out_feat = self.activation(out_feat)

		out_feat = out_feat.view(bsize,num_pts,self.out_c)
		if 'zero_padding' in kwargs:
			zero_padding=kwargs['zero_padding']
		else:
			zero_padding = x.new_ones(1,x.size(1),1)
			zero_padding[0,-1,0] = 0.0
		out_feat = out_feat * zero_padding

		return out_feat
