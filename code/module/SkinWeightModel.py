import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATConv,GCNConv,ChebConv,global_max_pool
from module.GCNs import ResidualAdd,MultiPerceptro

class SkinWeightNet(Module):
	def __init__(self,depth=4,use_normal=False):
		super(SkinWeightNet,self).__init__()
		self.d=depth
		self.use_normal=use_normal
		if not use_normal:
			self.inMLP=MultiPerceptro([3+24,48,64],False)
		else:
			self.inMLP=MultiPerceptro([3+3+24,48,64],False)
		self.ress=nn.ModuleList([ResidualAdd(GATConv(64,64,heads=2,concat=False),GATConv(64,64,heads=2,concat=False),nn.ReLU(),nn.ReLU()) for i in range(depth)])
		self.midMLP=MultiPerceptro([64,128,512],False)
		self.outMLP=MultiPerceptro([64+512,128,24],False)
	def forward(self,x,edge_index,batch):
		fs=self.inMLP(x)
		for ind,res in enumerate(self.ress):
			fs=res(fs,edge_index)
			if ind==0:
				mid=fs
		gl_fs=global_max_pool(self.midMLP(mid),batch)
		return F.softmax(self.outMLP(torch.cat((fs,gl_fs[batch,:]),-1)),dim=-1)

class NetWithLoss(Module):
	def __init__(self,net,l1=False):
		super(NetWithLoss,self).__init__()
		self.net=net
		self.l1=l1
	def forward(self,data):
		if self.net.use_normal:
			x=torch.cat((data.x,data.xn,data.xdis),-1)
		else:	
			x=torch.cat((data.x,data.xdis),-1)
		rec_weights=self.net(x,data.edge_index,data.batch)
		if self.l1:
			l1error=torch.abs(rec_weights-data.xws).mean()
		#deal with divide zero, use kld loss
		weights=torch.clamp(data.xws,min=1.0e-5,max=1.0)
		index=rec_weights>0.
		out=rec_weights.new_zeros(rec_weights.shape)
		out[index]=rec_weights[index]*torch.log(rec_weights[index]/weights[index])
		if self.l1:
			return (out.sum(dim=-1)).mean(),l1error
		else:
			return (out.sum(dim=-1)).mean()