import torch
import os
import torch.nn as nn
import numpy as np
#this prior is from keep it SMPL:...(SMPLify)
#see max_mixture_prior.py of SMPLify
class MaxMixturePosePrior(nn.Module):
	def __init__(self):
		super(MaxMixturePosePrior, self).__init__()
		self.prefix=3	#smpl first 3 pose parameters are root rotate, not need to constrain
		data=np.load(os.path.join(os.path.dirname(__file__),'gmm_data.npz'))
		self.register_buffer('precs',torch.from_numpy(data['precs']).float())	#this is covars inverse cholesky decompose matrix
		self.register_buffer('logweights',torch.from_numpy(data['logweights']).float())  #this is .py weights log value
		self.register_buffer('means',torch.from_numpy(data['means']).float())
		self.gmm_num=self.means.shape[0]	#this value should be 8
	def forward(self,theta):
		sqrt_0point5=0.70710678118
		theta=theta[:,self.prefix:]
		num,pdim=theta.shape
		delta=theta.expand(self.gmm_num,num,pdim).permute(1,0,2)		
		delta=sqrt_0point5*(delta-self.means)
		delta=delta.reshape(num*self.gmm_num,1,pdim)
		loglikelihoods=torch.bmm(delta,self.precs.expand(num,*self.precs.shape).reshape(num*self.gmm_num,pdim,pdim)).reshape(num,self.gmm_num,pdim)
		# min_indeices=torch.argmin((loglikelihoods*loglikelihoods).sum(-1)-self.logweights.view(1,-1),dim=-1)
		# out1=torch.gather(loglikelihoods,1,min_indeices.reshape(-1,1).expand(num,pdim).reshape(num,1,pdim)).reshape(num,pdim)
		# out2=torch.sqrt(-self.logweights[min_indeices])
		# return out1,out2
		results=(loglikelihoods*loglikelihoods).sum(-1)-self.logweights.view(1,-1)
		min_indeices=torch.argmin(results,dim=-1)
		return torch.gather(results,1,min_indeices.reshape(-1,1)).view(-1)
		
def PoseAngleConstrain(theta):
	return torch.exp(theta[:,55])+torch.exp(-theta[:,58])+torch.exp(-theta[:,12])+torch.exp(-theta[:,15])
