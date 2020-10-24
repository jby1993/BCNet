import os
os.sys.path.append('../')
import torch
import os.path as osp
import numpy as np
import torch
import pickle
from module.SkinWeightModel import SkinWeightNet
import module.ImageReconstructModel as M
import os
from glob import glob
import argparse
import cv2
def save_obj(ps,tris,name):
	with open(name, 'w') as fp:
		for v in ps:
			fp.write( 'v {:f} {:f} {:f}\n'.format( v[0], v[1], v[2]) )
		if tris is not None:
			for f in tris: # Faces are 1-based, not 0-based in obj files
				fp.write( 'f {:d} {:d} {:d}\n'.format(f[0] + 1, f[1] + 1, f[2] + 1) )

def save_batch_objs(bps,face_index,batch,names):
	if len(bps.shape)==2:
		assert(len(names)==batch.max()+1)
		voffset=0
		for ind in range(len(names)):
			select=batch==ind
			vnum=select.sum()
			tris=face_index[(face_index>=voffset) * (face_index<voffset+vnum)].reshape(-1,3)-voffset
			ps=bps[select]
			save_obj(ps,tris,names[ind])
			voffset+=vnum
	elif len(bps.shape)==3:
		assert(bps.shape[0]==len(names))
		for ps,n in zip(bps,names):
			save_obj(ps,face_index,n)
	else:
		assert(False)

def read_img(file):
	img=cv2.imread(file)
	h=img.shape[0]
	w=img.shape[1]
	if h!=w:
		l=max(h,w)
		nimg=np.zeros((l,l,3),np.uint8)
		hs=max(int((l-h)/2.),0)
		he=min(int((l+h)/2.),l)
		he=min(he,hs+h)
		ws=max(int((l-w)/2.),0)
		we=min(int((l+w)/2.),l)
		we=min(we,ws+w)
		nimg[hs:he,ws:we]=img[:he-hs,:we-ws]
	else:
		nimg=img
	nimg=cv2.resize(nimg,(540,540))
	nimg=nimg.transpose(2,0,1)
	nimg=nimg.astype(np.float32)/255.
	return nimg

parser = argparse.ArgumentParser(description='img rec comparing')
parser.add_argument('--gpu-id',default=0,type=int,metavar='ID',
                    help='gpu id')
parser.add_argument('--save-root',default=None,metavar='M',
                    help='save root')
parser.add_argument('--inputs',default=None,metavar='M',
                    help='read inputs')
args = parser.parse_args()
inputs=args.inputs
img_files=[]
img_gtypes=[]
if osp.isdir(inputs):
	img_files.extend(glob(osp.join(inputs,'*.jpg')))
	img_files.extend(glob(osp.join(inputs,'*.png')))
	for imf in img_files:
		temp=imf[:-4]+'_gtypes.txt'
		if osp.isfile(temp):
			with open(temp,'r') as ff:
				img_gtypes.append([int(v) for v in ff.read().split()])
		else:
			img_gtypes.append([-1,-1])
elif osp.isfile(inputs):
	with open(inputs,'r') as ff:
		temp=ff.read().split('\n')				
		for line in temp:
			line=line.split()
			if len(line)>0:
				img_files.append(line[0])
			temp=[]
			if len(line)>1 and line[1].isdigit():
				temp.append(int(line[1]))
			else:
				temp.append(-1)
			if len(line)>2 and line[2].isdigit():
				temp.append(int(line[2]))
			else:
				temp.append(-1)
			img_gtypes.append(temp)

if len(img_files)==0:
	print('zeros img files, exit.')
	exit()


save_root=args.save_root
if not osp.isdir(save_root):
	os.makedirs(save_root)
batch_size=20
if args.gpu_id==None:
	device=torch.device('cpu')
else:
	device=torch.device(args.gpu_id)

skinWsNet=SkinWeightNet(4,True)
net=M.ImageReconstructModel(skinWsNet,True)
net.load_state_dict(torch.load('../models/garNet.pth',map_location='cpu'),True)
net=net.to(device)
net.eval()

# img_files=glob('MGN_datas/*.jpg')
batch_num=len(img_files)//batch_size
if batch_num*batch_size<len(img_files):
	batch_num+=1
# save_num=20
dis_ablation=False
print('total %d imgfiles'%len(img_files))
with torch.no_grad():	
	for batch_id in range(0,batch_num):
		s_id=batch_id*batch_size
		e_id=s_id+batch_size
		if e_id>len(img_files):
			e_id=len(img_files)
		batch_files=img_files[s_id:e_id]
		imgs=[]
		for file in batch_files:
			imgs.append(read_img(file))
		imgs=torch.from_numpy(np.stack(imgs,axis=0)).to(device)
		gps_pca,gps_diss,gps_rec,ws,shape_rec,pose_rec,tran_rec,pca_perg,displacement,body_js,body_ns,body_ps,_,_=\
			net(imgs,gtypes=img_gtypes[s_id:e_id])
		face_index=net.face_index.cpu().numpy()
		garbatch=net.garbatch.cpu().numpy()
		names=[]
		names_body=[]
		for ind,file in enumerate(batch_files):
			basename=osp.splitext(osp.basename(file))[0]
			names.append(osp.join(save_root,basename+'_up.obj'))
			names.append(osp.join(save_root,basename+'_bottom.obj'))
			names_body.append(osp.join(save_root,basename+'_smpl.obj'))

		save_batch_objs(gps_rec.cpu().numpy(),face_index,garbatch,names)
		save_batch_objs(body_ps.cpu().numpy(),net.smpl.faces,None,names_body)
		print(batch_id)
print('done.')
