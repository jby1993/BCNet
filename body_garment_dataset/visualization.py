import os
import os.path as osp
import numpy as np
import random
import torch
from vedo import *
import vtk
os.sys.path.append('../')
from smpl_pytorch.SMPL import getSMPL,getTmpFile

# garvnums=[4248,4258,5327,3721,5404,2818]
gartypes=['shirts','short_shirts','pants','short_pants','skirts','short_skirts']
garPcaDatas={}
for gar in gartypes:
	garPcaDatas[gar]=None
smpl=getSMPL()

def decode_info_folder(folder):
	SPRING=folder.split('_')[0]
	#up
	if 'short_shirts' in folder:
		up='short_shirts'		
	elif 'shirts' in folder:
		up='shirts'
	up_id_str=folder[folder.find(up)+len(up)+1]
	assert(int(up_id_str) in [1,2,3])
	#bottom
	if 'short_pants' in folder:
		bottom='short_pants'
	elif 'short_skirts' in folder:
		bottom='short_skirts'
	elif 'pants' in folder:
		bottom='pants'
	elif 'skirts' in folder:
		bottom='skirts'
	bottom_id_str=folder[folder.find(bottom)+len(bottom)+1]
	motion_str=folder[folder.find(bottom)+len(bottom)+3:]
	return SPRING,up,up_id_str,bottom,bottom_id_str,motion_str

def pca_verts(gartype,pca):
	if garPcaDatas[gartype] is None:
		data=np.load(osp.join('tmps',gartype,'pca_data.npz'))
		garPcaDatas[gartype]=[data['mean'],data['components']]
	verts=(pca.reshape(1,-1)@garPcaDatas[gartype][1]+garPcaDatas[gartype][0].reshape(1,-1))
	return verts.reshape(-1,3)


def read_data():
	global upN,upP,bottomN,bottomP,smplN,smplP,image
	imgfile=imgfiles[current_fid]
	print('%d: '%current_fid+imgfile)
	image._update(Picture(imgfile)._data)

	info=imgfile[imgfile.find('SPRING'):imgfile.find('/vmode')]
	SPRING,up,up_id_str,bottom,bottom_id_str,motion_str=decode_info_folder(info)
	up_verts=pca_verts(up,np.load(osp.join(up,SPRING,up_id_str,'pca_param.npy')))
	bottom_verts=pca_verts(bottom,np.load(osp.join(bottom,SPRING,bottom_id_str,'pca_param.npy')))	
	
	gartmps[up].points(pts=up_verts)
	polyCopy = vtk.vtkPolyData()
	polyCopy.DeepCopy(gartmps[up].polydata(False))
	upN._update(polyCopy)
	gartmps[bottom].points(pts=bottom_verts)
	polyCopy = vtk.vtkPolyData()
	polyCopy.DeepCopy(gartmps[bottom].polydata(False))
	bottomN._update(polyCopy)

	temp=np.load(osp.join('motion_datas/all_train_datas/%d.npz'%current_fid))
	shape=temp['shape']
	pose=temp['pose']	
	_,vs=smpl.skeleton(torch.from_numpy(shape).view(1,-1),True)
	smplN.points(pts=vs.numpy().reshape(-1,3))
	vs,_,_=smpl(torch.from_numpy(shape).view(1,-1),torch.from_numpy(pose).view(1,-1),True)
	vs=vs.numpy().reshape(-1,3)
	smplP.points(pts=vs)
	smplP.rotateX(180)
	tran=temp['tran'].reshape(-1,3)	
	gartmps[up].points(pts=temp['up']-tran)
	polyCopy = vtk.vtkPolyData()
	polyCopy.DeepCopy(gartmps[up].polydata(False))
	upP._update(polyCopy)
	gartmps[bottom].points(pts=temp['bottom']-tran)
	polyCopy = vtk.vtkPolyData()
	polyCopy.DeepCopy(gartmps[bottom].polydata(False))
	bottomP._update(polyCopy)

def	upNvis():
	uNbu.switch()
	if uNbu.status()=='Uhide':
		upN.off()
	elif uNbu.status()=='Ushow':
		upN.on()

def smplNvis():
	smplNbu.switch()
	if smplNbu.status()=='Shide':
		smplN.off()
	elif smplNbu.status()=='Sshow':
		smplN.on()

def	bottomNvis():
	bNbu.switch()
	if bNbu.status()=='Bhide':
		bottomN.off()
	elif bNbu.status()=='Bshow':
		bottomN.on()

def	upPvis():
	uPbu.switch()
	if uPbu.status()=='Uhide':
		upP.off()
	elif uPbu.status()=='Ushow':
		upP.on()

def smplPvis():
	smplPbu.switch()
	if smplPbu.status()=='Shide':
		smplP.off()
	elif smplPbu.status()=='Sshow':
		smplP.on()

def	bottomPvis():
	bPbu.switch()
	if bPbu.status()=='Bhide':
		bottomP.off()
	elif bPbu.status()=='Bshow':
		bottomP.on()

def sample():
	global current_fid
	current_fid=random.choice(list(range(max_fid)))
	read_data()

with open('motion_datas/imgfiles.txt','r') as ff:
	imgfiles=ff.read().split('\n')
	imgfiles=[osp.join('motion_datas',file) for file in imgfiles]
current_fid=0
max_fid=len(imgfiles)


custom_shape = [dict(bottomleft=(0.0,0.0), topright=(0.33,1), bg='w', bg2='w'), dict(bottomleft=(0.33,0.0), topright=(0.66,1.0), bg='w', bg2='w'), dict(bottomleft=(0.66,0.0), topright=(1.0,1.0), bg='w', bg2='w')]
vp=Plotter(shape=custom_shape, axes=0, sharecam=False, size=(1800,600))

temp=vp.load([osp.join('tmps',gartype,'garment_tmp.obj') for gartype in gartypes])
gartmps={}
for gartype,mesh in zip(gartypes,temp):
	gartmps[gartype]=mesh

image=vp.load(imgfiles[current_fid])
upN=vp.load(osp.join('tmps',gartypes[0],'garment_tmp.obj'))
bottomN=vp.load(osp.join('tmps',gartypes[0],'garment_tmp.obj'))
upP=vp.load(osp.join('tmps',gartypes[0],'garment_tmp.obj'))
bottomP=vp.load(osp.join('tmps',gartypes[0],'garment_tmp.obj'))
smplN=vp.load(getTmpFile())
smplP=vp.load(getTmpFile())
upN.c('blue').lw(0.2)
bottomN.c('cyan').lw(0.2)
upP.c('blue').lw(0.2)
bottomP.c('cyan').lw(0.2)
smplN.c('yellow')
smplP.c('yellow')
upP.rotateX(180)
bottomP.rotateX(180)

read_data()


vp.renderer=vp.renderers[0]
uNbu = vp.addButton(
	upNvis,
	pos=(0.2, 0.05),  # x,y fraction from bottom left corner
	states=["Ushow", "Uhide"],
	c=["w", "w"],
	bc=["dg", "dv"],  # colors of states
	font="courier",   # arial, courier, times
	size=25,
	bold=True,
	italic=False,
)
smplNbu = vp.addButton(
	smplNvis,
	pos=(0.5, 0.05),  # x,y fraction from bottom left corner
	states=["Sshow", "Shide"],
	c=["w", "w"],
	bc=["dg", "dv"],  # colors of states
	font="courier",   # arial, courier, times
	size=25,
	bold=True,
	italic=False,
)
bNbu = vp.addButton(
	bottomNvis,
	pos=(0.8, 0.05),  # x,y fraction from bottom left corner
	states=["Bshow", "Bhide"],
	c=["w", "w"],
	bc=["dg", "dv"],  # colors of states
	font="courier",   # arial, courier, times
	size=25,
	bold=True,
	italic=False,
)

vp.renderer=vp.renderers[1]
uPbu = vp.addButton(
	upPvis,
	pos=(0.2, 0.05),  # x,y fraction from bottom left corner
	states=["Ushow", "Uhide"],
	c=["w", "w"],
	bc=["dg", "dv"],  # colors of states
	font="courier",   # arial, courier, times
	size=25,
	bold=True,
	italic=False,
)
smplPbu = vp.addButton(
	smplPvis,
	pos=(0.5, 0.05),  # x,y fraction from bottom left corner
	states=["Sshow", "Shide"],
	c=["w", "w"],
	bc=["dg", "dv"],  # colors of states
	font="courier",   # arial, courier, times
	size=25,
	bold=True,
	italic=False,
)
bPbu = vp.addButton(
	bottomPvis,
	pos=(0.8, 0.05),  # x,y fraction from bottom left corner
	states=["Bshow", "Bhide"],
	c=["w", "w"],
	bc=["dg", "dv"],  # colors of states
	font="courier",   # arial, courier, times
	size=25,
	bold=True,
	italic=False,
)

vp.renderer=vp.renderers[0]
vp.addButton(
	sample,
	pos=(0.15, 0.85),  # x,y fraction from bottom left corner
	states=["sample"],
	c=["w"],
	bc=["dg"],  # colors of states
	font="courier",   # arial, courier, times
	size=25,
	bold=True,
	italic=False,
)

vp.show([upN,bottomN,smplN],axes=0,at=0)
vp.show([upP,bottomP,smplP],axes=0,at=1)
vp.show(image,axes=0,at=2)
vp.renderers[2].InteractiveOff()
interactive()