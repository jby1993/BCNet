import torch
import torch_scatter
import numpy as np

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

def compute_connectivity_infos_from_mesh(mesh,device=None):
	if type(mesh) is str:
		mesh=om.read_trimesh(mesh)
	face_index=torch.from_numpy(mesh.face_vertex_indices().astype(np.int64))
	vf_fid=torch.zeros(0,dtype=torch.long)
	vf_vid=torch.zeros(0,dtype=torch.long)
	for vid,fids in enumerate(mesh.vertex_face_indices()):
		fids=torch.from_numpy(fids[fids>=0]).to(torch.long)
		vf_fid=torch.cat((vf_fid,fids),dim=0)
		vf_vid=torch.cat((vf_vid,fids.new_ones(fids.shape)*vid),dim=0)
	if device is not None:
		face_index=face_index.to(device)
		vf_fid=vf_fid.to(device)
		vf_vid=vf_vid.to(device)
	return face_index,vf_fid,vf_vid

#verts:(v,3) or (b,v,3), tri_fs:(f,3)
def compute_fnorms(verts,tri_fs):
	v0=verts.index_select(-2,tri_fs[:,0])
	v1=verts.index_select(-2,tri_fs[:,1])
	v2=verts.index_select(-2,tri_fs[:,2])
	e01=v1-v0
	e02=v2-v0
	fnorms=torch.cross(e01,e02,-1)
	diss=fnorms.norm(2,-1).unsqueeze(-1)
	diss=torch.clamp(diss,min=1.e-6,max=float('inf'))
	fnorms=fnorms/diss
	return fnorms


def compute_vnorms(verts,tri_fs,vertex_index,face_index):
	fnorms=compute_fnorms(verts,tri_fs)
	vnorms=torch_scatter.scatter(fnorms.index_select(-2,face_index),vertex_index,-2,None,verts.shape[-2])
	diss=vnorms.norm(2,-1).unsqueeze(-1)
	diss=torch.clamp(diss,min=1.e-6,max=float('inf'))
	vnorms=vnorms/diss
	return vnorms

def Geman_McClure_Loss(x,c):
	return x*x*2.0/c/c/(x*x/c/c + 4.)

