# Instalación de dependencias

## Requisitos
- Cuda 10.1 (+CUDNN compatible)
- Pytorch==1.6.0+cu101
- Pytorch geometric

## Instalación mediante pip

**NO SE DEBE UTILIZAR ENTORNO VIRTUAL, GENERA PROBLEMAS**

1. Instalar `pytorch` desde las versiones archivadas, buscar v1.6.0: https://pytorch.org/get-started/previous-versions/ Se puede instalar mediante el siguiente comando
```
# CUDA 10.1
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
2. Instalar `pytorch-geometric`, para la versión 1.6.0 de Pytorch, se puede encontrar los comandos necesarios en el siguiente enlace: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

```
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.7.0.html
pip install torch-geometric
```
**Nota: Se está dejando fuera `torch-geometric-temporal`, sin embargo, su instalación no afecta.**

3. Instalar `opencv` para python:
```
pip install opencv-python
```

4. Instalar `openmesh`:
```
pip install openmesh
```

## Comprobación de funcionamiento

Si todo ha resultado sin problemas, se puede correr el siguiente comando para probar el funcionamiento del programa. En la carpeta `images` se ingresan las fotos con los modelos y en la carpeta `recs` se obtiene los modelos SMPL y los modelos de prenda.

```
python infer.py --inputs ../images --save-root ../recs
```

### UTILS
Visualizado de `.obj` online:  https://www.graphics.rwth-aachen.de/software/openmesh/
