# Install packages for linux with cuda enabled

```bash
conda create --name ST-Imputation python=3.8
conda activate ST-Imputation
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install pandas==1.3.5
pip install pytorch_lightning==1.5.0
pip install torchmetrics==0.7
pip install torch-spatiotemporal==0.1.1
pip install matplotlib
pip install timm==0.4.12
```

# Install packages for M1 macOS.

```bash
CONDA_SUBDIR=osx-64 conda create --name tsl python=3.8
conda activate tsl
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install torch-spatiotemporal==0.1.1
pip install torch_geometric
pip install torch_scatter torch_sparse
pip install pandas==1.3.5
pip install pytorch_lightning==1.5.0
pip install torchmetrics==0.7
pip install matplotlib
pip install timm==0.4.12
```


