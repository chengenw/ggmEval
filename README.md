# Graph Generative Models Evaluation with Masked Autoencoder

Pytorch implementation for [*Graph Generative Models Evaluation with Masked Autoencoder*](https://arxiv.org/abs/2503.13271). The paper has been accepted to PAKDD 2025 Workshop: Graph Learning with Foundation Models (GLFM).

The implementation is based on the code from [Evaluating Graph Generative Models with Contrastively Learned Features](https://github.com/hamed1375/Self-Supervised-Models-for-GGM-Evaluation), [GraphMAE](https://github.com/THUDM/GraphMAE) and [MaskGAE](https://github.com/EdisonLeeeee/MaskGAE).

## Environment

Create a new Conda environment and install PyTorch:

```bash
conda create -n ggm python=3.8 --yes
conda activate ggm
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch --yes
```

Install other libraries:

```bash
pip install dgl_cu102==0.9.1 -f https://data.dgl.ai/wheels/repo.html
pip install torch_scatter==2.1.0 torch_sparse==0.6.16 torch_cluster==1.6.0 torch_spline_conv==1.2.1 -f https://data.pyg.org/whl/torch-1.12.1%2Bcu102.html 
pip install git+https://github.com/fabriziocosta/EDeN.git
pip install torch_geometric==2.2.0 chardet pyemd GraKeL ray PyGCL seaborn setGPU termcolor tensorboardX ogb tensorboard
```

## Datasets

The processed datasets are under folder `data/`.

## Evaluation

Evaulate with random graph neural networks
```bash
python main.py --seed=0 --permutation_type=mixing-random --dataset=PROTEINS --feature_extractor=gin-random --results_directory=testing_no_feats --use_predefined_hyperparams=True
```

Evaluate with contrastive graph learning:
```bash
python main.py --seed=0 --permutation_type=mixing-random --dataset=PROTEINS --feature_extractor=graphcl --results_directory=testing_deg_feats --deg_feats --use_predefined_hyperparams=True
```

Evaluate with Graph Masked Autoencoder:

```bash
python main.py --seed=4 --permutation_type=rewiring-edges --dataset=PROTEINS --results_directory=testing_clus_feats --deg_feats --clus_feats --orbit_feats graphmae
```

`--permutation_type`: permutation type: `mixing-random`, `mode-collapse`, `mode-dropping`, or `rewiring-edges`.

`--dataset`: `REDDIT-MULTI-5K`, `DBLP v1`, or `PROTEINS`.

`--feature_extractor`: random graph neural networks `gin-random` or contrastive graph learning `graphcl`. Note: Graph Masked Autoencoder `graphmae` command format is different

`--deg_feats`: use degree features

`--clus_feats`: use clustering features

`--orbit_feats`: use orbit_features

The results are saved in a folder under `experiment_results/`.

Use `create_bash_script.py` to create batch commands.

Use `Experiments Visualization.py` to generate figures.

## Citation

```bibtex
@article{wang2025graph,
  title={Graph Generative Models Evaluation with Masked Autoencoder},
  author={Wang, Chengen and Kantarcioglu, Murat},
  journal={arXiv preprint arXiv:2503.13271},
  year={2025}
}
```
