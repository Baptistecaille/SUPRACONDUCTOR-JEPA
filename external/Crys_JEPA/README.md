# Crys_JEPA
This is the official implement of paper [Crys-JEPA: Accelerating Crystal Discovery via Embedding Screening and Generative Refinement](https://arxiv.org/pdf/2605.14759).

<img src="https://github.com/liun-online/Crys_JEPA/blob/main/model.png" width="800">

## Citation
```
@article{liu2026crysjepa,
  title         = {Crys-JEPA: Accelerating Crystal Discovery via Embedding Screening and Generative Refinement},
  author        = {Liu, Nian and Kazeev, Nikita and Dale, Stephen Gregory and Maevskiy, Artem and Zeng, Yuwei and Kubo, Ryoji and Huang, Pengru and Laurent, Thomas and LeCun, Yann and Novoselov, Kostya S. and Bresson, Xavier},
  journal       = {arXiv preprint},
  archivePrefix = {arXiv},
  eprint        = {2605.14759},
  year          = {2026}
}
```

## Python environment setup with Conda
```
conda create -n crys_jepa python=3.12.0
conda activate crys_jepa
conda install -c conda-forge mattersim==1.2.0
conda install -c conda-forge pymatgen=2025.6.14 ase=3.25.0 matminer=0.9.3
pip install --no-cache-dir --force-reinstall torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
pip install p_tqdm lmdb easydict einops atomate2
pip install setuptools==80.9.0 lmdb==1.6.2 scipy==1.16.2 smact==3.2.0

git clone https://github.com/liun-online/Crys_JEPA.git
cd Crys_JEPA/
```

## Download and Reproduce
Firstly, install [Hugging Face](https://huggingface.co/) and login
```
pip install -U huggingface_hub
hf auth login
```
Then, download the folders, i.e., `./data`, `./ref_dataset`, and `exp_logs.zip` by running the following commands:
```
hf download liun-online/Crys_JEPA --repo-type dataset --local-dir ./
mv ./ref_dataset ./eval/vsun/
```
The `exp_logs.zip` includes the checkpoints and intermediate data for reproducing the results in paper.

To reproduce V.S.U.N.
- MP-20: 44.9
- Alex-MP-20: 64.6
```
unzip ./exp_logs.zip
mv exp_logs logs
cd ./data
unzip ./exp_finetune.zip
mv exp_finetune finetune
cd ..

## MP-20
python VII_final_gen.py
python VIII_final_eval.py

## Alex-MP-20
python VII_final_gen.py --dataset alex_mp_20
python VIII_final_eval.py --dataset alex_mp_20

## [Optional] Run the commands below to prevent name clashes when starting a new training run
mv logs exp_logs
cd ./data
mv finetune exp_finetune
cd ..
```
Remarks
> 1. `python VII_final_gen.py` generates 10 batches of 1,000 crystals, stored in `./logs/finetune/mp_20/gen` and `./logs/finetune/alex_mp_20/gen`
> 2. Find results at `./logs/finetune/mp_20/eval/metrics_summary.log` and `./logs/finetune/alex_mp_20/eval/metrics_summary.log`

## Training Pipeline
### a. Train JEPA and base generative model
```
python I_train_jepa.py
python II_train_base.py
```

### b. Use base model to generate candidates, relax and screen
```
python III_base_gen.py
python IV_relax.py
python V_screen.py
```

### c. Fine-tune the base model, and evaluate
```
python VI_ft_base.py
python VII_final_gen.py
python VIII_final_eval.py
```
Remarks
> 1. Add `--dataset alex_mp_20` behind commands `II ~ VIII` to use Alex-MP-20 dataset.
> 2. Change hyper-parameters, e.g., ```python II_train_base.py --conf_new training.batch_size=256 training.lr=0.0001```.
> 3. Use partial of GPUs, e.g., ```CUDA_VISIBLE_DEVICES=0,1 II_train_base.py```. Default: use all GPUs.
> 4. The evaluation code is mainly adapted from [MatterGen](https://github.com/microsoft/mattergen) and [FlowMM](https://github.com/facebookresearch/flowmm).
