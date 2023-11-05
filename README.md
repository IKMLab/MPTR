# MPTR: Multi-label Prompt-Tuning for Liver Computer Tomography Report Labeling with Limited Training Data

This repository contains the code for the paper [MPTR: Multi-label Prompt-Tuning for Liver Computer Tomography Report Labeling with Limited Training Data](under review).

## Pre-requisites
- Python version: 3.9.5
- CUDA version: 10.1
- OS: Ubuntu 20.04.4 LTS

## Installation

```bash
pip install torch==1.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
mkdir logs
```

## Download Dataset
The test set for evaluating the performance is provided in this repository for data validation use and academic purpose only.

## Download Checkpoints
Trained using 32 labeled reports:
- [MPTR](https://ncku365-my.sharepoint.com/:u:/g/personal/p78081057_ncku_edu_tw/EfoqAI7s9_5Fo_FEHVIOAOkBrQlAMYLmQi66Y47tpj4Aog?e=1EA18A) 
- [BERT](https://ncku365-my.sharepoint.com/:u:/g/personal/p78081057_ncku_edu_tw/ESye26hhiIVHjDte0IrhbQQBwiipevYldshpUtjpne1kJg?e=PlKNly)

```bash
mkdir checkpoints
# Put the downloaded zip file in the checkpoints folder and unzip it.
```

## Folder Structure
```
.
├── checkpoints
│   ├── MPTR
│   │   ├── seed_0
│   │   ├── seed_1
│   │   ├── seed_3
│   │   ├── seed_7
│   │   └── seed_10
│   └── BERT
│       ├── seed_0
│       ├── seed_1
│       ├── seed_3
│       ├── seed_7
│       └── seed_10
├── data
│   ├── class_names.pkl
│   └── test_after_15_cleaned.pkl
├── logs
├── src
```
For each seed, the folder structure is as follows:
```
├── seed_n
    ├── checkpoint-ckpt_number
    │   ├── config.json
    │   ├── optimizer.pt
    │   ├── rng_state.pth
    │   ├── scheduler.pt
    │   ├── special_tokens_map.json
    │   ├── tokenizer_config.json
    │   ├── tokenizer.json
    │   ├── trainer_state.json
    │   ├── training_args.bin
    │   ├── vocab.txt
    │   └── pytorch_model.bin
    └── args.json
```

## Run Inference
- Take MPTR+AutoT as an example:
```bash
python src/test_model.py \
--data_type train_32 \
--test_filename test_after_15_cleaned.pkl \
--test_method MPTR \
--ckpt_path checkpoint-400
```
- Replace `MPTR+AutoT` with `MPTR` or `BERT` for `--test_method` to infer the report labeling task with the other methods.
- This script is also placed in the `scripts` folder.

## Run Evaluation
- Take MPTR+AutoT as an example:
```bash
python src/evaluate.py \
--data_type train_32 \
--test_filename test_after_15_cleaned.pkl \
--test_method MPTR+AutoT
```
- Replace `MPTR+AutoT` with `MPTR` or `BERT` for `--test_method` to evaluate the report labeling task with the other methods.
- This script is also placed in the `scripts` folder.
- The scores will be printed after running this script.
- The scores will be slightly different from the ones in the paper due to the removal of private information in the reports.