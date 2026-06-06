# G2T: Gate Guided Transformer for Medical Image Segmentation

PyTorch implementation of **G2T: Gate Guided Transformer for Medical Image Segmentation**.

G2T is an encoder-decoder segmentation model with a memory-inspired decoder. The core components are:

- **Feature-Aware Decoder (FAD)** with Refine, Expand, and Produce gates.
- **REP gates** that retain relevant decoder memory, add useful current features, and produce final decoded features.
- **LH-RAAM**: Low-to-High Resolution Attention Accumulation Module for consistent encoder-decoder spatial focus.
- **Deep supervision** with CE + Dice loss over decoder prediction maps and their fused output.

This repository provides a clean runnable implementation using a lightweight hierarchical encoder, while keeping the architecture modular so a MaxViT or other hierarchical transformer encoder can be plugged in later.

## Repository Layout

```text
G2T/
  g2t/
    data/              CSV image-mask dataset loader
    models/            G2T, FAD decoder blocks, LH-RAAM, encoder blocks
    training/          Losses and segmentation metrics
    utils/             Reproducibility helpers
  configs/             Example experiment configs
  scripts/             Demo and training commands
  tests/               Forward and loss tests
```

## Installation

```bash
git clone https://github.com/Rahman-Motiur/G2T.git
cd G2T
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Demo Forward Pass

```bash
python scripts/demo_g2t_forward.py
```

## Training

Prepare a CSV file:

```csv
image,mask
data/images/case_001.npy,data/masks/case_001.npy
```

Run:

```bash
python scripts/train_g2t.py --config configs/g2t_acdc.yaml
```


## Datasets

The paper evaluates G2T on six medical imaging datasets, including ACDC, Synapse, Kvasir-SEG, ClinicDB, ColonDB, and lung segmentation data.

## Notes

The paper uses a frozen ImageNet-pretrained MaxViT encoder. This project includes a compact hierarchical convolutional encoder for reproducible demos and easy extension. To reproduce paper-scale results, replace `HierarchicalEncoder` with a pretrained MaxViT-style encoder that returns four feature maps and attention maps.

## Citation

```bibtex
@article{rahman2026g2t,
  title={G2T: Gate Guided Transformer for Medical Image Segmentation},
  author={Rahman, Md Motiur and Shokouhmand, Shiva and Rahman, Saeka and Bhatt, Smriti and Mirbozorgi, S. Abdollah and Faezipour, Miad},
  journal={Submitted},
  year={2026}
}
```
