# Mini CLIP — Contrastive Language-Image Pretraining on Flickr30k

A from-scratch implementation of CLIP (Contrastive Language-Image Pretraining) trained on the Flickr30k dataset. Supports text-to-image retrieval, image-to-text retrieval, and zero-shot image classification — no task-specific labels required.

---

## Demo

**Text-to-Image Retrieval** — query: *"a dog running on the beach"*

> Type any natural language query → retrieve the most semantically similar images from the dataset.

**Zero-Shot Classification** — no classification head, no fine-tuning on labels.

> Embed class names as text prompts → find the closest match to the image embedding.

---

## How It Works

CLIP learns a **shared embedding space** for images and text using contrastive learning.

- An **image encoder** (ResNet-50) and a **text encoder** (DistilBERT) independently map their inputs to a 256-d embedding space
- A **projection head** on each encoder aligns the two modalities
- **InfoNCE loss** pulls matched (image, caption) pairs together and pushes unmatched pairs apart
- After training, similarity between any image and any text can be computed as a dot product

```
Image ──► ResNet-50 ──► Projection ──►  ┐
                                         ├──► Cosine Similarity ──► InfoNCE Loss
Text  ──► DistilBERT ──► Projection ──► ┘
```

---

## Project Structure

```
mini-clip/
├── model.py       # ImageEncoder, TextEncoder, InfoNCE loss, MiniCLIP
├── dataset.py     # Flickr30k dataset loader, transforms, DataLoader factory
├── train.py       # Two-phase training loop, Recall@K eval, checkpointing
├── eval.py        # Retrieval demos, zero-shot classification, plots
└── README.md
```

---

## Setup

**Requirements**
```bash
pip install torch torchvision transformers pandas matplotlib pillow tqdm
```

**Dataset**

Download [Flickr30k from Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) and place it as:
```
/kaggle/input/flickr30k/
    flickr30k_images/
        *.jpg
    results.csv
```

---

## Training

Training is split into two phases:

**Phase 1 — Frozen Backbones** (5 epochs, lr=1e-3)
Only the projection heads are trained. Fast convergence, stable loss.

**Phase 2 — Fine-tuned** (10 epochs, lr=1e-5)
Both encoders are unfrozen and fine-tuned end-to-end with a lower learning rate.

```bash
python train.py
```

Checkpoints are saved to `/kaggle/working/checkpoints/` whenever validation Recall@1 improves.

**Training config** (edit in `train.py`):

| Parameter | Phase 1 | Phase 2 |
|---|---|---|
| Epochs | 5 | 10 |
| Learning Rate | 1e-3 | 1e-5 |
| Batch Size | 64 | 64 |
| Backbone | Frozen | Unfrozen |

---

## Evaluation

Metric: **Recall@K** (image-to-text retrieval on the test set)

For each image, all captions are ranked by cosine similarity. Recall@K measures how often the correct caption appears in the top-K results.

| Metric | Expected (after Phase 2) |
|---|---|
| R@1 | ~40–50% |
| R@5 | ~70–80% |
| R@10 | ~80–85% |

*Results on Flickr30k test set (1000 images, 1 caption per image).*

---

## Inference

```python
from model import MiniCLIP
from transformers import DistilBertTokenizer
import torch

model = MiniCLIP(embed_dim=256, freeze_backbones=False)
ckpt  = torch.load("checkpoints/phase_2_best.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
```

**Text-to-Image Retrieval**
```python
from eval import retrieve_images, build_image_index

image_embs, image_paths = build_image_index(model, test_loader)
results = retrieve_images(
    query="two children playing in the snow",
    model=model, tokenizer=tokenizer,
    image_embs=image_embs, image_paths=image_paths,
    root_dir="flickr30k_images/", top_k=5
)
```

**Zero-Shot Classification**
```python
from eval import zero_shot_classify

ranked = zero_shot_classify(
    image_path="flickr30k_images/123456.jpg",
    class_names=["dog", "cat", "person", "car", "beach"],
    model=model, tokenizer=tokenizer
)
# [('dog', 0.82), ('beach', 0.11), ...]
```

---

## Architecture Details

| Component | Details |
|---|---|
| Image Encoder | ResNet-50 (ImageNet pretrained), 2048-d → 256-d projection |
| Text Encoder | DistilBERT-base, [CLS] token → 256-d projection |
| Embedding Dim | 256 |
| Loss | Symmetric InfoNCE (contrastive cross-entropy) |
| Temperature τ | Learnable, initialized at 0.07, log-space parameterized |
| Optimizer | AdamW, weight_decay=1e-4 |
| LR Scheduler | Cosine Annealing |
| Max Seq Length | 64 tokens |

---

## References

- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — Radford et al., OpenAI (2021)
- [Flickr30k Entities](https://arxiv.org/abs/1505.04870) — Plummer et al. (2015)
- [DistilBERT](https://arxiv.org/abs/1910.01108) — Sanh et al. (2019)

---

## Author

**Deetya Mehta** — B.Tech + M.Tech CS, IIITDM Kancheepuram

[![GitHub](https://img.shields.io/badge/GitHub-deerobo1-181717?logo=github)](https://github.com/deerobo1)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Deetya%20Mehta-0A66C2?logo=linkedin)](https://linkedin.com/in/deetya-mehta-046582282)
