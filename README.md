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

## Notebook Structure

All code is in a single Kaggle notebook, organized as:

| Section | Description |
|---|---|
| 1. Imports & Config | Dependencies, hyperparameters, paths |
| 2. Dataset & Dataloaders | Flickr30k loader, train/val/test split, transforms |
| 3. Model Architecture | ImageEncoder, TextEncoder, projection heads |
| 4. InfoNCE Loss | Symmetric contrastive loss, learnable temperature |
| 5. Training Loop | Per-epoch train, gradient clipping, LR scheduling |
| 6. Phase 1 — Frozen Backbones | Train only projection heads (5 epochs, lr=1e-3) |
| 7. Phase 2 — Fine-tuning | Unfreeze encoders, end-to-end training (10 epochs, lr=1e-5) |
| 8. Evaluation — Recall@K | R@1, R@5, R@10 on test set |
| 9. Demo — Text-to-Image Retrieval | Natural language query → top-5 images |
| 10. Demo — Zero-Shot Classification | Image → ranked class predictions, no labels |

---

## Setup

**Dataset**

Uses [Flickr30k from Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset) — add it directly to your Kaggle notebook as an input dataset.

**Dependencies** — all pre-installed on Kaggle except:
```python
!pip install transformers -q
```

**Hardware** — trained on Kaggle T4 GPU. Total training time ~2 hours.

---

## Training

Training is split into two phases:

**Phase 1 — Frozen Backbones** (5 epochs, lr=1e-3)
Only the projection heads are trained. Fast convergence, stable loss.

**Phase 2 — Fine-tuned** (10 epochs, lr=1e-5)
Both encoders are unfrozen and fine-tuned end-to-end with a lower learning rate.

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

| Metric | Result |
|---|---|
| R@1 | — |
| R@5 | — |
| R@10 | — |

*Fill in after training. Test set: 1000 images, 1 caption per image.*

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

**Deetya Mehta**
