# Refining CLIP Visual and Language Representations via Self-Supervised Visual Structure

This repository contains the official implementation of my thesis.

This work addresses a fundamental trade-off in vision models: DINOv2 possesses a highly discriminative visual space but lacks linguistic grounding, while CLIP enables language alignment but suffers from a distorted intra-modal geometric structure.

By enforcing CLIP’s image encoder to follow the intra-modal structure of DINOv2, we bridge the gap between these two paradigms, achieving superior cross-modal retrieval and a geometrically richer visual embedding space.

---

## Key Contributions

* **DINO-Soft Targets:** A training approach that uses DINOv2 features as soft targets to guide CLIP's image encoder, resulting in a **3% improvement** in Text R@1 and **2.8%** in Image R@1.
* **Geometric Enrichment:** Our methodology significantly reduces "CLIP-blind pairs" and enriches the visual similarity distribution by leveraging DINOv2's geometric consistency.
* **Modality Gap Insights:** We demonstrate that a smaller average distance between modalities (Modality Gap) does not always equate to better internal alignment.
* **Ablation Studies:** Investigation into symmetric loss terms and lightweight projection layers to effectively unify visual and multimodal spaces.

---

## Methodology Overview



We integrate DINOv2's structural priors into the CLIP training pipeline using two primary techniques:
1.  **Lightweight Projection:** Mapping CLIP features into an intermediate space before enforcing the DINO-Soft loss.
2.  **Symmetric Influence:** Utilizing loss terms that balance the refinement of both visual and textual representations.

---

## Performance Results (Flickr30k Retrieval)

| Method | Text R@1 ↑ | Image R@1 ↑ | Clip-blind pairs ↓ |
| :--- | :---: | :---: | :---: |
| CLIP (Baseline) | 67.4% | 52.7% | 65.10% |
| **DINO-Soft (Ours)** | **70.0% (+3.0%)** | **54.5% (+2.8%)** | **45.45%** |
---
