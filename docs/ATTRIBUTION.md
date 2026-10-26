# Attribution, dataset citations, and licensing

This repository contains original integration, fine-tuning, evaluation, API, and deployment work built **on an externally pretrained UNETR model**. The pretrained architecture and weights must not be presented as trained from scratch by this repository's author.

## Upstream pretrained UNETR

- **Model author / publisher:** MohanadKombar (upstream GitHub project: MohanadMahran/Brain_Tumor_Benchmark)
- **Model:** [MohanadKombar/brain-tumor-unetr-brats](https://huggingface.co/MohanadKombar/brain-tumor-unetr-brats)
- **Upstream source:** [Brain_Tumor_Benchmark](https://github.com/MohanadMahran/Brain_Tumor_Benchmark)
- **Model card license:** MIT (as stated on the upstream Hugging Face model card; consult the original repository for the full applicable license and copyright notice before redistributing upstream code or weights)
- **Reported upstream training:** composite of BraTS 2021 and BraTS 2024 Adult Glioma datasets, not this project's BraTS 2020 fine-tuning run.

This project's fresh fine-tuning initializes from the upstream `pytorch_model.bin` and trains on 258 BraTS 2020 cases, selects checkpoints on 55 validation cases, and evaluates on 55 reserved cases. The 55 cases were not used in *this fine-tuning run*; potential overlap with upstream training has not been independently excluded. Upstream reported validation Dice (0.8362) is not directly comparable to this project's validation/test results because evaluation cohorts and protocols differ.

Upstream model's model card cites *Fair Benchmarking of 3D CNN vs Vision Transformer Architectures for Volumetric Brain Tumor Segmentation*. Refer to the model card for its preferred citation and full bibliographic details.

## BraTS 2020 dataset

The dataset is administered by the Brain Tumor Segmentation (BraTS) challenge and its contributors. **The MRI volumes and expert annotations are not authored by this repository.** Do not upload scans, ground-truth segmentations, or data-derived patient identifiers to the public repository without authorization.

Official source and data-use/citation instructions: https://www.med.upenn.edu/cbica/brats2020/data.html

The BraTS 2020 organizers specify these three required citations:

1. Menze, B. H., et al. (2015). **The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)**. *IEEE Transactions on Medical Imaging*, 34(10), 1993–2024. https://doi.org/10.1109/TMI.2014.2377694
2. Bakas, S., et al. (2017). **Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features**. *Scientific Data*, 4, 170117. https://doi.org/10.1038/sdata.2017.117
3. Bakas, S., et al. (2018). **Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge**. arXiv:1811.02629. https://arxiv.org/abs/1811.02629

The organizers additionally recommend citing the TCIA TCGA-GBM and TCGA-LGG data records when publication rules allow data citations; see the official BraTS 2020 page for those references. Dataset access, use, and redistribution are subject to the applicable BraTS terms; this repository does not grant rights to redistribute BraTS data.

## This repository's code and third-party components

- Original application code, experiment scripts, tests, and deployment integration are distinct from the upstream pretrained model and dataset.
- Python dependencies (PyTorch, MONAI, FastAPI, and others) retain their own licenses.
- **Original repository code is provided under the [MIT License](../LICENSE), to the extent the owner has the rights to license it.** Third-party code, pretrained weights, and BraTS data retain their own terms.
- The upstream MIT license does **not** automatically relicense the BraTS dataset or all files in this repository. Retain upstream copyright/license notices if upstream source code is copied or redistributed.

## Scope of results

The fresh model's held-out fine-tuning test mean Dice is **0.8294**, compared with **0.8153** for the pretrained baseline on the same 55-patient cohort. This is a patient-disjoint split relative to fresh fine-tuning, **not** independently verified external generalization or clinical validation. The legacy 60-case exploratory evaluation is separate and must not be advertised as an independent held-out test.

**Research/portfolio use only. Not validated for clinical diagnosis, treatment, or medical decision-making.**
