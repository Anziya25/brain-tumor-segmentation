# Final repository audit (pre-merge)

**Scope:** Static review of tracked changes, README, CI, dependency manifests, Docker/Compose configuration, training, evaluation, and reproducibility helpers on `fix/segmentation-consistency`. This is **not** a full history secret scan, dependency vulnerability scan, runtime penetration test, or real-MRI validation.

## Confirmed

- The branch is ahead of `main` with no behind commits at audit time; `main` has not been modified by this work.
- CPU CI checks Ruff, synthetic/unit tests, dependency compatibility, Docker image build/import, and Compose configuration. User confirmed the latest run green.
- Default Compose publishes on `127.0.0.1:8000` and mounts model assets read-only.
- `.gitignore` excludes `data/`, `models/`, predictions, NIfTI images, and checkpoint formats. `.dockerignore` excludes the corresponding local assets and environment files.
- Training now seeds Python/NumPy/PyTorch/MONAI and records experiment provenance. Evaluation writes a JSON report with valid metric counts and hashes.
- README distinguishes synthetic checks from real MRI performance, and distinguishes CI from deployment.

## Outstanding release blockers / caveats

| Priority | Finding | Required follow-up |
| --- | --- | --- |
| High | Real end-to-end inference with the original model implementation and trusted checkpoint has **not** been verified. | Run an authorized, non-identifiable four-modality MRI case and inspect the saved NIfTI and ET/TC/WT volumes. |
| High | Original pretrained-model revision, checkpoint provenance, and license are not recorded. | Record exact source, revision, license, and SHA-256 before redistributing or advertising results. |
| High | API has no authentication and handles potentially sensitive medical images. | Keep private/local only; do not expose directly to the public internet. |
| High | `torch.load(..., weights_only=False)` is used for trusted fine-tuned checkpoints. | Never load untrusted checkpoints; consider a restricted loading strategy after checking checkpoint compatibility. |
| Medium | GPU Docker path is not covered by CI; `requirements.txt` still uses `monai[nibabel,itk]` while the CPU manifest removed the ITK extra due to invalid wheel metadata. | Test GPU build on an NVIDIA host, validate package wheels and ONNX Runtime compatibility, and simplify unused extras only after confirming requirements. |
| Medium | Random seeding improves repeatability, but bitwise reproducibility is not guaranteed and complete RNG/optimizer states are not saved. | Capture full training state and test deterministic reruns if exact resume/reproduction is required. |
| Medium | Evaluation metrics use independently thresholded region channels, while the API enforces nested regions. | Align evaluation with final-label-map postprocessing before comparing published segmentation results. |
| Medium | CPU CI Docker smoke test only imports `app.main`; no real checkpoint inference or GPU runtime test. | Add opt-in integration tests in a trusted environment with model assets. |
| Medium | Current documentation does not include verified numeric test Dice/HD95 or actual prediction screenshots. | Populate only from recorded, verified evaluation reports and authorized example data. |
| Low | CI workflow uses `actions/checkout@v4`, `actions/setup-python@v5`, and `ubuntu-latest`. | Review runner migration and action-version deprecation warnings before a long-lived release. |
| Low | README curl sample is Bash-oriented, not native PowerShell. | Add a PowerShell-compatible example if Windows setup is a primary audience. |

## Before merging to main

- [ ] Review the full diff and confirm no secrets, private data, or model binaries are tracked; scan repository **history** separately.
- [ ] Verify pretrained model source, attribution, and checkpoint licensing.
- [ ] Validate real inference with authorized data and trusted weights.
- [ ] Verify the held-out evaluation JSON against the actual test cohort.
- [ ] Decide whether GPU deployment is supported or explicitly experimental.
- [ ] Confirm security assumptions (local-only, no authentication, retention policy).
- [ ] Run a fresh green CI check on the final commit.
- [ ] Open a pull request for human review; merge only with explicit approval.

**Audit disposition:** The repository is suitable as an actively developed **engineering portfolio project**, with clear caveats. It is **not yet ready to claim clinically validated accuracy, verified real-data inference, or production deployment**.
