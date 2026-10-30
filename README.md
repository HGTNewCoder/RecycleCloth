# ♻️ RecycleCloth *(work in progress)*

Computer-vision groundwork for clothing reuse and recycling: preparing the **DeepFashion2** dataset and testing vision-language models on garment images.

- `scripts/datasetPreprocessing.py` — scans DeepFashion2 annotations and extracts the "shop" (catalog-style) image IDs
- `scripts/downloadDeepFashion2.sh` — dataset download helper
- `usr/hgt/testhuggingfacemodel/vllm.py` — batched multimodal inference against a locally served VLM (vLLM, OpenAI-compatible API)
- `usr/hgt/fiftyoneviewer.py` — browse the dataset in FiftyOne

**Stack:** Python · PyTorch · vLLM · FiftyOne · OpenCV
