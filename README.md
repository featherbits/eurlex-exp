# experimental legal assistant

## installation

### on macos

`python3 -m pip install -r requirements.txt`

### windows with nvidia

> (Windows needs the CUDA wheels explicitly.)

* `pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 -f https://download.pytorch.org/whl/cu121`
* `pip install -r requirements-windows.txt`
* verify cuda is active: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`; expect something like: `True NVIDIA GeForce RTX 3080`
* edit config based on gpu, for 3080: `"batch_size": 8, "epochs": 3`

### run

* `python -m src.prepare_multi_eurlex`
* `python -m src.train_classifier en`
* `python -m src.infer_classifier`
