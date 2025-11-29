pip install -e .
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -U flash-attn --no-build-isolation
git clone https://github.com/TimDettmers/bitsandbytes.git
cd bitsandbytes; CUDA_VERSION=124 pip install -e .