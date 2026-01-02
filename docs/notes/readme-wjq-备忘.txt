  MTMD_BACKEND_DEVICE=CPU ./infra/llama.cpp/build/bin/llama-mtmd-cli \
    --log-verbosity 2 \
    -m artifacts/gguf/gemma-3n-finetuned-Q4_K_M.gguf \
    --mmproj artifacts/gguf/gemma-3n-audio-mmproj-f16.gguf \
    --audio assets/data/audio/mlk_speech.wav \
    -p "Please transcribe this audio." \
    -n 64 --temp 0 --no-warmup \
    --device none --n-gpu-layers 0
