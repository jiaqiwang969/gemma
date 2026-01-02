# Convert the merged model to GGUF

python infra/llama.cpp/convert_hf_to_gguf.py artifacts/merged_model \
 --outfile artifacts/gguf/gemma-3n-finetuned-fp16.gguf \
 --outtype f16

# Quantize to Q4_K_M (recommended - good balance)

./infra/llama.cpp/build/bin/llama-quantize \
 artifacts/gguf/gemma-3n-finetuned-fp16.gguf \
 artifacts/gguf/gemma-3n-finetuned-Q4_K_M.gguf \
 Q4_K_M

# Then test it

./infra/llama.cpp/build/bin/llama-cli \
 -m artifacts/gguf/gemma-3n-finetuned-Q4_K_M.gguf \
 -p "What is 2 plus 3?" \
 -n 50
