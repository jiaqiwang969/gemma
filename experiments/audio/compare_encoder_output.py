#!/usr/bin/env python3
"""
Compare PyTorch and llama.cpp encoder outputs for Gemma 3n audio
"""
import numpy as np
import torch
import soundfile as sf

def get_pytorch_encoder_output():
    """Get encoder output from PyTorch model"""
    from transformers import AutoProcessor, AutoModelForImageTextToText

    # Load audio
    audio, sr = sf.read('/tmp/mlk_speech_16k.wav')
    audio = audio.astype(np.float32)
    print(f"Audio: {len(audio)} samples at {sr} Hz")

    # Load model
    model_name = "google/gemma-3n-E2B-it"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for comparison
        device_map="cpu",  # CPU for easier debugging
        trust_remote_code=True
    )

    # Create message with audio
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio, "sample_rate": sr},
                {"type": "text", "text": "Transcribe this audio."}
            ]
        }
    ]

    # Process
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    print(f"\nInput keys: {inputs.keys()}")

    if 'input_features' in inputs:
        mel = inputs['input_features']
        print(f"Mel shape: {mel.shape}")
        print(f"Mel stats: min={mel.min():.4f}, max={mel.max():.4f}, mean={mel.mean():.4f}")

        # Save mel for llama.cpp comparison
        np.save('/tmp/pytorch_mel.npy', mel.numpy())
        print(f"Saved mel to /tmp/pytorch_mel.npy")

    # Hook to capture encoder output
    encoder_outputs = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            encoder_outputs.append(output.detach().cpu().numpy())
        elif hasattr(output, 'last_hidden_state'):
            encoder_outputs.append(output.last_hidden_state.detach().cpu().numpy())
        print(f"Hook captured output: {type(output)}")

    # Find the audio encoder in the model
    print(f"\nModel structure (top level):")
    for name, module in model.named_children():
        print(f"  {name}: {type(module).__name__}")

    # Register hook on audio encoder
    # Gemma 3n structure: model.audio_encoder (Gemma3nAudioEncoder)
    if hasattr(model, 'audio_encoder'):
        print(f"\nAudio encoder: {type(model.audio_encoder).__name__}")
        handle = model.audio_encoder.register_forward_hook(hook_fn)
    else:
        print("\nNo audio_encoder found, searching...")
        for name, module in model.named_modules():
            if 'audio' in name.lower() and 'encoder' in name.lower():
                print(f"  Found: {name}")

    # Run forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        # Just run forward, don't generate
        outputs = model(**inputs, return_dict=True, output_hidden_states=True)

    print(f"\nOutput type: {type(outputs)}")
    if hasattr(outputs, 'keys'):
        print(f"Output keys: {outputs.keys()}")

    # Gemma 3n provides audio_hidden_states directly
    if hasattr(outputs, 'audio_hidden_states') and outputs.audio_hidden_states is not None:
        enc_out = outputs.audio_hidden_states[0].detach().cpu().numpy()
        print(f"\nAudio hidden states shape: {enc_out.shape}")
        print(f"Audio hidden states stats: min={enc_out.min():.4f}, max={enc_out.max():.4f}, mean={enc_out.mean():.4f}")

        # Save encoder output
        np.save('/tmp/pytorch_encoder_output.npy', enc_out)
        print(f"Saved encoder output to /tmp/pytorch_encoder_output.npy")

        # Print first few values
        print(f"\nFirst 10 values (flat): {enc_out.flatten()[:10]}")
        print(f"Last 10 values (flat): {enc_out.flatten()[-10:]}")

        return enc_out

    if encoder_outputs:
        enc_out = encoder_outputs[0]
        print(f"\nEncoder output shape: {enc_out.shape}")
        print(f"Encoder output stats: min={enc_out.min():.4f}, max={enc_out.max():.4f}, mean={enc_out.mean():.4f}")

        # Save encoder output
        np.save('/tmp/pytorch_encoder_output.npy', enc_out)
        print(f"Saved encoder output to /tmp/pytorch_encoder_output.npy")

        # Print first few values
        print(f"\nFirst 10 values (flat): {enc_out.flatten()[:10]}")

        return enc_out

    return None

def compare_with_llamacpp():
    """Compare PyTorch output with llama.cpp output"""
    try:
        pytorch_out = np.load('/tmp/pytorch_encoder_output.npy')
        print(f"\nPyTorch encoder output shape: {pytorch_out.shape}")
        print(f"PyTorch stats: min={pytorch_out.min():.4f}, max={pytorch_out.max():.4f}, mean={pytorch_out.mean():.4f}")
    except FileNotFoundError:
        print("PyTorch encoder output not found. Run get_pytorch_encoder_output() first.")
        return

    try:
        llamacpp_out = np.load('/tmp/llamacpp_encoder_output.npy')
        print(f"\nllama.cpp encoder output shape: {llamacpp_out.shape}")
        print(f"llama.cpp stats: min={llamacpp_out.min():.4f}, max={llamacpp_out.max():.4f}, mean={llamacpp_out.mean():.4f}")

        # Compare
        if pytorch_out.shape == llamacpp_out.shape:
            diff = np.abs(pytorch_out - llamacpp_out)
            print(f"\nDifference: max={diff.max():.6f}, mean={diff.mean():.6f}")

            # Find where differences are largest
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"Max diff at index {max_idx}: pytorch={pytorch_out[max_idx]:.6f}, llamacpp={llamacpp_out[max_idx]:.6f}")
        else:
            print(f"\nShape mismatch! Cannot compare directly.")
            print(f"PyTorch: {pytorch_out.shape}, llama.cpp: {llamacpp_out.shape}")
    except FileNotFoundError:
        print("llama.cpp encoder output not found.")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_with_llamacpp()
    else:
        get_pytorch_encoder_output()
        compare_with_llamacpp()
