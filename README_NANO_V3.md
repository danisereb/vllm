# Run Nemotron Nano v3 with MXFP8 linear

Download model from HF:
https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

Replace the `config.json` of the model with [config_nano_mxfp8.json](config_nano_mxfp8.json).

Run the benchmark (change `MODEL_PATH` as needed):

```bash
export MODEL_PATH=/my_home/hf_models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/

export VLLM_USE_FLASHINFER_MOE_FP8=1

vllm bench throughput --model $MODEL_PATH \
--tensor-parallel-size 1 \
--trust-remote-code \
--disable-log-requests \
--max-model-len 8192 \
--backend vllm \
--dataset-name random \
--num-prompts 512 \
--input-len 1000 \
--output-len 1000 \
--enforce-eager
```

Using `--enforce-eager` because flashinfer did not work with CUDA graphs (need to solve this).

In the `config_nano_mxfp8.json` file:<br>
If parameter `use_flashinfer` is `true`, flashinfer MXFP8 GEMM (`bmm_mxfp8`) will be used.

Tested with flashinfer code that's based on flashinfer `v0.5.3`.

The file [config_nano_mxfp8_more_ignores.json](config_nano_mxfp8_more_ignores.json) has an extended ignored_layers list.

The file [config_nano_mxfp8_ignore_conv1d.json](config_nano_mxfp8_ignore_conv1d.json) has only conv1d in the ignored_layers list.
