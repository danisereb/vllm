# Run Nemotron Nano v3 with MXFP8 linear

Download model from HF:
https://huggingface.co/nvidia/Super-phase1-mtp-bf16

Replace the `config.json` of the model with [config_super_mxfp8.json](config_super_mxfp8.json).

Run the benchmark (change `MODEL_PATH` as needed):

## Bench serve

Run server:

```bash
export MODEL_PATH=/my_home/hf_models/nvidia/Super-phase1-mtp-bf16

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_FLASHINFER_MOE_FP8=1

vllm serve $MODEL_PATH --trust-remote-code --served-model-name my_model --load-format dummy \
--no-enable-prefix-caching --async-scheduling --kv-cache-dtype auto \
--max-model-len 32768 --tensor-parallel-size 4

```

Using `VLLM_ALLOW_LONG_MAX_MODEL_LEN` because `--max-model-len` is larger than the model's `max_position_embeddings` (from `config.json`).

Using `--load-format dummy` for faster init time.

Run client (use the same `MODEL_PATH`):

```bash
vllm bench serve \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name my_model \
  --model $MODEL_PATH \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len 1000 \
  --random-output-len 16000 \
  --num-warmups 30 \
  --ignore-eos \
  --max-concurrency 16 \
  --num-prompts 80
```

## Bench throughput

```bash
export MODEL_PATH=/my_home/hf_models/nvidia/Super-phase1-mtp-bf16

export VLLM_USE_FLASHINFER_MOE_FP8=1

vllm bench throughput --model $MODEL_PATH \
--tensor-parallel-size 2 \
--trust-remote-code \
--async-scheduling \
--max-model-len 8192 \
--backend vllm \
--dataset-name random \
--num-prompts 512 \
--input-len 1000 \
--output-len 7000 \
--load-format dummy

```

In the `config_super_mxfp8.json` file:<br>
If parameter `use_flashinfer` is `true`, flashinfer MXFP8 GEMM (`bmm_mxfp8`) will be used.

Tested with flashinfer code that's based on flashinfer `v0.5.3`.
