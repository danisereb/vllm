# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple example demonstrating streaming offline inference with AsyncLLM (V1 engine).

This script shows the core functionality of vLLM's AsyncLLM engine for streaming
token-by-token output in offline inference scenarios. It demonstrates DELTA mode
streaming where you receive new tokens as they are generated.

This script can be used to test the reload weights functionality of vLLM with MXFP8 quantization.

Usage:
    python test_vllm_mxfp8_with_reload_weights.py --help
"""

import asyncio
import json
import datasets
import os
import traceback
from typing import Any
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size.")
    parser.add_argument("--enforce-eager", action="store_true", default=False, help="Enforce eager mode in vllm.")
    parser.add_argument("--bf16", action="store_true", default=False, help="Use BF16 model instead of quantization.")
    parser.add_argument("--use-flashinfer", action="store_true", default=False, help="Use flashinfer for MXFP8 quantization.")
    parser.add_argument("--reload-weights", action="store_true", default=False, help="Reload weights after refit.")
    parser.add_argument("--weight-scheme", type=str, default="dynamic", choices=["dynamic", "static"], help="Weight scheme for quantization.")
    parser.add_argument("--load-format", type=str, default="auto", choices=["dummy", "auto"], help="Load format for the model.")
    parser.add_argument("--fused-moe-backend", type=str, default="flashinfer", choices=["flashinfer", "default"], help="Fused MOE backend for the model.")
    return parser.parse_args()


# logprobs=[{1044: Logprob(logprob=-0.36647382378578186, rank=1, decoded_token=',')}]
def extract_logprob(logprobs):
    for lp in logprobs:
        if lp is not None:
            return list(lp.values())[0].logprob

async def stream_response(engine: AsyncLLM, prompt_token_ids: list[int], request_id: str) -> dict[str, Any]:
    """
    Stream response from AsyncLLM and display tokens as they arrive.

    This function demonstrates the core streaming pattern:
    1. Create SamplingParams with DELTA output kind
    2. Call engine.generate() and iterate over the async generator
    3. Print new tokens as they arrive
    4. Handle the finished flag to know when generation is complete
    """

    # Configure sampling parameters for streaming
    sampling_params = SamplingParams(
        max_tokens=8192,
        temperature=0.0,
        # temperature=1.0,
        # top_p=1.0,
        seed=42,  # For reproducible results
        output_kind=RequestOutputKind.DELTA,  # Get only new tokens each iteration
        logprobs=0,
    )

    token_ids = []
    logprobs = []

    try:
        # Stream tokens from AsyncLLM
        async for output in engine.generate(
            request_id=request_id, prompt={"prompt_token_ids": prompt_token_ids}, sampling_params=sampling_params
        ):
            # Process each completion in the output
            for completion in output.outputs:
                # In DELTA mode, we get only new tokens generated since last iteration
                token_ids.extend(completion.token_ids)
                logprobs.append(extract_logprob(completion.logprobs))

            # Check if generation is finished
            if output.finished:
                return {"token_ids": token_ids, "logprobs": logprobs}

    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")
        raise


class MyVllmWorkerExtension:
    def emulate_refit_in_nemorl(self):
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )
        from vllm.model_executor.model_loader import get_model_loader

        try:
            self.model_runner.reload_weights()
            process_weights_after_loading(
                self.model_runner.model,
                self.model_runner.model_config,
                next(self.model_runner.model.parameters()).device,
            )
        except Exception as e:
            print(f"Error in MyVllmWorkerExtension.emulate_refit_in_nemorl: {e}; {traceback.format_exc()}")
            return False
        return True

async def main():
    args = parse_args()

    # setup envvars
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

    if args.fused_moe_backend == "flashinfer":
        os.environ["VLLM_USE_FLASHINFER_MOE_FP8"] = "1"
        os.environ["VLLM_FLASHINFER_MOE_BACKEND"] = "latency"
    else:
        os.environ["VLLM_USE_FLASHINFER_MOE_FP8"] = "0"

    # Create AsyncLLM engine with simple configuration
    print("🔧 Initializing AsyncLLM...")
    mxfp8_hf_overrides = {
        "quantization_config": {
            "activation_scheme": "dynamic",
            "weight_scheme": args.weight_scheme,
            "quant_method": "fp8",
            "is_mx": True,
            "use_flashinfer": args.use_flashinfer,
            "ignore": [
                "backbone.layers.46.mixer.conv1d",
                "backbone.layers.57.mixer.conv1d",
                "backbone.layers.24.mixer.conv1d",
                "backbone.layers.62.mixer.conv1d",
                "backbone.layers.18.mixer.conv1d",
                "backbone.layers.68.mixer.conv1d",
                "backbone.layers.15.mixer.conv1d",
                "backbone.layers.71.mixer.conv1d",
                "backbone.layers.38.mixer.conv1d",
                "backbone.layers.22.mixer.conv1d",
                "backbone.layers.13.mixer.conv1d",
                "backbone.layers.33.mixer.conv1d",
                "backbone.layers.82.mixer.conv1d",
                "backbone.layers.6.mixer.conv1d",
                "backbone.layers.9.mixer.conv1d",
                "backbone.layers.75.mixer.conv1d",
                "backbone.layers.84.mixer.conv1d",
                "backbone.layers.51.mixer.conv1d",
                "backbone.layers.66.mixer.conv1d",
                "backbone.layers.40.mixer.conv1d",
                "backbone.layers.55.mixer.conv1d",
                "backbone.layers.20.mixer.conv1d",
                "backbone.layers.35.mixer.conv1d",
                "backbone.layers.53.mixer.conv1d",
                "backbone.layers.11.mixer.conv1d",
                "backbone.layers.31.mixer.conv1d",
                "backbone.layers.0.mixer.conv1d",
                "backbone.layers.73.mixer.conv1d",
                "backbone.layers.77.mixer.conv1d",
                "backbone.layers.2.mixer.conv1d",
                "backbone.layers.44.mixer.conv1d",
                "backbone.layers.60.mixer.conv1d",
                "backbone.layers.80.mixer.conv1d",
                "backbone.layers.29.mixer.conv1d",
                "backbone.layers.42.mixer.conv1d",
                "backbone.layers.27.mixer.conv1d",
                "backbone.layers.49.mixer.conv1d",
                "backbone.layers.64.mixer.conv1d",
                "backbone.layers.4.mixer.conv1d",
                "backbone.layers.86.mixer.conv1d"
            ]
        }
    }
    hf_overrides = {} if args.bf16 else mxfp8_hf_overrides

    # point worker_extension to this module and MyVllmWorkerExtension class
    this_module = __name__
    worker_extension = f"{this_module}.MyVllmWorkerExtension" if args.reload_weights else ""

    engine_args = AsyncEngineArgs(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        max_model_len=100,
        mamba_ssm_cache_dtype="float32",
        max_logprobs=1,
        logprobs_mode="processed_logprobs",
        enforce_eager=args.enforce_eager,
        load_format=args.load_format,
        hf_overrides=hf_overrides,
        worker_extension_cls=worker_extension,
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    if args.reload_weights:
        print("🔧 Emulating refit in nemo-rl...")
        coroutine = engine.collective_rpc("emulate_refit_in_nemorl")
        success = await coroutine
        if not success[0]:
            print("❌ Failed to emulate refit in nemo-rl")
            return
        print("✅ Emulated refit in nemo-rl")

    tokenizer = await engine.get_tokenizer()

    # aime2025_i = datasets.load_dataset("opencompass/AIME2025", name='AIME2025-I', split='test')
    # aime2025_ii = datasets.load_dataset("opencompass/AIME2025", name='AIME2025-II', split='test')
    # dset = datasets.concatenate_datasets([aime2025_i, aime2025_ii]).select(range(25, 30))
    # dset = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    dset = list([
        {
            "problem": "The future of AI is"
        }
    ])

    try:
        # Process each prompt
        for i, example in enumerate(dset, 1):

            print(f"Processing example {i}")

            request_id = f"stream-example-{i}"

            # prompt_token_ids = tokenizer.encode(example["question"])
            prompt_token_ids = tokenizer.encode(example["problem"])
            response = await stream_response(engine, prompt_token_ids, request_id)
            
            # write a jsonl file with the response
            with open(f"vllm_mxfp8.jsonl", "a") as f:
                vllm_commit_id = os.popen("git -C 3rdparty/vllm rev-parse HEAD").read().strip()
                config = {
                    "model_path": args.model_path,
                    "tp": args.tp,
                    "enforce_eager": args.enforce_eager,
                    "bf16": args.bf16,
                    "reload_weights": args.reload_weights,
                    "fused_moe_backend": args.fused_moe_backend,
                    "vllm_commit_id": vllm_commit_id,
                }
                output = config
                output.update({"prompt_token_ids": prompt_token_ids, "token_ids": response["token_ids"], "logprobs": response["logprobs"]})
                f.write(json.dumps(output) + "\n")

            # Brief pause between examples
            if i < len(dset):
                await asyncio.sleep(0.5)

        print("\n🎉 All streaming examples completed!")

    finally:
        # Always clean up the engine
        print("🔧 Shutting down engine...")
        engine.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
