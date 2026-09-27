"""Independent CUDA tensor sharing check using the documented PyTorch API.

The producer stays alive until the consumer reports a result. This avoids
depending on LMCache or private ``torch.UntypedStorage`` methods.
"""

import multiprocessing as mp
import queue
import sys

import torch


def consumer(tensor_queue, result_queue):
    try:
        tensor = tensor_queue.get(timeout=20)
        actual = float(tensor[0].item())
        result_queue.put(("OK", actual))
    except Exception as exc:
        result_queue.put(("FAIL", type(exc).__name__, str(exc)))


def main():
    ctx = mp.get_context("spawn")
    tensor_queue = ctx.Queue()
    result_queue = ctx.Queue()
    tensor = torch.full((1024, 1024), 3.14159, device="cuda:0")
    torch.cuda.synchronize()
    process = ctx.Process(target=consumer, args=(tensor_queue, result_queue))
    process.start()
    tensor_queue.put(tensor)
    try:
        result = result_queue.get(timeout=30)
    except queue.Empty:
        result = ("FAIL", "Timeout", "consumer produced no result")
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join()
    print("CUDA_IPC_PUBLIC_API_RESULT", result)
    print("CONSUMER_EXIT_CODE", process.exitcode)
    return 0 if result[0] == "OK" and abs(result[1] - 3.14159) < 1e-5 else 1


if __name__ == "__main__":
    sys.exit(main())
