# SPDX-License-Identifier: Apache-2.0
"""Experimental gfx1100 fixed scalar lowering for six Wan5B decode convolutions.

The artifact is built and verified before model requests. This eager-only
prototype preserves native causal padding, BF16 cast, GEMM and bias semantics.
"""
import ctypes
import copy
import atexit
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import threading
import types
import time

import torch

MODULES = tuple(f"decoder.up_blocks.3.resnets.{block}.conv{conv}" for block in range(3) for conv in (1, 2))
SOURCE_DIR = Path(__file__).with_name("hip")
ACTIVATION_CONFIG = Path(__file__).with_name("wan3d_lowering_activation.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy():
    return not (torch.are_deterministic_algorithms_enabled() or torch.backends.cudnn.deterministic)


def eligible(module, x, weight, bias):
    if torch.compiler.is_compiling() or torch.is_grad_enabled() or not torch.is_inference_mode_enabled() or module.training:
        return "compile_grad_or_training"
    if x.device.type != "cuda" or weight.device != x.device or bias is None or bias.device != x.device:
        return "device_or_bias"
    if torch.cuda.get_device_properties(x.device).gcnArchName.split(":")[0] != "gfx1100":
        return "architecture"
    if x.device.index is not None and torch.cuda.current_device() != x.device.index:
        return "current_device"
    if torch.cuda.is_current_stream_capturing():
        return "graph_capture"
    if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.bfloat16:
        return "autocast"
    if x.dtype != torch.float32 or weight.dtype != torch.bfloat16 or bias.dtype != torch.bfloat16:
        return "dtype"
    if not (x.is_contiguous() and weight.is_contiguous() and bias.is_contiguous()):
        return "layout"
    if x.ndim != 5 or x.shape[0] != 1 or x.shape[1] not in (256, 512) or x.shape[2] not in (3, 6) or tuple(x.shape[3:]) != (130, 130):
        return "input_shape"
    if tuple(weight.shape) != (256, x.shape[1], 3, 3, 3) or tuple(bias.shape) != (256,):
        return "parameter_shape"
    if module.groups != 1 or tuple(module.padding) != (0, 0, 0) or tuple(module.stride) != (1, 1, 1) or tuple(module.dilation) != (1, 1, 1):
        return "convolution_geometry"
    return None


class LoweringBackend:
    def __init__(self, artifact, device):
        destination = os.environ.get("IMP_PROBE_TRACE_DIR")
        if destination:
            Path(destination).mkdir(parents=True, exist_ok=True)
        self.artifact = Path(artifact)
        manifest = json.loads((self.artifact / "manifest.json").read_text())
        for name in ("wan3d_lowering.cpp", "MIOpenIm3d2Col.cpp"):
            if sha256(SOURCE_DIR / name) != manifest["source_sha256"][name]:
                raise RuntimeError("Wan lowering artifact/source mismatch")
        if manifest["torch_git"] != torch.version.git_version or manifest["hip"] != torch.version.hip:
            raise RuntimeError("Wan lowering artifact/runtime mismatch")
        library_path = self.artifact / "lowering-helper.so"
        if sha256(library_path) != manifest["library_sha256"]:
            raise RuntimeError("Wan lowering library hash mismatch")
        self.library = ctypes.CDLL(str(library_path))
        self.library.lowering.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.library.gemm.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
        self.library.create_handle.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int]
        self.library.handle_policy.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p)]
        self.library.destroy_handle.argtypes = [ctypes.c_void_p]
        self.handles = {}
        self.handle_metadata = []
        self.preinitialize(device)
        atexit.register(self.close)

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f"Wan HIP/rocBLAS status {status}")

    @staticmethod
    def key(device):
        return (device.index if device.index is not None else torch.cuda.current_device(), threading.get_ident(), torch.cuda.current_stream(device).cuda_stream, policy())

    def preinitialize(self, device):
        if device.index is not None and torch.cuda.current_device() != device.index:
            raise RuntimeError("Wan handle initialization current device mismatch")
        key = self.key(device)
        if key in self.handles:
            return
        pointer = ctypes.c_void_p()
        self.check(self.library.create_handle(ctypes.byref(pointer), key[2], int(key[3])))
        atomics, mode, stream = ctypes.c_int(), ctypes.c_int(), ctypes.c_void_p()
        try:
            self.check(self.library.handle_policy(pointer, ctypes.byref(atomics), ctypes.byref(mode), ctypes.byref(stream)))
            if atomics.value != int(key[3]) or mode.value != 0 or (stream.value or 0) != key[2]:
                raise RuntimeError("Wan private rocBLAS policy mismatch")
        except BaseException:
            self.library.destroy_handle(pointer)
            raise
        self.handles[key] = pointer
        self.handle_metadata.append({"device": key[0], "thread": key[1], "stream": key[2], "atomics_allowed": key[3], "effective_atomics": atomics.value, "pointer_mode": mode.value})

    def ready(self, x):
        return self.key(x.device) in self.handles

    def run(self, x, weight, bias):
        key = self.key(x.device)
        handle = self.handles[key]
        c, d = x.shape[1:3]
        # Same cast as native autocast; no padding/cache change or persistent pool.
        operand = x.to(torch.bfloat16)
        col = torch.empty((c * 27, (d - 2) * 128 * 128), device=x.device, dtype=torch.bfloat16)
        out = torch.empty((1, 256, d - 2, 128, 128), device=x.device, dtype=torch.bfloat16)
        self.check(self.library.lowering(operand.data_ptr(), col.data_ptr(), c, d, 1, key[2]))
        self.check(self.library.gemm(handle, col.data_ptr(), weight.data_ptr(), out.data_ptr(), c, d))
        out.add_(bias.view(1, -1, 1, 1, 1))
        return out

    def close(self):
        # Explicit shutdown only, outside requests; do not destroy in a finalizer.
        if hasattr(self, "cleanup_result") and self.cleanup_result["ok"]:
            return self.cleanup_result
        before = len(self.handles)
        errors = []
        for key, handle in list(self.handles.items()):
            try:
                with torch.cuda.device(key[0]):
                    torch.cuda.synchronize()
                    self.check(self.library.destroy_handle(handle))
                del self.handles[key]
            except BaseException as error:
                errors.append(repr(error))
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        result = {"executed": True, "rank": rank, "pid": os.getpid(), "handles_before": before,
                  "handles_after": len(self.handles), "errors": errors, "ok": not errors,
                  "artifact": str(self.artifact),
                  "activation_identity": getattr(self, "activation_identity", {"route": "explicit local probe", "artifact": str(self.artifact)})}
        self.cleanup_result = result
        destination = os.environ.get("IMP_PROBE_TRACE_DIR")
        if destination:
            path = Path(destination) / f"wan-lowering-cleanup-rank{rank}-pid{os.getpid()}.json"
            path.write_text(json.dumps(result, indent=2) + "\n")
        if errors:
            raise RuntimeError("Wan private handle cleanup failed: " + "; ".join(errors))
        atexit.unregister(self.close)
        return result


class WanLoweringAdapter:
    def __init__(self, vae, backend):
        self.vae, self.backend = vae, backend
        self.counts = {name: {"hits": {}, "fallbacks": {}} for name in MODULES}
        self.originals = []
        self.request_count = 0
        modules = dict(vae.named_modules())
        if any(name not in modules for name in MODULES):
            raise RuntimeError("Wan lowering six-module scope mismatch")
        try:
            for name in MODULES:
                module = modules[name]
                original = module._conv_forward
                had_local = "_conv_forward" in module.__dict__
                previous = module.__dict__.get("_conv_forward")
                self.originals.append((module, had_local, previous))

                def forward(this, input, weight, bias, original=original, name=name):
                    reason = eligible(this, input, weight, bias)
                    if reason is None and (self.vae.is_distributed_enabled() or self.vae.distributed_executor.parallel_size != 1 or not self.vae.use_tiling):
                        reason = "distributed_vae"
                    if reason is None and not self.backend.ready(input):
                        reason = "uninitialized_thread_stream_policy"
                    if reason is not None:
                        counts = self.counts[name]["fallbacks"]
                        counts[reason] = counts.get(reason, 0) + 1
                        return original(input, weight, bias)
                    shape = "x".join(map(str, input.shape))
                    counts = self.counts[name]["hits"]
                    counts[shape] = counts.get(shape, 0) + 1
                    annotation = (
                        torch.profiler.record_function(f"WAN3D_LOWERING/{name}/{shape}")
                        if os.environ.get("WAN5B_LOWERING_PROFILE") == "1"
                        else nullcontext()
                    )
                    with annotation:
                        return self.backend.run(input, weight, bias)

                module._conv_forward = types.MethodType(forward, module)
        except BaseException:
            self.remove()
            raise

    def preinitialize(self):
        self.backend.preinitialize(next(self.vae.parameters()).device)

    def begin_request(self, req):
        # The first warmup establishes ownership on the actual execution thread.
        # Later unexpected thread/stream changes fall back without handle creation.
        if self.request_count == 0:
            self.preinitialize()
        self.request_baseline = copy.deepcopy(self.counts)
        self.request_ids = [request.request_id for request in req.requests]
        self.dummy_requests = [bool(request.is_dummy_run()) for request in req.requests]
        self.request_started_ns = time.time_ns()
        self.request_count += 1

    def write_report(self):
        destination = os.environ.get("IMP_PROBE_TRACE_DIR")
        if destination:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
            delta = {}
            for name, groups in self.counts.items():
                delta[name] = {}
                for category, counts in groups.items():
                    previous = self.request_baseline[name][category]
                    delta[name][category] = {key: value - previous.get(key, 0) for key, value in counts.items() if value - previous.get(key, 0)}
            path = Path(destination) / f"wan-lowering-stats-rank{rank}-request{self.request_count}.json"
            record = {**self.report(), "request_index": self.request_count, "rank": rank,
                                       "per_request_modules": delta, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                       "request_ids": self.request_ids, "dummy_requests": self.dummy_requests,
                                       "started_time_ns": self.request_started_ns, "ended_time_ns": time.time_ns(),
                                       "activation": "source config or environment", "index_convention": "one-based all pipeline calls, never infer warmup/scored ordinal"}
            path.write_text(json.dumps(record, indent=2) + "\n")
            if rank == 0:
                print("WAN_LOWERING_REQUEST " + json.dumps({"rank": rank, "request_index": self.request_count,
                      "request_ids": self.request_ids, "dummy_requests": self.dummy_requests, "stats_file": path.name,
                      "started_time_ns": self.request_started_ns, "ended_time_ns": record["ended_time_ns"]}), flush=True)

    def report(self):
        return copy.deepcopy({"modules": self.counts, "handles": self.backend.handle_metadata, "artifact": str(self.backend.artifact),
                              "activation_identity": getattr(self.backend, "activation_identity", {"route": "explicit local probe"})})

    def remove(self):
        remaining, errors = [], []
        for entry in self.originals:
            module, had_local, previous = entry
            try:
                if had_local:
                    module._conv_forward = previous
                else:
                    del module._conv_forward
            except BaseException as error:
                remaining.append(entry)
                errors.append(repr(error))
        self.originals = remaining
        if errors:
            raise RuntimeError("Wan adapter restoration failed: " + "; ".join(errors))


def activation_supported(config):
    return bool(getattr(config, "enforce_eager", False)) and not (
        getattr(config, "diffusion_offload_config", None)
        or any(getattr(config, flag, False) for flag in (
            "enable_cpu_offload", "enable_layerwise_offload", "enable_distributed_layerwise_offload"
        ))
    )


def install_if_requested(vae, config):
    if not activation_supported(config):
        return None
    artifact = os.environ.get("WAN5B_LOWERING_ARTIFACT")
    expected_manifest = None
    activation = "environment"
    if not artifact:
        config = json.loads(ACTIVATION_CONFIG.read_text())
        artifact = config.get("artifact_path")
        expected_manifest = config.get("manifest_sha256")
        activation = "source_archived_config"
    if not artifact:
        return None
    if activation == "source_archived_config" and (
        not expected_manifest or not Path(artifact).is_absolute()
        or sha256(Path(artifact) / "manifest.json") != expected_manifest
    ):
        raise RuntimeError("Configured Wan lowering artifact manifest mismatch")
    parameter = next(vae.parameters())
    if parameter.device.type != "cuda" or torch.cuda.get_device_properties(parameter.device).gcnArchName.split(":")[0] != "gfx1100":
        return None
    if vae.is_distributed_enabled() or vae.distributed_executor.parallel_size != 1 or not vae.use_tiling:
        return None
    backend = LoweringBackend(artifact, parameter.device)
    try:
        adapter = WanLoweringAdapter(vae, backend)
    except BaseException as error:
        try:
            backend.close()
        except BaseException as cleanup_error:
            error.add_note("Backend cleanup error: " + repr(cleanup_error))
        raise
    adapter.backend.activation_identity = {"route": activation, "artifact": str(artifact),
                                           "manifest_sha256": sha256(Path(artifact) / "manifest.json"),
                                           "config_sha256": sha256(ACTIVATION_CONFIG)}
    return adapter
