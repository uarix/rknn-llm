import argparse
import base64
import ctypes
import json
import os
import resource
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

rkllm_lib = ctypes.CDLL("lib/librkllmrt.so")

RKLLM_Handle_t = ctypes.c_void_p

LLMCallState = ctypes.c_int
LLMCallState.RKLLM_RUN_NORMAL = 0
LLMCallState.RKLLM_RUN_WAITING = 1
LLMCallState.RKLLM_RUN_FINISH = 2
LLMCallState.RKLLM_RUN_ERROR = 3

RKLLMInputType = ctypes.c_int
RKLLMInputType.RKLLM_INPUT_PROMPT = 0
RKLLMInputType.RKLLM_INPUT_TOKEN = 1
RKLLMInputType.RKLLM_INPUT_EMBED = 2
RKLLMInputType.RKLLM_INPUT_MULTIMODAL = 3

RKLLMInferMode = ctypes.c_int
RKLLMInferMode.RKLLM_INFER_GENERATE = 0


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMMultiModalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput),
    ]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", RKLLMInputType),
        ("input_data", RKLLMInputUnion),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", RKLLMInferMode),
        ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("keep_history", ctypes.c_int),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat),
    ]


lock = threading.Lock()
chunk_buffer: List[str] = []
run_state = -1


def callback_impl(result, userdata, state):
    global chunk_buffer, run_state
    run_state = state
    if state == LLMCallState.RKLLM_RUN_NORMAL:
        if result and result.contents.text:
            chunk_buffer.append(result.contents.text.decode("utf-8", errors="ignore"))
    return 0


callback_type = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int
)
callback = callback_type(callback_impl)


class RKLLMServerModel:
    def __init__(
        self,
        model_path: str,
        platform: str,
        max_context_len: int,
        max_new_tokens: int,
        img_start: str,
        img_end: str,
        img_content: str,
    ) -> None:
        param = RKLLMParam()
        param.model_path = model_path.encode("utf-8")
        param.max_context_len = max_context_len
        param.max_new_tokens = max_new_tokens
        param.skip_special_token = True
        param.n_keep = -1
        param.top_k = 1
        param.top_p = 0.9
        param.temperature = 0.8
        param.repeat_penalty = 1.1
        param.frequency_penalty = 0.0
        param.presence_penalty = 0.0
        param.mirostat = 0
        param.mirostat_tau = 5.0
        param.mirostat_eta = 0.1
        param.is_async = False

        param.img_start = img_start.encode("utf-8")
        param.img_end = img_end.encode("utf-8")
        param.img_content = img_content.encode("utf-8")

        param.extend_param.base_domain_id = 0
        param.extend_param.embed_flash = 1
        param.extend_param.n_batch = 1
        param.extend_param.use_cross_attn = 0
        param.extend_param.enabled_cpus_num = 4
        if platform.lower() in ["rk3576", "rk3588"]:
            param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
        else:
            param.extend_param.enabled_cpus_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)

        self.handle = RKLLM_Handle_t()

        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [
            ctypes.POINTER(RKLLM_Handle_t),
            ctypes.POINTER(RKLLMParam),
            callback_type,
        ]
        self.rkllm_init.restype = ctypes.c_int

        ret = self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(param), callback)
        if ret != 0:
            raise RuntimeError("rkllm init failed")

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [
            RKLLM_Handle_t,
            ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam),
            ctypes.c_void_p,
        ]
        self.rkllm_run.restype = ctypes.c_int

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

        self.infer_param = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.infer_param), 0, ctypes.sizeof(RKLLMInferParam))
        self.infer_param.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        self.infer_param.keep_history = 0

    def run_prompt(self, prompt: str, role: str = "user", enable_thinking: bool = False) -> int:
        rk_input = RKLLMInput()
        rk_input.role = role.encode("utf-8")
        rk_input.enable_thinking = ctypes.c_bool(enable_thinking)
        rk_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
        rk_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode("utf-8"))
        return self.rkllm_run(self.handle, ctypes.byref(rk_input), ctypes.byref(self.infer_param), None)

    def run_multimodal(
        self,
        prompt: str,
        image_embed: np.ndarray,
        n_image_tokens: int,
        image_width: int,
        image_height: int,
        role: str = "user",
        enable_thinking: bool = False,
    ) -> int:
        if image_embed.dtype != np.float32:
            image_embed = image_embed.astype(np.float32)
        if not image_embed.flags["C_CONTIGUOUS"]:
            image_embed = np.ascontiguousarray(image_embed)

        rk_input = RKLLMInput()
        rk_input.role = role.encode("utf-8")
        rk_input.enable_thinking = ctypes.c_bool(enable_thinking)
        rk_input.input_type = RKLLMInputType.RKLLM_INPUT_MULTIMODAL
        rk_input.input_data.multimodal_input.prompt = ctypes.c_char_p(prompt.encode("utf-8"))
        rk_input.input_data.multimodal_input.image_embed = image_embed.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        rk_input.input_data.multimodal_input.n_image_tokens = n_image_tokens
        rk_input.input_data.multimodal_input.n_image = 1
        rk_input.input_data.multimodal_input.image_width = image_width
        rk_input.input_data.multimodal_input.image_height = image_height

        return self.rkllm_run(self.handle, ctypes.byref(rk_input), ctypes.byref(self.infer_param), None)

    def release(self) -> None:
        self.rkllm_destroy(self.handle)


def parse_data_url(data_url: str) -> Tuple[bytes, str]:
    header, b64_data = data_url.split(",", 1)
    ext = ".jpg"
    if ";base64" in header:
        if "image/png" in header:
            ext = ".png"
        elif "image/webp" in header:
            ext = ".webp"
    return base64.b64decode(b64_data), ext


def fetch_image_bytes(image_ref: str) -> Tuple[bytes, str]:
    if image_ref.startswith("data:image"):
        return parse_data_url(image_ref)

    rsp = requests.get(image_ref, timeout=20)
    rsp.raise_for_status()
    content_type = rsp.headers.get("Content-Type", "").lower()
    ext = ".jpg"
    if "png" in content_type:
        ext = ".png"
    elif "webp" in content_type:
        ext = ".webp"
    return rsp.content, ext


def extract_user_content(messages: List[Dict]) -> Tuple[str, Optional[str]]:
    latest_text = ""
    latest_image_url = None

    for msg in messages:
        if msg.get("role") != "user":
            continue

        content = msg.get("content")
        text_parts: List[str] = []
        image_ref: Optional[str] = None

        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                item_type = item.get("type")
                if item_type in ["text", "input_text"]:
                    text_parts.append(item.get("text", ""))
                elif item_type in ["image_url", "input_image"]:
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        image_ref = image_url.get("url")
                    elif isinstance(image_url, str):
                        image_ref = image_url
        latest_text = "\n".join([p for p in text_parts if p]).strip()
        latest_image_url = image_ref

    return latest_text, latest_image_url


def run_img_encoder(
    imgenc_path: str,
    vision_model_path: str,
    core_num: int,
    image_ref: str,
) -> Tuple[np.ndarray, int, int, int]:
    with tempfile.TemporaryDirectory(prefix="rkvl_") as temp_dir:
        image_bytes, image_ext = fetch_image_bytes(image_ref)
        image_path = os.path.join(temp_dir, f"image{image_ext}")
        embed_path = os.path.join(temp_dir, "img_vec.bin")
        meta_path = os.path.join(temp_dir, "img_meta.json")

        with open(image_path, "wb") as f:
            f.write(image_bytes)

        cmd = [
            imgenc_path,
            vision_model_path,
            image_path,
            str(core_num),
            embed_path,
            meta_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"imgenc failed: {proc.stderr or proc.stdout}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        embed = np.fromfile(embed_path, dtype=np.float32)
        if embed.size == 0:
            raise RuntimeError("empty image embedding output")

        return (
            embed,
            int(meta["n_image_tokens"]),
            int(meta["image_width"]),
            int(meta["image_height"]),
        )


def infer_text(
    model: RKLLMServerModel,
    prompt: str,
    enable_thinking: bool,
) -> Tuple[int, str]:
    ret = model.run_prompt(prompt=prompt, role="user", enable_thinking=enable_thinking)
    if ret != 0:
        raise RuntimeError(f"rkllm_run(prompt) failed: {ret}")
    return ret, ""


def infer_multimodal(
    model: RKLLMServerModel,
    prompt: str,
    enable_thinking: bool,
    image_embed: np.ndarray,
    n_image_tokens: int,
    image_width: int,
    image_height: int,
) -> Tuple[int, str]:
    ret = model.run_multimodal(
        prompt=prompt,
        image_embed=image_embed,
        n_image_tokens=n_image_tokens,
        image_width=image_width,
        image_height=image_height,
        role="user",
        enable_thinking=enable_thinking,
    )
    if ret != 0:
        raise RuntimeError(f"rkllm_run(multimodal) failed: {ret}")
    return ret, ""


def collect_output(model_thread: threading.Thread) -> str:
    out_parts: List[str] = []
    while model_thread.is_alive() or chunk_buffer:
        while chunk_buffer:
            out_parts.append(chunk_buffer.pop(0))
        model_thread.join(timeout=0.005)
    return "".join(out_parts)


def stream_output(request_id: str, model_name: str, model_thread: threading.Thread):
    while model_thread.is_alive() or chunk_buffer:
        while chunk_buffer:
            part = chunk_buffer.pop(0)
            payload = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": part},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        model_thread.join(timeout=0.005)

    done_payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def stream_output_with_cleanup(
    request_id: str,
    model_name: str,
    model_thread: threading.Thread,
    run_error: Dict[str, Optional[str]],
):
    try:
        yield from stream_output(request_id, model_name, model_thread)
        if run_error.get("msg"):
            err_payload = {
                "id": request_id,
                "object": "error",
                "message": run_error["msg"],
            }
            yield f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
    finally:
        lock.release()


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"}), 200


@app.route("/v1/models", methods=["GET"])
def models_endpoint():
    return jsonify(
        {
            "object": "list",
            "data": [{"id": SERVER_ARGS.model_name, "object": "model", "owned_by": "rockchip"}],
        }
    )


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    global chunk_buffer, run_state

    if not request.is_json:
        return jsonify({"error": {"message": "JSON body required"}}), 400

    body = request.get_json(silent=True) or {}
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": {"message": "messages is required"}}), 400

    stream = bool(body.get("stream", False))
    enable_thinking = bool(body.get("enable_thinking", False))

    prompt, image_ref = extract_user_content(messages)
    if not prompt:
        prompt = "Describe the image." if image_ref else ""
    if not prompt:
        return jsonify({"error": {"message": "empty user content"}}), 400

    # Keep multimodal token in prompt if image is present.
    if image_ref and "<image>" not in prompt:
        prompt = "<image>\n" + prompt

    if not lock.acquire(blocking=False):
        return jsonify({"error": {"message": "server is busy"}}), 503

    release_in_finally = True
    try:
        chunk_buffer = []
        run_state = -1

        image_embed = None
        n_image_tokens = 0
        image_width = 0
        image_height = 0

        run_error: Dict[str, Optional[str]] = {"msg": None}

        def run_target_multimodal() -> None:
            try:
                infer_multimodal(
                    RK_MODEL,
                    prompt,
                    enable_thinking,
                    image_embed,
                    n_image_tokens,
                    image_width,
                    image_height,
                )
            except Exception as exc:  # pylint: disable=broad-except
                run_error["msg"] = str(exc)

        def run_target_text() -> None:
            try:
                infer_text(RK_MODEL, prompt, enable_thinking)
            except Exception as exc:  # pylint: disable=broad-except
                run_error["msg"] = str(exc)

        if image_ref:
            image_embed, n_image_tokens, image_width, image_height = run_img_encoder(
                imgenc_path=SERVER_ARGS.imgenc_path,
                vision_model_path=SERVER_ARGS.vision_model_path,
                core_num=SERVER_ARGS.rknn_core_num,
                image_ref=image_ref,
            )
            model_thread = threading.Thread(target=run_target_multimodal)
        else:
            model_thread = threading.Thread(target=run_target_text)

        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        model_thread.start()

        if stream:
            release_in_finally = False
            return Response(
                stream_output_with_cleanup(
                    request_id,
                    SERVER_ARGS.model_name,
                    model_thread,
                    run_error,
                ),
                content_type="text/event-stream",
            )

        output_text = collect_output(model_thread)
        if run_error.get("msg"):
            raise RuntimeError(run_error["msg"])
        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": SERVER_ARGS.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
        }
        return jsonify(response), 200

    except Exception as exc:
        return jsonify({"error": {"message": str(exc)}}), 500
    finally:
        if release_in_finally:
            lock.release()


def maybe_fix_frequency(platform: str) -> None:
    script_name = f"fix_freq_{platform.lower()}.sh"
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    if os.path.exists(script_path):
        subprocess.run(["bash", script_path], check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rkllm_model_path", type=str, required=True)
    parser.add_argument("--vision_model_path", type=str, required=True)
    parser.add_argument("--imgenc_path", type=str, required=True)
    parser.add_argument("--target_platform", type=str, default="rk3588")
    parser.add_argument("--model_name", type=str, default="qwen3-vl-4b-rk3588")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--rknn_core_num", type=int, default=3)
    parser.add_argument("--max_context_len", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--img_start", type=str, default="<|vision_start|>")
    parser.add_argument("--img_end", type=str, default="<|vision_end|>")
    parser.add_argument("--img_content", type=str, default="<|image_pad|>")
    parser.add_argument("--fix_freq", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    SERVER_ARGS = parse_args()

    if not os.path.exists(SERVER_ARGS.rkllm_model_path):
        raise FileNotFoundError(f"rkllm model not found: {SERVER_ARGS.rkllm_model_path}")
    if not os.path.exists(SERVER_ARGS.vision_model_path):
        raise FileNotFoundError(f"vision model not found: {SERVER_ARGS.vision_model_path}")
    if not os.path.exists(SERVER_ARGS.imgenc_path):
        raise FileNotFoundError(f"imgenc not found: {SERVER_ARGS.imgenc_path}")

    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    if SERVER_ARGS.fix_freq:
        maybe_fix_frequency(SERVER_ARGS.target_platform)

    RK_MODEL = RKLLMServerModel(
        model_path=SERVER_ARGS.rkllm_model_path,
        platform=SERVER_ARGS.target_platform,
        max_context_len=SERVER_ARGS.max_context_len,
        max_new_tokens=SERVER_ARGS.max_new_tokens,
        img_start=SERVER_ARGS.img_start,
        img_end=SERVER_ARGS.img_end,
        img_content=SERVER_ARGS.img_content,
    )

    app.run(host=SERVER_ARGS.host, port=SERVER_ARGS.port, threaded=True, debug=False)
