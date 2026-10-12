"""Start local services, exercise HTTP endpoints, then stop only our child processes."""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minillm.common import write_json


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read()


def wait_for(proc, url):
    deadline = time.monotonic()+60
    last_error = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Child service exited with {proc.returncode}; inspect reports service logs.")
        try:
            return get(url)
        except (OSError, TimeoutError) as exc:
            last_error = exc
            time.sleep(0.2)
    raise TimeoutError(f"Service failed to become ready: {last_error}")


def verify():
    results = {}
    env = {**os.environ, "PYTHONUTF8": "1"}
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    (ROOT / "reports").mkdir(exist_ok=True)
    api_port = free_port()
    with (ROOT / "reports" / "api-server.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-m", "minillm", "serve", "--run", "smoke", "--stage", "dpo", "--port", str(api_port)], cwd=ROOT, env=env, creationflags=flags, stdout=log, stderr=subprocess.STDOUT)
        try:
            base = f"http://127.0.0.1:{api_port}"
            results["health"] = json.loads(wait_for(proc, base+"/health"))
            request = urllib.request.Request(base+"/v1/chat/completions", data=json.dumps({"messages": [{"role": "user", "content": "Explain LoRA."}], "max_tokens": 8, "temperature": 0.8, "top_p": 0.9, "use_cache": True}).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=20) as response:
                result = json.load(response)
                assert response.status == 200
                assert result["usage"]["completion_tokens"] <= 8
                results["chat_completion"] = result
            results["api_metrics"] = json.loads(get(base+"/metrics"))
            assert results["api_metrics"]["requests"] == 1
            print("Local Uvicorn API: health, generation and metrics passed.", flush=True)
        finally:
            proc.terminate()
            proc.wait(timeout=20)
    tb_port = free_port()
    with (ROOT / "reports" / "tensorboard-server.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-m", "tensorboard.main", "--logdir", str(ROOT / "artifacts" / "smoke" / "tensorboard"), "--host", "127.0.0.1", "--port", str(tb_port)], cwd=ROOT, env=env, creationflags=flags, stdout=log, stderr=subprocess.STDOUT)
        try:
            base = f"http://127.0.0.1:{tb_port}"
            wait_for(proc, base)
            tags = json.loads(get(base+"/data/plugin/scalars/tags"))
            assert "base" in tags and "train_loss" in tags["base"]
            results["tensorboard_scalar_tags"] = tags
            print("Local TensorBoard: HTTP and recorded scalar tags passed.", flush=True)
        finally:
            proc.terminate()
            proc.wait(timeout=20)
    write_json(ROOT / "reports" / "services.json", results)


if __name__ == "__main__":
    verify()
