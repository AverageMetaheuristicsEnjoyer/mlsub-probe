"""Инвентаризация контейнера Cloud.ru: что реально доступно задаче mlsub."""

import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

WORKSPACE = "/home/jovyan"


def section(title):
    print(f"\n===== {title} =====", flush=True)


def run(cmd):
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return (out.stdout + out.stderr).strip() or "(пусто)"
    except Exception as exc:
        return f"(не выполнилось: {exc})"


def probe_python():
    section("PYTHON")
    print("version:", sys.version.replace("\n", " "), flush=True)
    print("executable:", sys.executable, flush=True)
    print("platform:", platform.platform(), flush=True)
    print("user:", run("id"), flush=True)
    print("hostname:", socket.gethostname(), flush=True)


def probe_env():
    section("ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ")
    keys = [
        "PYTHONUSERBASE", "PYTHONPATH", "PYTHONUNBUFFERED", "PATH",
        "HF_HOME", "TORCH_HOME", "CUDA_HOME", "CUDA_VISIBLE_DEVICES",
        "WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
        "OMPI_COMM_WORLD_SIZE", "SLURM_JOB_ID",
    ]
    for key in keys:
        print(f"{key}={os.environ.get(key, '(не задана)')}", flush=True)


def probe_resources():
    section("CPU / RAM")
    print("cpu_count:", os.cpu_count(), flush=True)
    print("affinity:", len(os.sched_getaffinity(0)), flush=True)
    for line in open("/proc/meminfo").read().splitlines()[:3]:
        print(line, flush=True)
    print("cgroup memory.max:", run("cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null"), flush=True)


def probe_gpu():
    section("GPU")
    print(run("nvidia-smi"), flush=True)
    print("--- topology ---", flush=True)
    print(run("nvidia-smi topo -m"), flush=True)
    try:
        import torch
    except ImportError:
        print("torch НЕ импортируется", flush=True)
        return
    print("torch:", torch.__version__, flush=True)
    print("torch.version.cuda:", torch.version.cuda, flush=True)
    print("cuda.is_available:", torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        return
    print("device_count:", torch.cuda.device_count(), flush=True)
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {props.name} | {props.total_memory / 1024**3:.1f} GiB | sm_{props.major}{props.minor}", flush=True)
    print("--- matmul smoke ---", flush=True)
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        a @ a
    torch.cuda.synchronize()
    tflops = 10 * 2 * 4096**3 / (time.time() - start) / 1e12
    print(f"bf16 4096^3 matmul: {tflops:.1f} TFLOP/s", flush=True)


def probe_build_toolchain():
    section("ТУЛЧЕЙН ДЛЯ СБОРКИ РАСШИРЕНИЙ (apex / transformer-engine)")
    for tool in ("gcc --version", "g++ --version", "nvcc --version", "ninja --version", "cmake --version"):
        print(f"--- {tool} ---\n{run(tool)}", flush=True)
    print("which nvcc:", run("which nvcc"), flush=True)
    print("ls /usr/local/cuda*/bin/nvcc:", run("ls -d /usr/local/cuda*/bin/nvcc 2>/dev/null"), flush=True)

    if shutil.which("ninja") is None:
        print("--- ninja нет, ставлю pip install --user ninja ---", flush=True)
        print(run("pip install --user -q ninja 2>&1 | tail -5"), flush=True)
        # обновляем PATH, чтобы свежий ninja стал виден
        userbase = os.environ.get("PYTHONUSERBASE", os.path.expanduser("~/.local"))
        os.environ["PATH"] = f"{userbase}/bin:" + os.environ.get("PATH", "")
        print("ninja после установки:", run("ninja --version"), flush=True)

    try:
        from torch.utils import cpp_extension
        cuda_home = cpp_extension.CUDA_HOME
        print("cpp_extension.CUDA_HOME:", cuda_home, flush=True)
        build_dir = os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp/torch_ext")
        os.makedirs(build_dir, exist_ok=True)

        print("--- пробная сборка C++ расширения (нужен только g++ + ninja) ---", flush=True)
        mod = cpp_extension.load_inline(
            name="mlsub_probe_cpp",
            cpp_sources="int answer() { return 42; }",
            functions=["answer"],
            build_directory=build_dir,
            verbose=False,
        )
        print("сборка C++ расширения: OK, answer() =", mod.answer(), flush=True)

        # Настоящий тест на apex/TE: собирается ли CUDA-ядро из исходников.
        import torch
        if cuda_home and torch.cuda.is_available():
            print("--- пробная сборка CUDA расширения (тест на apex/TE) ---", flush=True)
            cuda_src = (
                "#include <torch/extension.h>\n"
                "__global__ void addone(float* x){ x[threadIdx.x] += 1.0f; }\n"
                "torch::Tensor run_addone(torch::Tensor t){\n"
                "  addone<<<1, t.numel()>>>(t.data_ptr<float>()); return t; }\n"
            )
            cmod = cpp_extension.load_inline(
                name="mlsub_probe_cuda",
                cpp_sources="torch::Tensor run_addone(torch::Tensor t);",
                cuda_sources=cuda_src,
                functions=["run_addone"],
                build_directory=build_dir,
                verbose=False,
            )
            out = cmod.run_addone(torch.zeros(8, device="cuda"))
            print("сборка CUDA расширения: OK, sum =", float(out.sum().item()), flush=True)
        else:
            print("CUDA расширение не проверялось (нет CUDA_HOME или GPU) — это и есть блокер для apex/TE, если так же на GPU-ноде", flush=True)
    except Exception as exc:
        print(f"сборка расширения: ПРОВАЛ -> {type(exc).__name__}: {exc}", flush=True)


def probe_workspace():
    section("ДИСК /home/jovyan")
    print("существует:", os.path.isdir(WORKSPACE), flush=True)
    if not os.path.isdir(WORKSPACE):
        return
    usage = shutil.disk_usage(WORKSPACE)
    print(f"всего {usage.total / 1024**3:.1f} GiB, свободно {usage.free / 1024**3:.1f} GiB", flush=True)
    print("--- содержимое ---", flush=True)
    print(run(f"ls -la {WORKSPACE}"), flush=True)

    marker = os.path.join(WORKSPACE, "mlsub_probe_marker.txt")
    if os.path.exists(marker):
        print("--- маркер прошлых запусков (диск переживает задачу) ---", flush=True)
        print(open(marker).read().strip(), flush=True)
    else:
        print("маркера нет — это первый запуск пробы", flush=True)
    with open(marker, "a") as fh:
        fh.write(f"run at {time.strftime('%F %T')} on {socket.gethostname()}\n")
    print("запись в /home/jovyan: OK", flush=True)

    print("--- скорость записи (1 GiB) ---", flush=True)
    print(run(f"dd if=/dev/zero of={WORKSPACE}/mlsub_probe_io.bin bs=1M count=1024 conv=fsync 2>&1; rm -f {WORKSPACE}/mlsub_probe_io.bin"), flush=True)


def probe_network():
    section("СЕТЬ")
    for url in ("https://pypi.org/simple/", "https://huggingface.co", "https://github.com"):
        try:
            start = time.time()
            with urllib.request.urlopen(url, timeout=20) as resp:
                print(f"{url} -> {resp.status} за {time.time() - start:.1f} с", flush=True)
        except Exception as exc:
            print(f"{url} -> НЕДОСТУПЕН: {type(exc).__name__}: {exc}", flush=True)


def main():
    print("проба стартовала:", time.strftime("%F %T"), flush=True)
    probe_python()
    probe_env()
    probe_resources()
    probe_gpu()
    probe_build_toolchain()
    probe_workspace()
    probe_network()
    print("\nпроба завершена:", time.strftime("%F %T"), flush=True)


if __name__ == "__main__":
    main()
