"""Проверка гипотезы: можно ли получить рабочий nvcc без root через pip-wheel и
собрать им CUDA-расширение torch. Если да — apex/TransformerEngine на платформе реальны."""

import glob
import os
import subprocess
import sys
import time

USERBASE = os.environ.get("PYTHONUSERBASE", os.path.expanduser("~/.local"))
CUDA_HOME = "/home/jovyan/.cuda_home"  # синтетический CUDA_HOME, собираем из wheel'ов


def sh(cmd, timeout=600):
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return out.returncode, (out.stdout + out.stderr).strip()


def step(title):
    print(f"\n===== {title} =====", flush=True)


def find_under_site(relpath):
    """Ищет файл внутри установленных nvidia-* wheel'ов (PYTHONUSERBASE)."""
    hits = glob.glob(os.path.join(USERBASE, "lib", "python*", "site-packages", "nvidia", relpath), recursive=True)
    return hits


def find_by_name(name):
    """Надёжный поиск исполняемого файла по всему PYTHONUSERBASE."""
    rc, out = sh(f"find {USERBASE} -name '{name}' -type f 2>/dev/null")
    return [p for p in out.splitlines() if p.strip()]


def main():
    print("nvcc-проба старт:", time.strftime("%F %T"), flush=True)
    print("PYTHONUSERBASE:", USERBASE, flush=True)

    step("1. Ставлю pip-wheel'ы CUDA toolkit (nvcc + заголовки + cccl)")
    pkgs = "nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cccl-cu12"
    rc, out = sh(f"pip install --user {pkgs} 2>&1")
    print("\n".join(out.splitlines()[-25:]), flush=True)
    print("pip rc:", rc, flush=True)

    step("2. Где оказался nvcc")
    print("--- содержимое cuda_nvcc/bin/ ---", flush=True)
    print(sh(f"ls -la {USERBASE}/lib/python*/site-packages/nvidia/cuda_nvcc/bin/ 2>&1")[1], flush=True)
    print("--- всё, где встречается 'nvcc' в имени ---", flush=True)
    print(sh(f"find {USERBASE} /home/user/conda -iname '*nvcc*' 2>/dev/null | head -20")[1], flush=True)
    nvcc_hits = find_under_site("cuda_nvcc/bin/nvcc") or find_by_name("nvcc")
    print("nvcc-кандидаты:", nvcc_hits, flush=True)
    if not nvcc_hits:
        print("nvcc из wheel не найден — гипотеза не сработала на этом шаге", flush=True)
        return
    nvcc = nvcc_hits[0]

    step("3. Собираю синтетический CUDA_HOME из кусочков wheel'ов")
    nvcc_root = os.path.dirname(os.path.dirname(nvcc))          # .../nvidia/cuda_nvcc
    runtime_hits = find_under_site("cuda_runtime/include/cuda_runtime.h")
    cccl_hits = find_under_site("cu*/include/cuda/std/version") or find_under_site("cuda_cccl/include")
    print("cuda_runtime.h:", runtime_hits, flush=True)
    print("cccl include:", cccl_hits[:2], flush=True)

    os.makedirs(f"{CUDA_HOME}/bin", exist_ok=True)
    os.makedirs(f"{CUDA_HOME}/include", exist_ok=True)
    os.makedirs(f"{CUDA_HOME}/lib64", exist_ok=True)
    # bin: nvcc + вспомогательные (cicc, ptxas и пр.)
    sh(f"ln -sf {nvcc_root}/bin/* {CUDA_HOME}/bin/ 2>/dev/null")
    sh(f"ln -sf {nvcc_root}/nvvm {CUDA_HOME}/nvvm 2>/dev/null")
    # include: заголовки nvcc + runtime + cccl
    sh(f"ln -sf {nvcc_root}/include/* {CUDA_HOME}/include/ 2>/dev/null")
    if runtime_hits:
        rt_inc = os.path.dirname(runtime_hits[0])
        sh(f"ln -sf {rt_inc}/* {CUDA_HOME}/include/ 2>/dev/null")
    for c in cccl_hits:
        sh(f"ln -sf {c}/* {CUDA_HOME}/include/ 2>/dev/null")
    print("собран CUDA_HOME:", CUDA_HOME, flush=True)
    print(sh(f"ls -la {CUDA_HOME}/bin | head")[1], flush=True)

    step("4. nvcc --version")
    os.environ["CUDA_HOME"] = CUDA_HOME
    os.environ["PATH"] = f"{CUDA_HOME}/bin:{USERBASE}/bin:" + os.environ.get("PATH", "")
    rc, out = sh("nvcc --version")
    print(out, flush=True)
    print("nvcc rc:", rc, flush=True)
    if rc != 0:
        print("nvcc не запускается — стоп", flush=True)
        return

    step("5. nvcc компилирует тривиальный .cu в PTX")
    with open("/tmp/k.cu", "w") as fh:
        fh.write("__global__ void k(float* x){ x[threadIdx.x]+=1.0f; }\n")
    rc, out = sh("nvcc -ptx /tmp/k.cu -o /tmp/k.ptx -arch=sm_90 2>&1")
    print(out or "(без вывода)", flush=True)
    print("компиляция .cu -> PTX:", "OK" if rc == 0 else f"ПРОВАЛ rc={rc}", flush=True)

    step("6. Главный тест: torch собирает CUDA-расширение из исходников")
    try:
        import torch
        from torch.utils import cpp_extension
        print("torch:", torch.__version__, "| cuda avail:", torch.cuda.is_available(), flush=True)
        print("cpp_extension.CUDA_HOME:", cpp_extension.CUDA_HOME, flush=True)
        build_dir = "/home/jovyan/.cache/torch_ext_nvcc"
        os.makedirs(build_dir, exist_ok=True)
        cuda_src = (
            "#include <torch/extension.h>\n"
            "__global__ void addone(float* x){ x[threadIdx.x] += 1.0f; }\n"
            "torch::Tensor run_addone(torch::Tensor t){\n"
            "  addone<<<1, t.numel()>>>(t.data_ptr<float>()); return t; }\n"
        )
        mod = cpp_extension.load_inline(
            name="mlsub_nvcc_cuda",
            cpp_sources="torch::Tensor run_addone(torch::Tensor t);",
            cuda_sources=cuda_src,
            functions=["run_addone"],
            build_directory=build_dir,
            verbose=True,
        )
        out = mod.run_addone(torch.zeros(8, device="cuda"))
        print("СБОРКА CUDA-РАСШИРЕНИЯ TORCH: OK, sum =", float(out.sum().item()), flush=True)
        print(">>> ВЫВОД: apex/TransformerEngine на платформе СОБИРАЕМЫ через pip-nvcc", flush=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f">>> СБОРКА CUDA-РАСШИРЕНИЯ: ПРОВАЛ -> {type(exc).__name__}: {exc}", flush=True)

    print("\nnvcc-проба финиш:", time.strftime("%F %T"), flush=True)


if __name__ == "__main__":
    main()
