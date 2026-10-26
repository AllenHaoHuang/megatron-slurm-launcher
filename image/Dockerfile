# Experimental comparison image: upstream DeepGEMM PR432 + PR375 pre-drain.
# Build and test on GH200 before replacing the training image.
# FROM jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/apertus-2:alps5-dev
# FROM jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-nemo:25.11.01-alps4-dev
#FROM jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:25.12-py3-alps3
FROM jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:25.12-py3-alps6

# ---- UCCL-EP (MoE flex/DeepEP token dispatcher) -----------------------------
# Builds what common/uccl.sh:install_uccl() builds per job, so jobs can skip the
# srun build. Lands in site-packages as `uccl.ep` (native ext) + `deep_ep` (the
# drop-in wrapper Megatron's flex dispatcher imports), so no PYTHONPATH needed.
# Move the source pin at build time:
#   sbatch build.sh <tag> --build-arg UCCL_REF=<sha>
ARG UCCL_REPO=https://github.com/swiss-ai/uccl.git
ARG UCCL_BRANCH=slingshot-dev-clean
ARG UCCL_REF=fa4c325c
ARG UCCL_NVL_PEERS=4
ARG UCCL_CUDA_ARCH=9.0
ARG UCCL_MAX_JOBS=32
# NB: the last three are deliberately NOT named NUM_MAX_NVL_PEERS /
# TORCH_CUDA_ARCH_LIST / MAX_JOBS. An ENV in the base image beats an ARG of the
# same name, and this image ships TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0 10.0 11.0
# 12.0+PTX" — which silently won, built all seven gencodes, and exhausted the
# container fd limit mid-build.

# Pinned versions
ARG TE_REF=v2.12
ARG FLA_REF=v0.5.2
ARG NVRX_EXT_VERSION=0.6.0
ARG CAUSAL_CONV1D_REF=v1.7.0

# Give the venv its own pip. /opt/venv is a uv venv (uv venvs ship without
# pip), so bare `pip` falls through PATH to /usr/local/bin/pip -> /usr/bin/
# python, which installs into /usr/local's dist-packages — a tier the venv
# shadows at run time (fla stayed 0.4.0 for two image generations this way).
# /opt/venv/bin precedes /usr/local/bin on PATH, so with /opt/venv/bin/pip
# present every later bare `pip` — build layers and interactive use alike —
# targets the interpreter that actually runs training. --force-reinstall
# because pip otherwise sees the system-site copy as already satisfied
# (ensurepip is not an option: Debian disables it on the alps6 base's python).
RUN python3 -m pip install --force-reinstall pip

# FlashAttention-3 for GH200 (Hopper/SM90), alongside the base image's FA2.
# Pin newer upstream hopper source with native CUDA 13 toolchain handling.
# Force an aarch64 source build against the existing PyTorch, without replacing it.
# CUDA 13 puts CCCL under include/cccl; FA3's bundled nvcc needs it explicitly.
# This revision uses system nvcc on CUDA 13 and sets the CCCL include paths.
ARG FA3_REF=28ef22c99a135b234fb54bc33cfb638078bacb65
ARG FA3_MAX_JOBS=4
# Workaround to try for CUDA 13 ptxas C7907 on FA3 BF16 forward kernels.
# Lower optimization may affect throughput; verify on GPU and benchmark.
# Set --build-arg FA3_PTXAS_OPT=3 to restore the normal optimization level.
ARG FA3_PTXAS_OPT=1
RUN set -eux; \
    ulimit -n "$(ulimit -Hn)" || true; \
    python3 -m pip install ninja packaging wheel einops; \
    git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention; \
    git -C /tmp/flash-attention checkout "$FA3_REF"; \
    git -C /tmp/flash-attention rev-parse HEAD > /opt/fa3-source-commit.txt; \
    git -C /tmp/flash-attention submodule update --init csrc/cutlass; \
    cd /tmp/flash-attention/hopper; \
    test -f /usr/local/cuda/include/cccl/cuda/std/utility; \
    /usr/local/cuda/bin/nvcc --version; \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    FLASH_ATTENTION_DISABLE_SM80=TRUE \
    CPATH=/usr/local/cuda/include/cccl${CPATH:+:$CPATH} \
    PYTORCH_NVCC=/usr/local/cuda/bin/nvcc \
    NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS }-Xptxas=-O${FA3_PTXAS_OPT}" \
    MAX_JOBS=$FA3_MAX_JOBS NVCC_THREADS=2 CUDA_HOME=/usr/local/cuda \
        python3 -m pip install --no-build-isolation --no-deps --force-reinstall .

# Use one module instance for both upstream and Transformer Engine import paths.
# Validate outside the source checkout, whose egg-info can shadow installed metadata.
# Separate from compilation so a validation failure does not discard its layer.
RUN set -eux; \
    SITE="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    mkdir -p "$SITE/flash_attn_3"; \
    touch "$SITE/flash_attn_3/__init__.py"; \
    printf '%s\n' 'import sys' 'import flash_attn_interface as _impl' 'sys.modules[__name__] = _impl' > "$SITE/flash_attn_3/flash_attn_interface.py"; \
    printf '%s\n' 'import sys' 'import flash_attn_config as _impl' 'sys.modules[__name__] = _impl' > "$SITE/flash_attn_3/flash_attn_config.py"; \
    cd /; \
    python3 -c 'import torch, sysconfig; from pathlib import Path; from importlib.metadata import distribution; import flash_attn_interface as upstream; from flash_attn_3 import flash_attn_interface as fa3; import flash_attn_3._C as kernels; site = Path(sysconfig.get_paths()["purelib"]).resolve(); dist = Path(distribution("flash-attn-3").locate_file("")).resolve(); print("FA3 paths:", site, fa3.__file__, kernels.__file__, dist, flush=True); assert Path(fa3.__file__).resolve().is_relative_to(site), ("interface outside site", fa3.__file__, site); assert Path(kernels.__file__).resolve().is_relative_to(site), ("extension outside site", kernels.__file__, site); assert dist.is_relative_to(site), ("metadata outside site", dist, site); assert fa3 is upstream, "FA3 interface loaded twice"; assert callable(fa3.flash_attn_func) and callable(fa3.flash_attn_varlen_func), "missing FA3 entry points"; print("FA3 import OK")'; \
    rm -rf /tmp/flash-attention

# DeepGEMM 2.8.0: upstream Public Release 26/09 (PR432), pinned for comparison.
# GH200/SM90: drain pending TMA reads BEFORE replacing expert descriptors.
# Port PR375 pre-drain to this pinned source; verify the installed header.
COPY patch_deepgemm_pr375.py /opt/patch_deepgemm_pr375.py
ARG DEEPGEMM_REPO=https://github.com/deepseek-ai/DeepGEMM.git
ARG DEEPGEMM_REF=ab69f76be5bb9ea3499bc755002b1a876cb0b3d9
ARG DEEPGEMM_MAX_JOBS=4
ARG DEEPGEMM_PR375_COMMIT=4e4e54036becc2a1ee9b5470015566673045acfd
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libdw-dev libelf-dev \
    && test -r /usr/include/elfutils/libdwfl.h \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    python3 -m pip uninstall -y deep_gemm; \
    git clone "$DEEPGEMM_REPO" /opt/deepgemm; \
    cd /opt/deepgemm; \
    git checkout "$DEEPGEMM_REF"; \
    git rev-parse HEAD > /opt/deepgemm-pr375.txt; \
    python3 /opt/patch_deepgemm_pr375.py; \
    printf 'PR375=%s\n' "$DEEPGEMM_PR375_COMMIT" >> /opt/deepgemm-pr375.txt; \
    git submodule update --init --recursive; \
    DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=1 \
    MAX_JOBS=$DEEPGEMM_MAX_JOBS CUDA_HOME=/usr/local/cuda \
    CPATH=/usr/local/cuda/include/cccl${CPATH:+:$CPATH} \
        python3 -m pip install --no-build-isolation --no-deps --force-reinstall .; \
    cd /; \
    python3 -c 'from pathlib import Path; import sysconfig; from importlib.metadata import distribution; d = distribution("deep_gemm"); assert d.version.split("+")[0] == "2.8.0", ("Unexpected DeepGEMM version", d.version); site = Path(sysconfig.get_paths()["purelib"]).resolve(); h = Path(d.locate_file("deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh")).resolve(); assert h.is_relative_to(site), ("Installed header outside site", h, site); s = h.read_text(); assert s == Path("/opt/deepgemm/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh").read_text(), "Installed DeepGEMM header differs from patched source"; b = s.split("last_group_idx = scheduler.current_group_idx;", 1)[1].split("ptx::tensor_map_acquire_gpu", 1)[0]; r = b.index("ptx::tensor_map_replace_global_addr_in_smem"); assert b.index("cute::tma_desc_commit_group();") < b.index("cute::tma_desc_wait_group();") < b.index("__syncwarp(") < r, "PR375 pre-drain missing"; assert "cute::tma_desc_wait_group();" in b[r:], "Post-update wait missing"; print("DeepGEMM PR375 installed:", d.version, h)'; \
    cat /opt/deepgemm-pr375.txt

RUN pip uninstall -y deep_ep uccl
RUN pip install nanobind
RUN git clone --single-branch --branch $UCCL_BRANCH $UCCL_REPO /opt/uccl \
    && git -C /opt/uccl checkout $UCCL_REF

# ep/setup.py picks -DUSE_GRACE_HOPPER from `nvidia-smi --query-gpu=name` (GH200
# + aarch64) and offers no env override, but `podman build` sees no GPU — without
# a stand-in the image would silently get a non-GH200 build. The stub answers the
# two queries setup.py makes and is removed inside the same layer, so it can
# never shadow the real nvidia-smi at run time.
RUN set -eux; \
    ulimit -n "$(ulimit -Hn)" || true; \
    SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    if [ ! -e /usr/lib/aarch64-linux-gnu/libibverbs.so ] && [ -e /usr/lib/aarch64-linux-gnu/libibverbs.so.1 ]; then \
        ln -sfn /usr/lib/aarch64-linux-gnu/libibverbs.so.1 /usr/lib/aarch64-linux-gnu/libibverbs.so; \
    fi; \
    mkdir -p /tmp/gpustub "$SITE/uccl"; \
    printf '#!/bin/sh\ncase "$*" in\n*query-gpu=name*) echo "NVIDIA GH200 120GB";;\n*query-gpu=compute_cap*) echo "9.0";;\n*) exit 1;;\nesac\n' > /tmp/gpustub/nvidia-smi; \
    chmod +x /tmp/gpustub/nvidia-smi; \
    touch "$SITE/uccl/__init__.py"; \
    cd /opt/uccl/ep; \
    PATH=/tmp/gpustub:$PATH \
    USE_LIBFABRIC_CXI=1 \
    NUM_MAX_NVL_PEERS=$UCCL_NVL_PEERS \
    TORCH_CUDA_ARCH_LIST=$UCCL_CUDA_ARCH \
    MAX_JOBS=$UCCL_MAX_JOBS \
    CUDA_HOME=/usr/local/cuda \
    INSTALL_DIR="$SITE/uccl" \
    python3 setup.py build --build-base /tmp/uccl-build install; \
    rm -rf "$SITE/deep_ep"; \
    cp -aL /opt/uccl/ep/deep_ep_wrapper/deep_ep "$SITE/deep_ep"; \
    rm -rf /tmp/gpustub /tmp/uccl-build /opt/uccl/ep/*.egg-info; \
    ls -l "$SITE/uccl" "$SITE/deep_ep"; \
    python3 -c 'import uccl.ep, deep_ep; print(uccl.ep.__file__); print(deep_ep.Buffer)' \
        || echo 'WARNING: uccl.ep/deep_ep import failed in the build container (no GPU here); re-check on a compute node'

# Selects UCCL's RDMA transport (ep/src/proxy.cpp: use_cxi_transport_ is true
# only for the exact string "cxi"). Anything else takes the ibverbs path, which
# on Slingshot finds no verbs device and aborts every rank with
# "[FATAL] Selected RDMA NIC '' not found in verbs device list". common/uccl.sh
# exports it too, but only when USE_UCCL=true — and this image makes deep_ep
# importable regardless, so the default belongs here.
ENV UCCL_EP_TRANSPORT=cxi

# te for cuda graph
RUN pip uninstall -y transformer_engine
RUN pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@$TE_REF

# grouped_gemm
RUN pip uninstall -y grouped_gemm
# COPY ./nv_grouped_gemm-1.1.4.post8-cp312-cp312-linux_aarch64.whl /tmp/
# RUN pip install /tmp/nv_grouped_gemm-1.1.4.post8-cp312-cp312-linux_aarch64.whl
RUN pip install --no-build-isolation git+https://github.com/FFGGSSJJ/grouped_gemm.git@dev/grad_acc_fuse

# Megatron-LM pins nvidia-resiliency-ext==0.6.0 (pyproject.toml)
RUN pip uninstall -y nvidia-resiliency-ext \
    && pip install nvidia-resiliency-ext==$NVRX_EXT_VERSION \
    && python3 -c "import nvidia_resiliency_ext as n; assert n.__version__ == '$NVRX_EXT_VERSION', n.__version__"

# switch to community version
RUN pip install --no-build-isolation git+https://github.com/FFGGSSJJ/Emerging-Optimizers.git@fuguan/moonlight_scale

# fla
# Bare `pip` here is the SYSTEM python's (/usr/local/bin/pip -> /usr/bin/python):
# it sees /opt/venv through the base image's PYTHONPATH but installs into
# /usr/local's dist-packages, where the venv's own `fla` — shipped as the
# fla-core dist, 0.4.0 — shadows it at run time. python3 -m pip targets the
# venv the training python actually imports from; fla-core must go too (the
# flash-linear-attention uninstall alone never touches the venv copy).
RUN python3 -m pip uninstall -y fla-core flash-linear-attention
RUN python3 -m pip install --no-build-isolation git+https://github.com/fla-org/flash-linear-attention@$FLA_REF

# causal-conv1d
ARG CAUSAL_CONV1D_MAX_JOBS=4
RUN python3 -m pip uninstall -y causal_conv1d \
    && CAUSAL_CONV1D_FORCE_BUILD=TRUE \
       MAX_JOBS=$CAUSAL_CONV1D_MAX_JOBS \
       CUDA_HOME=/usr/local/cuda \
       python3 -m pip install --no-build-isolation \
           git+https://github.com/Dao-AILab/causal-conv1d.git@$CAUSAL_CONV1D_REF

# wandb
RUN python3 -m pip install --upgrade wandb


# Tier check: fail the build if any pinned package resolves from outside the
# interpreter's own site (sysconfig purelib — the venv on nemo-style bases,
# the system site where python3 IS the system python). A version-only assert
# passes via the system-site fall-through, so this pins location too — that is
# exactly how fla 0.4.0 went unnoticed.
RUN python3 -c "import sysconfig; SITE = sysconfig.get_paths()['purelib']; import fla; \
    assert fla.__version__ == '${FLA_REF#v}' and fla.__file__.startswith(SITE), (fla.__version__, fla.__file__); \
    from importlib.metadata import version, distribution; \
    assert version('transformer-engine').startswith('${TE_REF#v}'), version('transformer-engine'); \
    assert version('nvidia-resiliency-ext') == '$NVRX_EXT_VERSION', version('nvidia-resiliency-ext'); \
    assert version('causal_conv1d') == '${CAUSAL_CONV1D_REF#v}', version('causal_conv1d'); \
    assert all(str(distribution(d).locate_file('')).startswith(SITE) \
        for d in ('flash-linear-attention', 'transformer-engine', 'wandb', 'nvidia-resiliency-ext', \
                  'causal_conv1d'))"
