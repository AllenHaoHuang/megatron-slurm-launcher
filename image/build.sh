#!/bin/bash
set -Eeuo pipefail
trap 'build_rc=$?; printf "ERROR: build.sh line %s exited %s: %s\n" "$LINENO" "$build_rc" "$BASH_COMMAND" >&2; exit "$build_rc"' ERR
printf 'build.sh started: host=%s job=%s cwd=%s\n' "$(hostname)" "${SLURM_JOB_ID:-none}" "$PWD"

# Run on an allocated GH200 compute node, outside a container.
: "${SLURM_JOB_ID:?Run this script inside a Slurm compute allocation}"
[[ $(uname -m) == aarch64 ]] || { echo 'This Dockerfile targets aarch64/GH200.' >&2; exit 1; }
# sbatch executes a spool copy of this script; $0 is not the source directory.
# BUILD_CONTEXT lets Slurm start in home while reading the build files from scratch.
build_context="${BUILD_CONTEXT:-${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}}"
printf 'Build context: %s\n' "$build_context"
cd -- "$build_context"
[[ -f Dockerfile ]] || { echo "Dockerfile not found in $PWD; submit from the build directory." >&2; exit 1; }
command -v podman || { echo 'podman is not available on PATH.' >&2; exit 1; }
command -v enroot || { echo 'enroot is not available on PATH.' >&2; exit 1; }
if [[ ${BUILD_CHECK_ONLY:-0} == 1 ]]; then
    echo 'Build startup checks passed; no image build requested.'
    exit 0
fi

BUILD_NAME=${BUILD_NAME:-megachonk-$SLURM_JOB_ID}
[[ "$BUILD_NAME" =~ ^[a-z0-9][a-z0-9._-]*$ ]] || { echo 'Invalid BUILD_NAME.' >&2; exit 1; }
build_root="/ritom/scratch/cscs/$USER/megachonk-image"
build_state="/dev/shm/$USER/megachonk-build-$SLURM_JOB_ID"
image_tag="localhost/$BUILD_NAME:ritom-$SLURM_JOB_ID"
image_file="$build_root/$BUILD_NAME.sqsh"
edf_file="$build_root/$BUILD_NAME.toml"
mkdir -p "$build_root" "$build_state" "$build_root/tmp" "$build_root/enroot-cache" "$build_root/enroot-data"
[[ ! -e "$image_file" && ! -e "$edf_file" ]] || { echo 'Output already exists; refusing to overwrite.' >&2; exit 1; }

# Job-specific configuration; do not alter the user's existing Podman config.
export CONTAINERS_STORAGE_CONF="$build_state/storage.conf"
cat > "$CONTAINERS_STORAGE_CONF" <<EOF
[storage]
driver = "overlay"
runroot = "$build_state/runroot"
graphroot = "$build_state/root"
EOF
# Buildah's temporary rootfs must be node-local; Ritom failed with permission denied.
export TMPDIR="$build_state/tmp"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"
export ENROOT_CACHE_PATH="$build_root/enroot-cache"
export ENROOT_DATA_PATH="$build_root/enroot-data"
export ENROOT_TEMP_PATH="$build_root/tmp"
export ENROOT_RUNTIME_PATH="$build_state/enroot-runtime"
mkdir -p "$ENROOT_RUNTIME_PATH"

podman info
df -h /dev/shm "$build_root"
podman build --network=host --layers -f Dockerfile -t "$image_tag" "$@" .

# Surface the tokenizer import failure before exporting a training image.
podman run --rm --network=host -e NVIDIA_VISIBLE_DEVICES=void \
    --entrypoint python3 "$image_tag" \
    -c 'import sys; from transformers import AutoTokenizer; print(sys.executable); print("AutoTokenizer import OK")'

# Must export before this allocation ends: Podman storage is node-local.
enroot import -x mount -o "$image_file" "podman://$image_tag"
cat > "$edf_file" <<EOF
image = "$image_file"
mounts = ["/iopsstor/", "/ritom/"]
writable = true
entrypoint = true

[env]
PMIX_MCA_psec = "native"

[annotations]
com.hooks.netstack.source = "host"
com.hooks.cxi.enabled = "false"
com.hooks.aws_ofi_nccl.enabled = "false"
EOF
printf '\nImage: %s\nEDF: %s\n' "$image_file" "$edf_file"
printf 'Use: export CONTAINER="%s"\n' "$edf_file"
