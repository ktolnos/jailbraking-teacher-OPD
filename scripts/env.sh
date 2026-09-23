# Project environment. Source before running anything in this repo's venv:
#   source scripts/env.sh
#
# Why this exists: flashinfer JIT-compiles its sampling kernels at vLLM warmup
# and needs nvcc. The compute nodes have no system CUDA toolkit
# (cuda_home='/usr/local/cuda' does not exist) and `module load cuda/13.2`
# sets neither CUDA_HOME nor PATH here. But the venv already ships a matching
# toolkit via the nvidia-cuda-nvcc wheel (13.4) alongside torch's CUDA 13.0 --
# same major, so it drives flashinfer's JIT fine.
#
# The wheel set also needs pinning to build at all -- see constraints.txt.
# Reinstall with:  uv pip install -c constraints.txt <pkg>
#
# Deliberately NOT in ~/.bashrc: that file is shared with the sibling devbox
# slots and with ~/Reward-tampering and ~/reasoning-distillation, whose venvs
# pin different CUDA majors. A global CUDA_HOME into this venv would break
# their jobs.
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_cu="$_repo/.venv/lib/python3.12/site-packages/nvidia/cu13"

if [ -x "$_cu/bin/nvcc" ]; then
    export CUDA_HOME="$_cu"
    export CUDA_PATH="$_cu"
    case ":$PATH:" in
        *":$_cu/bin:"*) ;;
        *) export PATH="$_cu/bin:$PATH" ;;
    esac
    case ":$LD_LIBRARY_PATH:" in
        *":$_cu/lib:"*) ;;
        *) export LD_LIBRARY_PATH="$_cu/lib:${LD_LIBRARY_PATH}" ;;
    esac
    # The pip CUDA wheels lay libraries out as lib/, but flashinfer links with
    # -L$CUDA_HOME/lib64 and -L$CUDA_HOME/lib64/stubs, and ships neither an
    # unversioned libcudart.so nor a libcuda.so driver stub. Without these three
    # links the compiles succeed and the final `c++ -shared` step dies with
    # "cannot find -lcudart / -lcuda". Idempotent, so a venv rebuild self-heals.
    [ -e "$_cu/lib64" ] || ln -s lib "$_cu/lib64"
    [ -e "$_cu/lib/libcudart.so" ] || ln -s libcudart.so.13 "$_cu/lib/libcudart.so"
    if [ ! -e "$_cu/lib/stubs/libcuda.so" ]; then
        mkdir -p "$_cu/lib/stubs"
        # link against the node's real driver lib; the .so symlink is stable
        for _d in /usr/lib64/libcuda.so /usr/lib64/nvidia/libcuda.so \
                  /usr/lib/x86_64-linux-gnu/libcuda.so; do
            [ -e "$_d" ] && { ln -s "$_d" "$_cu/lib/stubs/libcuda.so"; break; }
        done
    fi
else
    echo "WARNING: nvcc not found at $_cu/bin/nvcc -- flashinfer JIT will fail" >&2
fi
unset _repo _cu _d
