#!/usr/bin/env bash
# One-time setup on a RunPod pod (use an official PyTorch template; torch + CUDA come preinstalled).
# Keep the project under /workspace: that's the persistent volume. Everything else is wiped when the pod stops.
set -euo pipefail
cd "$(dirname "$0")/.."

case "$(pwd)" in
  /workspace/*) ;;
  *) echo "WARNING: $(pwd) is not under /workspace; runs here are lost when the pod stops." ;;
esac

pip install -q -r requirements.txt   # no-op for torch if the template already has it
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "no GPU visible: wrong template or pod type"
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0),
      f"| {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB")
EOF
python train.py signal --iters 3 --num-envs 32 --out /tmp/smoke --name smoke > /tmp/smoke.log 2>&1 \
  && echo "smoke test ok" || { cat /tmp/smoke.log; exit 1; }
