"""Run a grid of configs x seeds in parallel on one machine.

These models are tiny, so a single run leaves most of a rented GPU idle. Running 8-16 at once is
how you get your money's worth.

  python scripts/sweep.py composer --name tune1 --seeds 0 1 2 --iters 2000 --parallel 12 \
      --grid surprise_weight=1,2,4 ppo.ent_coef=0.01,0.02

Keys prefixed with 'ppo.' go to --ppo, everything else to --set. Grid values are comma-separated
(so list-valued kwargs can't be swept here; pass those through --fixed instead).
Each run lands in runs/<env>/<name>/<tag>; compare them with scripts/summarize.py.
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("env")
    p.add_argument("--name", required=True, help="sweep name, used as the folder under runs/<env>/")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--grid", nargs="*", default=[], help="key=v1,v2,... (prefix ppo. for PPO keys)")
    p.add_argument("--fixed", nargs="*", default=[], help="key=value applied to every run")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--num-envs", type=int)
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--out", default="runs")
    args = p.parse_args()

    keys, values = [], []
    for item in args.grid:
        k, v = item.split("=", 1)
        keys.append(k)
        values.append(v.split(","))

    jobs = []
    for combo in itertools.product(*values):
        for seed in args.seeds:
            pairs = list(zip(keys, combo)) + [tuple(f.split("=", 1)) for f in args.fixed]
            tag = "_".join(f"{k.removeprefix('ppo.')}{v}" for k, v in zip(keys, combo)) or "base"
            tag = f"{tag}_s{seed}"
            env_set = [f"{k}={v}" for k, v in pairs if not k.startswith("ppo.")]
            ppo_set = [f"{k[4:]}={v}" for k, v in pairs if k.startswith("ppo.")]
            cmd = [sys.executable, os.path.join(ROOT, "train.py"), args.env, "--seed", str(seed),
                   "--iters", str(args.iters), "--out", args.out, "--name", os.path.join(args.name, tag)]
            if args.num_envs:
                cmd += ["--num-envs", str(args.num_envs)]
            if env_set:
                cmd += ["--set", *env_set]
            if ppo_set:
                cmd += ["--ppo", *ppo_set]
            jobs.append((tag, cmd))

    sweep_dir = os.path.join(args.out, args.env, args.name)
    os.makedirs(sweep_dir, exist_ok=True)
    print(f"{len(jobs)} runs, {args.parallel} at a time -> {sweep_dir}")
    # Each torch process otherwise grabs every core and they thrash each other.
    env = {**os.environ, "OMP_NUM_THREADS": str(max(1, (os.cpu_count() or 1) // args.parallel))}
    running, failed, t0 = [], [], time.time()
    while jobs or running:
        while jobs and len(running) < args.parallel:
            tag, cmd = jobs.pop(0)
            log = open(os.path.join(sweep_dir, f"{tag}.stdout.log"), "w")
            running.append((tag, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=env), log))
            print(f"  start {tag}")
        for item in running[:]:
            tag, proc, log = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                if proc.returncode:
                    failed.append(tag)
                print(f"  done  {tag} {status}  [{(time.time() - t0) / 60:.1f} min]")
        time.sleep(2)
    print(f"finished in {(time.time() - t0) / 60:.1f} min" + (f", failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
