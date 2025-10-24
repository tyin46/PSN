"""
Display live progress for PRLT reversing trials and attempts.
Uses `PRLTSimulator.run_with_callback` to get per-trial updates and displays
progress bars for pre-reversal convergence and post-reversal switching.

Usage:
  python prlt_progress_monitor.py

This script will call the OpenAI API once per personality mix (same as the main PRLT script)
unless `use_api=False`.
"""

import time
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prlt_personality_proportions import ParamGenerator, AgentParams, PRLTSimulator

# tqdm is optional
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

OUTPUT_DIR = ROOT / 'prlt_progress'
OUTPUT_DIR.mkdir(exist_ok=True)

mixes = {
    'mostly_cautious': {'cautious':0.9, 'risk_taker':0.05, 'easily_tired':0.05},
    'mostly_risky': {'risk_taker':0.9, 'cautious':0.05, 'easily_tired':0.05},
    'impulsive_mix': {'risk_taker':0.5, 'cautious':0.1, 'easily_tired':0.2, 'unmotivated':0.2},
    'easily_tired': {'easily_tired':0.9, 'cautious':0.1}
}

USE_API = True
RUNS_PER_MIX = 4

summary = []

def monitor_mix(mix_name, proportions, runs=4, use_api=True):
    gen = ParamGenerator()
    total = sum(proportions.values())
    norm = {k: v/total for k,v in proportions.items()}

    if use_api and gen.client:
        params = gen.get_params_from_llm(norm, temperature=0.25)
    else:
        # heuristic fallback
        lr = 0.1 + 0.4 * norm.get('risk_taker', 0)
        eps = 0.15 + 0.4 * (1 - norm.get('cautious', 0))
        pers = 0.05 + 0.4 * norm.get('cautious', 0)
        patience = int(10 + 30 * norm.get('cautious', 0))
        params = AgentParams(lr, eps, pers, 0.05, patience, rationale='heuristic fallback')

    print(f"\n=== Mix: {mix_name} params: {params} ===")

    for run_idx in range(1, runs+1):
        sim = PRLTSimulator(params, rng_seed=int(time.time()*1000) % (2**32))
        pre_bar = None
        post_bar = None
        pre_done = False
        post_done = False
        pre_trial_target = 200
        post_trial_target = 1000

        if TQDM_AVAILABLE:
            pre_bar = tqdm(total=pre_trial_target, desc=f'{mix_name} run{run_idx} pre', unit='tr')
            post_bar = tqdm(total=post_trial_target, desc=f'{mix_name} run{run_idx} post', unit='tr')

        last_status = None
        attempts = 0

        def cb(status):
            nonlocal pre_done, post_done, last_status, attempts
            last_status = status
            if status['phase'] == 'pre' and not status['pre_converged']:
                attempts = status['t']
                if TQDM_AVAILABLE and pre_bar:
                    pre_bar.n = status['t']
                    pre_bar.refresh()
                else:
                    print(f"[pre] t={status['t']} choice={status['choice']} reward={status['reward']} QA={status['QA']:.2f} QB={status['QB']:.2f}", end='\r')
            elif status['phase'] == 'pre' and status['pre_converged']:
                pre_done = True
                attempts = status['t']
                if pre_bar:
                    pre_bar.n = status['t']
                    pre_bar.refresh()
                print(f"\n[pre] converged at t={status['t']}")
            elif status['phase'] == 'post' and not status['post_converged']:
                attempts = status['t']
                if TQDM_AVAILABLE and post_bar:
                    post_bar.n = status['t']
                    post_bar.refresh()
                else:
                    print(f"[post] t={status['t']} choice={status['choice']} reward={status['reward']} QA={status['QA']:.2f} QB={status['QB']:.2f}", end='\r')
            elif status['phase'] == 'post' and status['post_converged']:
                post_done = True
                attempts = status['t']
                if post_bar:
                    post_bar.n = status['t']
                    post_bar.refresh()
                print(f"\n[post] switched at t={status['t']}")

        print(f"Running {mix_name} run {run_idx}...")
        start = time.time()
        res, hist = sim.run_with_callback(progress_cb=cb, pre_reversal_trials=200, max_post=1000)
        end = time.time()

        if TQDM_AVAILABLE:
            if pre_bar:
                pre_bar.close()
            if post_bar:
                post_bar.close()

        record = {
            'mix': mix_name,
            'run': run_idx,
            'pre_conv': res.pre_rev_trials_to_converge,
            'post_switch': res.post_rev_trials_to_switch,
            'duration_s': round(end-start, 2),
            'params': res.params
        }
        summary.append(record)
        print(f"Result run {run_idx}: pre_conv={record['pre_conv']} post_switch={record['post_switch']} duration={record['duration_s']}s")

    # save intermediate summary
    with open(OUTPUT_DIR / f'{mix_name}_summary.json','w',encoding='utf-8') as f:
        json.dump([r for r in summary if r['mix']==mix_name], f, indent=2)


if __name__ == '__main__':
    for name, p in mixes.items():
        monitor_mix(name, p, runs=RUNS_PER_MIX, use_api=USE_API)
    # final save
    with open(OUTPUT_DIR / 'all_progress_summary.json','w',encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print('\nAll mixes complete. Summaries saved to', OUTPUT_DIR)
