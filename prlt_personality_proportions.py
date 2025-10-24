"""
Probabilistic Reversal Learning Task (PRLT) with Personality Proportions
- Two options (A & B) with initial reward probabilities 0.75 / 0.25
- Agent learns via trial-and-error and converges to the better option
- After a stable period, probabilities reverse (A=0.25, B=0.75) without telling the agent
- Measure Trials-to-converge pre-reversal and Trials-to-switch post-reversal

Design choices:
- Use OpenAI API once per personality mixture to get a compact parameterized decision policy
  (learning rate, exploration epsilon, perseveration bias, decision_threshold)
- Use an internal Q-learning + epsilon-greedy simulator that uses those parameters to run hundreds
  of trials quickly and deterministically (no per-trial API calls to avoid rate/cost blowup).
- Optional "api_online" mode: make an API call each trial to ask the persona which option to pick
  (very slow and expensive). Default is parameterized mode.

Output: JSON/CSV results and a matplotlib figure summarizing switch times.

Usage:
    python prlt_personality_proportions.py

Environment:
- Requires OPENAI_API_KEY for parameter extraction mode
- Uses personas present in repository (e.g., 'risk_taker.txt', 'cautious_thinker.txt', 'easilytired_agent copy.txt', 'unmotivated_agent.txt')

"""

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add repo root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False
    OpenAI = None

# -------------------- Data classes --------------------
@dataclass
class AgentParams:
    learning_rate: float  # alpha
    epsilon: float        # exploration probability
    perseveration: float  # tendency to repeat previous choice (0-1)
    decision_noise: float # additional softmax noise (unused if epsilon)
    patience: int         # number of trials to require stable performance for 'convergence'
    rationale: str = ""

@dataclass
class PRLTResult:
    persona_mix_name: str
    params: Dict
    pre_rev_trials_to_converge: Optional[int]
    post_rev_trials_to_switch: Optional[int]
    total_trials_run: int
    trial_history: List[Dict]

# -------------------- Utilities --------------------
PERSONA_FILES = {
    'risk_taker': ROOT / 'risk_taker.txt',
    'cautious': ROOT / 'cautious_thinker.txt',
    'easily_tired': ROOT / 'easilytired_agent copy.txt',
    'unmotivated': ROOT / 'unmotivated_agent.txt'
}

DEFAULT_PARAMS = AgentParams(
    learning_rate=0.2,
    epsilon=0.1,
    perseveration=0.1,
    decision_noise=0.05,
    patience=20,
    rationale='default fallback params'
)

# -------------------- LLM param generation --------------------
class ParamGenerator:
    def __init__(self):
        self.client = None
        if HAS_OPENAI:
            try:
                self.client = OpenAI()
            except Exception as e:
                print('OpenAI client init failed:', e)
                self.client = None

    def build_persona_blend_text(self, proportions: Dict[str, float]) -> str:
        lines = []
        lines.append('You are a simulated decision-maker whose personality is a weighted blend of the following personas:')
        for key, w in proportions.items():
            persona_path = PERSONA_FILES.get(key)
            txt = ''
            if persona_path and persona_path.exists():
                txt = persona_path.read_text(encoding='utf-8').strip().splitlines()[0:10]
                txt = ' '.join(txt)
            lines.append(f'- {key} (weight {w:.2f}): {txt}')
        lines.append('\nProvide compact reinforcement-learning style decision parameters (JSON) tuned to this personality blend for a 2-option probabilistic learning task.\n')
        lines.append('Return only a JSON object with numeric fields: learning_rate (0-1), epsilon (0-1), perseveration (0-1), decision_noise (0-1), patience (int 5-50). Also include a short rationale string field `rationale`.')
        return '\n'.join(lines)

    def get_params_from_llm(self, proportions: Dict[str, float], temperature: float = 0.3) -> AgentParams:
        # If no OpenAI client, return default
        if not self.client:
            print('No OpenAI client available, using default params')
            return DEFAULT_PARAMS

        prompt = self.build_persona_blend_text(proportions)
        try:
            resp = self.client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=200,
                temperature=temperature,
                timeout=15
            )
            content = resp.choices[0].message.content.strip()
            # Try to extract JSON from content
            import re
            m = re.search(r'\{.*\}', content, re.S)
            json_text = m.group(0) if m else content
            parsed = json.loads(json_text)
            # Coerce / sanitize
            lr = float(parsed.get('learning_rate', DEFAULT_PARAMS.learning_rate))
            eps = float(parsed.get('epsilon', DEFAULT_PARAMS.epsilon))
            pers = float(parsed.get('perseveration', DEFAULT_PARAMS.perseveration))
            noise = float(parsed.get('decision_noise', DEFAULT_PARAMS.decision_noise))
            patience = int(parsed.get('patience', DEFAULT_PARAMS.patience))
            rationale = parsed.get('rationale', '')
            return AgentParams(lr, eps, pers, noise, patience, rationale)
        except Exception as e:
            print('LLM param extraction failed:', e)
            return DEFAULT_PARAMS

# -------------------- PRLT simulation --------------------
class PRLTSimulator:
    def __init__(self, params: AgentParams, rng_seed: Optional[int] = None):
        self.params = params
        self.alpha = params.learning_rate
        self.epsilon = params.epsilon
        self.perseveration = params.perseveration
        self.noise = params.decision_noise
        self.patience = params.patience
        self.rng = random.Random(rng_seed)

    def run(self, pre_reversal_trials: int = 200, reversal_after_stable: bool = True) -> Tuple[PRLTResult, List[Dict]]:
        # Task probabilities
        pA = 0.75
        pB = 0.25

        # Q-values initial
        QA, QB = 0.5, 0.5
        prev_choice = None

        trial_history = []

        # Helper to check convergence: moving window fraction selecting correct action
        def check_convergence(history, correct_option, window=20, threshold=0.9):
            if len(history) < window:
                return False
            recent = history[-window:]
            picks = sum(1 for t in recent if t['choice'] == correct_option)
            return (picks / window) >= threshold

        # Run pre-reversal until converge or max
        pre_converge_trial = None
        max_pre = pre_reversal_trials
        for t in range(1, max_pre + 1):
            choice = self.choose(QA, QB, prev_choice)
            reward = 1 if self.rng.random() < (pA if choice == 'A' else pB) else 0
            QA, QB = self.update(QA, QB, choice, reward)
            prev_choice = choice
            trial_history.append({'phase':'pre', 'trial': t, 'choice': choice, 'reward': reward, 'QA': QA, 'QB': QB})
            # Check convergence onto A (the correct option)
            if check_convergence(trial_history, 'A', window=self.patience, threshold=0.9):
                pre_converge_trial = t
                break

        if pre_converge_trial is None:
            pre_converge_trial = max_pre

        # Reverse probabilities
        pA, pB = 0.25, 0.75

        # Continue until agent switches (converges onto B)
        post_converge_trial = None
        max_post = 1000
        for t2 in range(1, max_post + 1):
            t_index = pre_converge_trial + t2
            choice = self.choose(QA, QB, prev_choice)
            reward = 1 if self.rng.random() < (pA if choice == 'A' else pB) else 0
            QA, QB = self.update(QA, QB, choice, reward)
            prev_choice = choice
            trial_history.append({'phase':'post', 'trial': t_index, 'choice': choice, 'reward': reward, 'QA': QA, 'QB': QB})
            if check_convergence(trial_history, 'B', window=self.patience, threshold=0.9):
                post_converge_trial = t2
                break

        if post_converge_trial is None:
            post_converge_trial = max_post

        result = PRLTResult(
            persona_mix_name='',
            params=asdict(self.params),
            pre_rev_trials_to_converge=pre_converge_trial,
            post_rev_trials_to_switch=post_converge_trial,
            total_trials_run=len(trial_history),
            trial_history=trial_history
        )

        return result, trial_history

    def run_with_callback(self, progress_cb=None, pre_reversal_trials: int = 200, max_post: int = 1000) -> Tuple[PRLTResult, List[Dict]]:
        """Run the PRLT but call progress_cb after each trial with a status dict.

        progress_cb(status_dict) where status_dict includes:
          phase: 'pre'|'post'
          t: trial index within phase
          global_t: absolute trial index
          choice, reward, QA, QB
          pre_converged (bool), post_converged (bool)
        """
        # Task probabilities
        pA = 0.75
        pB = 0.25

        QA, QB = 0.5, 0.5
        prev_choice = None
        trial_history = []

        def check_convergence(history, correct_option, window=20, threshold=0.9):
            if len(history) < window:
                return False
            recent = history[-window:]
            picks = sum(1 for t in recent if t['choice'] == correct_option)
            return (picks / window) >= threshold

        pre_converge_trial = None
        max_pre = pre_reversal_trials
        global_t = 0
        # Pre-reversal
        for t in range(1, max_pre + 1):
            global_t += 1
            choice = self.choose(QA, QB, prev_choice)
            reward = 1 if self.rng.random() < (pA if choice == 'A' else pB) else 0
            QA, QB = self.update(QA, QB, choice, reward)
            prev_choice = choice
            trial_history.append({'phase':'pre', 'trial': t, 'choice': choice, 'reward': reward, 'QA': QA, 'QB': QB})
            pre_converged = check_convergence(trial_history, 'A', window=self.patience, threshold=0.9)

            status = {
                'phase': 'pre', 't': t, 'global_t': global_t,
                'choice': choice, 'reward': reward, 'QA': QA, 'QB': QB,
                'pre_converged': pre_converged, 'post_converged': False
            }
            if progress_cb:
                try:
                    progress_cb(status)
                except Exception:
                    pass

            if pre_converged:
                pre_converge_trial = t
                break

        if pre_converge_trial is None:
            pre_converge_trial = max_pre

        # Reverse
        pA, pB = 0.25, 0.75

        post_converge_trial = None
        # Post-reversal
        for t2 in range(1, max_post + 1):
            global_t += 1
            t_index = pre_converge_trial + t2
            choice = self.choose(QA, QB, prev_choice)
            reward = 1 if self.rng.random() < (pA if choice == 'A' else pB) else 0
            QA, QB = self.update(QA, QB, choice, reward)
            prev_choice = choice
            trial_history.append({'phase':'post', 'trial': t_index, 'choice': choice, 'reward': reward, 'QA': QA, 'QB': QB})
            post_converged = check_convergence(trial_history, 'B', window=self.patience, threshold=0.9)

            status = {
                'phase': 'post', 't': t2, 'global_t': global_t,
                'choice': choice, 'reward': reward, 'QA': QA, 'QB': QB,
                'pre_converged': True, 'post_converged': post_converged
            }
            if progress_cb:
                try:
                    progress_cb(status)
                except Exception:
                    pass

            if post_converged:
                post_converge_trial = t2
                break

        if post_converge_trial is None:
            post_converge_trial = max_post

        result = PRLTResult(
            persona_mix_name='',
            params=asdict(self.params),
            pre_rev_trials_to_converge=pre_converge_trial,
            post_rev_trials_to_switch=post_converge_trial,
            total_trials_run=len(trial_history),
            trial_history=trial_history
        )
        return result, trial_history

    def choose(self, QA, QB, prev_choice=None):
        # Epsilon-greedy plus perseveration
        if self.rng.random() < self.epsilon:
            return 'A' if self.rng.random() < 0.5 else 'B'
        # Add perseveration bias to action value
        vA = QA + (self.perseveration if prev_choice == 'A' else 0)
        vB = QB + (self.perseveration if prev_choice == 'B' else 0)
        if vA == vB:
            return 'A' if self.rng.random() < 0.5 else 'B'
        return 'A' if vA > vB else 'B'

    def update(self, QA, QB, choice, reward):
        if choice == 'A':
            QA = QA + self.alpha * (reward - QA)
        else:
            QB = QB + self.alpha * (reward - QB)
        return QA, QB

# -------------------- Runner / Experiment orchestration --------------------
def run_experiment(personality_mixtures: Dict[str, Dict[str, float]], runs_per_mix: int = 10, seed: int = 0, use_api: bool = True):
    gen = ParamGenerator()
    all_results: List[PRLTResult] = []

    for mix_name, proportions in personality_mixtures.items():
        print(f'\n=== Running mix: {mix_name} proportions={proportions} ===')
        # Normalize proportions
        total = sum(proportions.values())
        if total <= 0:
            raise ValueError('Proportions must sum to > 0')
        norm = {k: v/total for k,v in proportions.items()}

        # Get parameters (from LLM once per mixture unless use_api False)
        if use_api and gen.client:
            params = gen.get_params_from_llm(norm, temperature=0.25)
        else:
            # Heuristic mapping if no API
            lr = 0.1 + 0.4 * norm.get('risk_taker', 0)
            eps = 0.15 + 0.4 * (1 - norm.get('cautious', 0))
            pers = 0.05 + 0.4 * norm.get('cautious', 0)
            patience = int(10 + 30 * norm.get('cautious', 0))
            params = AgentParams(lr, eps, pers, 0.05, patience, rationale='heuristic fallback')

        mix_results = []
        for r in range(runs_per_mix):
            sim = PRLTSimulator(params, rng_seed=(seed + r))
            res, history = sim.run(pre_reversal_trials=200)
            res.persona_mix_name = mix_name
            all_results.append(res)
            mix_results.append(res)
            print(f'  run {r+1}/{runs_per_mix}: pre_conv={res.pre_rev_trials_to_converge}, post_switch={res.post_rev_trials_to_switch}')

    return all_results

# -------------------- Analysis & plotting --------------------
def analyze_and_plot(results: List[PRLTResult], outdir: Path):
    rows = []
    for r in results:
        rows.append({
            'mix': r.persona_mix_name,
            'learning_rate': r.params.get('learning_rate', None),
            'epsilon': r.params.get('epsilon', None),
            'perseveration': r.params.get('perseveration', None),
            'patience': r.params.get('patience', None),
            'pre_conv': r.pre_rev_trials_to_converge,
            'post_switch': r.post_rev_trials_to_switch,
            'total_trials': r.total_trials_run
        })
    df = pd.DataFrame(rows)
    summary = df.groupby('mix').agg({'pre_conv':['mean','std'],'post_switch':['mean','std'],'learning_rate':'mean','epsilon':'mean','perseveration':'mean'})
    print('\n=== Summary ===')
    print(summary)

    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / 'prlt_results_by_run.csv', index=False)
    summary.to_csv(outdir / 'prlt_summary_by_mix.csv')

    # Plotting
    # Use seaborn style if available; fall back to matplotlib default otherwise
    try:
        sns.set_style('whitegrid')
        sns.set_context('talk')
    except Exception:
        plt.style.use('default')

    fig, ax = plt.subplots(1,2, figsize=(14,6))
    sns.barplot(data=df, x='mix', y='pre_conv', ax=ax[0], ci='sd')
    ax[0].set_title('Trials to Converge (pre-reversal)')
    ax[0].set_ylabel('Trials')
    sns.barplot(data=df, x='mix', y='post_switch', ax=ax[1], ci='sd')
    ax[1].set_title('Trials to Switch (post-reversal)')
    ax[1].set_ylabel('Trials')
    plt.tight_layout()
    plt.savefig(outdir / 'prlt_switch_times.png', dpi=300)
    plt.show()

# -------------------- Example usage --------------------
def main():
    print('PRLT personality-proportion experiment')
    mixes = {
        'mostly_cautious': {'cautious':0.9, 'risk_taker':0.05, 'easily_tired':0.05},
        'mostly_risky': {'risk_taker':0.9, 'cautious':0.05, 'easily_tired':0.05},
        'impulsive_mix': {'risk_taker':0.5, 'cautious':0.1, 'easily_tired':0.2, 'unmotivated':0.2},
        'easily_tired': {'easily_tired':0.9, 'cautious':0.1}
    }

    start = time.time()
    results = run_experiment(mixes, runs_per_mix=8, seed=42, use_api=True)
    outdir = ROOT / 'prlt_results'
    analyze_and_plot(results, outdir)
    dur = time.time() - start
    print(f'Finished in {dur:.1f}s. Results saved to {outdir}')

if __name__ == '__main__':
    main()
