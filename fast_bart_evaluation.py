"""
Fast BART Persona Evaluation
Optimized version with fewer API calls and batched decisions
"""

import json
import random
import time
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Add project root to path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
    from openai import OpenAI
    HAS_OPENAI = True
except:
    HAS_OPENAI = False
    OpenAI = None

@dataclass
class FastTrialResult:
    """Results from a single trial"""
    trial_id: int
    pumps: int
    exploded: bool
    reward: float
    threshold: int
    risk_weight: float
    cautious_weight: float
    temperature: float
    decision_pattern: str  # Simplified pattern instead of full sequence

class FastBARTEnvironment:
    """Fast BART environment"""
    
    def __init__(self, max_pumps: int = 32, step_reward: float = 0.05):
        self.max_pumps = max_pumps
        self.step_reward = step_reward
        
    def reset(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
        self.threshold = random.randint(1, self.max_pumps)
        self.pumps = 0
        self.done = False
        self.exploded = False
        
    def step(self, action: str) -> Tuple[float, bool]:
        """Take action and return (reward, done)"""
        if action == "PUMP":
            self.pumps += 1
            if self.pumps >= self.threshold:
                self.exploded = True
                self.done = True
                return 0.0, True
            else:
                return self.step_reward, False
        else:  # COLLECT
            self.done = True
            return self.pumps * self.step_reward, True

class FastPersonaEvaluator:
    """Fast persona-based evaluator with reduced API calls"""
    
    def __init__(self, output_dir: str = "fast_bart_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load persona files
        self.risk_taker_persona = self._load_persona("risk_taker.txt")
        self.cautious_persona = self._load_persona("cautious_thinker.txt")
        
        # Initialize OpenAI client
        self.client = None
        if HAS_OPENAI:
            try:
                self.client = OpenAI()
                print("✓ OpenAI client initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
                
        # Human baseline data
        self.human_baseline = {
            'avg_pumps': 16.5,
            'explosion_rate': 0.37,
            'avg_adjusted_pumps': 20.2
        }
        
        # Cache for persona responses to reduce API calls
        self.response_cache = {}
        
    def _load_persona(self, filename: str) -> str:
        """Load persona text file"""
        filepath = _ROOT / filename
        if filepath.exists():
            return filepath.read_text(encoding='utf-8')
        else:
            print(f"Warning: Persona file {filename} not found")
            return f"You are a {filename.replace('.txt', '').replace('_', ' ')} persona."
            
    def _get_persona_strategy(self, persona_text: str, risk_weight: float, temperature: float) -> Dict[str, float]:
        """Get overall strategy from persona instead of decision-by-decision queries"""
        
        cache_key = f"{hash(persona_text)}_{risk_weight}_{temperature}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
            
        if not self.client:
            # Fallback strategy based on persona type
            if "risk" in persona_text.lower():
                strategy = {
                    'base_pump_prob': 0.8,
                    'pump_decay_rate': 0.03,
                    'risk_threshold': 0.2
                }
            else:
                strategy = {
                    'base_pump_prob': 0.5,
                    'pump_decay_rate': 0.08,
                    'risk_threshold': 0.7
                }
        else:
            # Get strategy from LLM
            strategy_prompt = f"""Given this persona:
{persona_text}

For a balloon risk task where you pump balloons to earn money but risk explosion:

What's your general strategy? Respond with ONLY these three numbers (0.0 to 1.0):
1. Initial willingness to pump (0=never, 1=always)
2. How quickly you become more cautious as pumps increase (0=never change, 1=quickly cautious)  
3. Risk threshold for stopping (0=stop early, 1=continue despite risk)

Example: 0.7 0.05 0.3

Temperature factor: {temperature} (higher = more random decisions)"""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": strategy_prompt}],
                    temperature=temperature * 0.5,
                    max_tokens=50
                )
                
                content = response.choices[0].message.content.strip()
                # Parse the three numbers
                numbers = [float(x) for x in content.split()[:3]]
                
                strategy = {
                    'base_pump_prob': max(0.1, min(0.95, numbers[0])),
                    'pump_decay_rate': max(0.01, min(0.15, numbers[1])),
                    'risk_threshold': max(0.1, min(0.9, numbers[2]))
                }
                
            except Exception as e:
                print(f"LLM strategy query failed: {e}")
                # Fallback
                if "risk" in persona_text.lower():
                    strategy = {'base_pump_prob': 0.8, 'pump_decay_rate': 0.03, 'risk_threshold': 0.2}
                else:
                    strategy = {'base_pump_prob': 0.5, 'pump_decay_rate': 0.08, 'risk_threshold': 0.7}
        
        self.response_cache[cache_key] = strategy
        return strategy
        
    def _make_decision(self, risk_strategy: Dict, cautious_strategy: Dict, 
                      risk_weight: float, cautious_weight: float, 
                      pumps: int, temperature: float) -> str:
        """Make decision based on blended strategies"""
        
        # Calculate pump probability for each strategy
        risk_prob = max(0.05, risk_strategy['base_pump_prob'] - pumps * risk_strategy['pump_decay_rate'])
        cautious_prob = max(0.02, cautious_strategy['base_pump_prob'] - pumps * cautious_strategy['pump_decay_rate'])
        
        # Weighted combination
        combined_prob = risk_weight * risk_prob + cautious_weight * cautious_prob
        
        # Apply temperature
        if temperature > 0:
            noise = np.random.normal(0, temperature * 0.1)
            combined_prob = max(0.01, min(0.99, combined_prob + noise))
            
        return "PUMP" if random.random() < combined_prob else "COLLECT"
        
    def run_single_trial(self, risk_weight: float, cautious_weight: float, 
                        temperature: float, trial_id: int, seed: int) -> FastTrialResult:
        """Run single trial with strategy-based decisions"""
        
        env = FastBARTEnvironment()
        env.reset(seed)
        
        # Get strategies for both personas (cached to reduce API calls)
        risk_strategy = self._get_persona_strategy(self.risk_taker_persona, risk_weight, temperature)
        cautious_strategy = self._get_persona_strategy(self.cautious_persona, cautious_weight, temperature)
        
        total_reward = 0
        decisions = []
        
        while not env.done:
            action = self._make_decision(risk_strategy, cautious_strategy, 
                                       risk_weight, cautious_weight, 
                                       env.pumps, temperature)
            decisions.append(action)
            reward, done = env.step(action)
            total_reward += reward
            
        # Create simplified decision pattern
        decision_pattern = f"{'P' if 'PUMP' in decisions[:3] else 'C'}{'P' if len([d for d in decisions if d == 'PUMP']) > len(decisions)/2 else 'C'}{len(decisions)}"
        
        return FastTrialResult(
            trial_id=trial_id,
            pumps=env.pumps,
            exploded=env.exploded,
            reward=total_reward,
            threshold=env.threshold,
            risk_weight=risk_weight,
            cautious_weight=cautious_weight,
            temperature=temperature,
            decision_pattern=decision_pattern
        )
        
    def run_fast_evaluation(self, n_trials: int = 10, n_seeds: int = 2) -> List[Dict]:
        """Run fast evaluation with fewer configurations but real LLM strategies"""
        
        if not self.client:
            print("Warning: No OpenAI client. Using fallback strategies.")
            
        # Reduced parameter space for speed
        risk_weights = [0.2, 0.4, 0.6, 0.8]  # 4 levels instead of 9
        temperatures = [0.5, 1.0]  # 2 levels instead of 5
        
        results = []
        total_configs = len(risk_weights) * len(temperatures)
        
        print(f"Running fast LLM-based evaluation: {total_configs} configurations")
        print(f"Each config: {n_trials} trials × {n_seeds} seeds = {n_trials * n_seeds} total trials")
        
        # Pre-load strategies to minimize API calls
        print("Pre-loading persona strategies...")
        for risk_w in risk_weights:
            for temp in temperatures:
                self._get_persona_strategy(self.risk_taker_persona, risk_w, temp)
                self._get_persona_strategy(self.cautious_persona, 1-risk_w, temp)
        
        config_idx = 0
        for risk_w, temp in itertools.product(risk_weights, temperatures):
            config_idx += 1
            cautious_w = 1.0 - risk_w
            
            print(f"Config {config_idx}/{total_configs}: Risk={risk_w:.1f}, Cautious={cautious_w:.1f}, Temp={temp:.1f}")
            
            all_trials = []
            
            for seed_offset in range(n_seeds):
                base_seed = hash(f"{risk_w}_{temp}_{seed_offset}") % 2**32
                
                for trial_id in range(n_trials):
                    trial_seed = base_seed + trial_id
                    result = self.run_single_trial(risk_w, cautious_w, temp, trial_id, trial_seed)
                    all_trials.append(result)
                    
            # Calculate metrics
            metrics = self._calculate_metrics(all_trials)
            
            result_dict = {
                'config': {
                    'risk_weight': risk_w,
                    'cautious_weight': cautious_w,
                    'temperature': temp
                },
                'trials': [asdict(t) for t in all_trials],
                **metrics
            }
            results.append(result_dict)
            
        return results
        
    def _calculate_metrics(self, trials: List[FastTrialResult]) -> Dict[str, float]:
        """Calculate performance metrics"""
        
        total_reward = sum(t.reward for t in trials)
        explosions = sum(1 for t in trials if t.exploded)
        explosion_rate = explosions / len(trials)
        
        avg_pumps = np.mean([t.pumps for t in trials])
        
        # Adjusted pumps (successful trials only)
        successful = [t for t in trials if not t.exploded]
        avg_adjusted_pumps = np.mean([t.pumps for t in successful]) if successful else 0
        
        # Consistency score
        pump_variance = np.var([t.pumps for t in trials])
        consistency_score = 1.0 / (1.0 + pump_variance)
        
        # Optimal stopping score
        optimal_distances = []
        for t in trials:
            if not t.exploded:
                optimal_distances.append(abs(t.pumps - (t.threshold - 1)))
            else:
                optimal_distances.append(t.pumps - t.threshold + 2)
                
        optimal_stopping_score = 1.0 / (1.0 + np.mean(optimal_distances))
        
        return {
            'total_reward': total_reward,
            'explosion_rate': explosion_rate,
            'avg_pumps': avg_pumps,
            'avg_adjusted_pumps': avg_adjusted_pumps,
            'consistency_score': consistency_score,
            'optimal_stopping_score': optimal_stopping_score,
            'n_trials': len(trials)
        }
        
    def save_results(self, results: List[Dict], filename: str = None) -> Path:
        """Save results to JSON file"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"fast_bart_llm_results_{timestamp}.json"
            
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {filepath}")
        return filepath

def main():
    """Run fast LLM-based evaluation"""
    
    print("Fast BART LLM Evaluation")
    print("=" * 50)
    print("Optimized for speed with strategic API usage")
    
    evaluator = FastPersonaEvaluator()
    
    if not evaluator.client:
        print("\nWARNING: OpenAI client not available.")
        print("Using fallback strategies instead of real LLM queries.")
        print("Set OPENAI_API_KEY for full LLM evaluation.\n")
    
    start_time = time.time()
    
    # Run fast evaluation - fewer configs but real LLM strategies
    results = evaluator.run_fast_evaluation(n_trials=15, n_seeds=2)
    
    end_time = time.time()
    print(f"\nEvaluation completed in {end_time - start_time:.1f} seconds")
    
    # Save results
    filepath = evaluator.save_results(results)
    
    # Quick analysis
    print(f"\n" + "=" * 60)
    print("FAST LLM EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"Configurations tested: {len(results)}")
    print(f"Total trials: {sum(r['n_trials'] for r in results)}")
    
    # Performance stats
    rewards = [r['total_reward'] for r in results]
    explosions = [r['explosion_rate'] for r in results]
    pumps = [r['avg_pumps'] for r in results]
    
    print(f"\nPerformance Summary:")
    print(f"- Reward range: {min(rewards):.1f} - {max(rewards):.1f}")
    print(f"- Explosion rate: {min(explosions):.3f} - {max(explosions):.3f}")
    print(f"- Average pumps: {min(pumps):.1f} - {max(pumps):.1f}")
    
    # Best configurations
    best_reward_idx = np.argmax(rewards)
    best_reward_config = results[best_reward_idx]['config']
    print(f"\nBest reward config: Risk={best_reward_config['risk_weight']:.1f}, Temp={best_reward_config['temperature']:.1f}")
    
    best_safety_idx = np.argmin(explosions)
    best_safety_config = results[best_safety_idx]['config']
    print(f"Safest config: Risk={best_safety_config['risk_weight']:.1f}, Temp={best_safety_config['temperature']:.1f}")
    
    # Human comparison
    human_pumps = 16.5
    closest_to_human = min(results, key=lambda x: abs(x['avg_pumps'] - human_pumps))
    closest_config = closest_to_human['config']
    print(f"Closest to human: Risk={closest_config['risk_weight']:.1f}, Temp={closest_config['temperature']:.1f}")
    print(f"  (AI: {closest_to_human['avg_pumps']:.1f} pumps vs Human: {human_pumps} pumps)")
    
    print(f"\n✓ Fast LLM evaluation complete!")
    print(f"✓ Results saved to: {filepath}")
    print(f"✓ Run 'python bart_analysis.py' to create visualizations")

if __name__ == "__main__":
    main()