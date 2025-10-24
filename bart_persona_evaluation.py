"""
BART Persona Evaluation - Integrated with existing system
Uses real persona files and LLM for more realistic evaluation
"""

import json
import random
import time
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import threading
import os
import sys

# Add project root to path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import OpenAI for evaluation
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
    from openai import OpenAI
    HAS_OPENAI = True
except:
    HAS_OPENAI = False
    OpenAI = None

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

@dataclass
class PersonaTrialResult:
    """Results from a single trial with persona-based decisions"""
    trial_id: int
    pumps: int
    exploded: bool
    reward: float
    threshold: int
    risk_weight: float
    cautious_weight: float
    temperature: float
    seed: int
    decision_sequence: List[str]
    persona_responses: List[str]  # actual LLM responses
    
@dataclass
class PersonaEvaluationConfig:
    """Configuration for persona-based evaluation"""
    risk_weight: float
    cautious_weight: float
    temperature: float
    model: str = "gpt-4o-mini"
    max_tokens: int = 64

class BARTPersonaEnvironment:
    """BART environment that matches the GUI version"""
    
    def __init__(self, max_pumps: int = 32, step_reward: float = 0.05):
        self.max_pumps = max_pumps
        self.step_reward = step_reward
        self.reset()
        
    def reset(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
        self.threshold = random.randint(1, self.max_pumps)
        self.pumps = 0
        self.done = False
        self.exploded = False
        return self._get_obs()
        
    def _get_obs(self):
        return {'pumps': self.pumps, 'exploded': int(self.exploded)}
        
    def step(self, action: str) -> Tuple[Dict, float, bool]:
        """Take action PUMP/COLLECT and return (obs, reward, done)"""
        if self.done:
            raise ValueError("Environment done, call reset()")
            
        if action == "PUMP":
            self.pumps += 1
            if self.pumps >= self.threshold:
                self.exploded = True
                self.done = True
                reward = 0.0
            else:  
                reward = self.step_reward
        else:  # COLLECT
            self.done = True
            reward = self.pumps * self.step_reward
            
        return self._get_obs(), reward, self.done

class PersonaEvaluator:
    """Evaluates BART performance with real persona files and LLM"""
    
    def __init__(self, output_dir: str = "bart_persona_results"):
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
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
                
        # Aggregation prompt template
        self.agg_prompt = """You are an Aggregation Agent for a balloon risk task.

Current state: Balloon has been pumped {pumps} times, earned ${earned:.2f} so far.

Advisor suggestions (weighted by importance):
- Risk-taker ({risk_pct}%): "{risk_advice}"
- Cautious-thinker ({cautious_pct}%): "{cautious_advice}"

Choose the single action that best reflects the highest-weighted advice.
Output EXACTLY one word: PUMP or COLLECT"""

        # Human baseline data
        self.human_baseline = {
            'avg_pumps': 16.5,
            'explosion_rate': 0.37,
            'avg_adjusted_pumps': 20.2
        }
        
    def _load_persona(self, filename: str) -> str:
        """Load persona text file"""
        filepath = _ROOT / filename
        if filepath.exists():
            return filepath.read_text(encoding='utf-8')
        else:
            print(f"Warning: Persona file {filename} not found")
            return f"You are a {filename.replace('.txt', '').replace('_', ' ')} persona."
            
    def _query_persona(self, persona_text: str, state_text: str, temperature: float, model: str) -> str:
        """Query a persona for decision"""
        if not self.client:
            # Fallback to simulated responses
            if "risk" in persona_text.lower():
                return "PUMP" if random.random() > 0.3 else "COLLECT"
            else:
                return "PUMP" if random.random() > 0.7 else "COLLECT"
                
        try:
            messages = [
                {"role": "system", "content": persona_text},
                {"role": "user", "content": state_text + " Reply with PUMP or COLLECT only."}
            ]
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=32
            )
            
            content = response.choices[0].message.content.strip().upper()
            return "PUMP" if "PUMP" in content else "COLLECT"
            
        except Exception as e:
            print(f"LLM query failed: {e}")
            # Fallback decision
            return "COLLECT"
            
    def run_single_evaluation(self, config: PersonaEvaluationConfig, trial_id: int, seed: int) -> PersonaTrialResult:
        """Run single trial with persona-based decisions"""
        
        env = BARTPersonaEnvironment()
        state = env.reset(seed)
        
        decision_sequence = []
        persona_responses = []
        total_reward = 0
        
        while not env.done:
            # Create state description
            state_text = f"Balloon {trial_id + 1}, pumps={env.pumps}, earned=${env.pumps * env.step_reward:.2f}"
            
            # Get individual persona advice
            risk_response = self._query_persona(self.risk_taker_persona, state_text, config.temperature, config.model)
            cautious_response = self._query_persona(self.cautious_persona, state_text, config.temperature, config.model)
            
            persona_responses.append(f"Risk:{risk_response}, Cautious:{cautious_response}")
            
            # Aggregate decision based on weights
            agg_prompt = self.agg_prompt.format(
                pumps=env.pumps,
                earned=env.pumps * env.step_reward,
                risk_pct=int(config.risk_weight * 100),
                cautious_pct=int(config.cautious_weight * 100),
                risk_advice=risk_response,
                cautious_advice=cautious_response
            )
            
            # Get aggregated decision
            if self.client:
                try:
                    agg_response = self.client.chat.completions.create(
                        model=config.model,
                        messages=[{"role": "user", "content": agg_prompt}],
                        temperature=config.temperature * 0.5,  # Less randomness for aggregation
                        max_tokens=16
                    )
                    final_action = agg_response.choices[0].message.content.strip().upper()
                    final_action = "PUMP" if "PUMP" in final_action else "COLLECT"
                except:
                    # Weighted fallback
                    if config.risk_weight > config.cautious_weight:
                        final_action = risk_response
                    else:
                        final_action = cautious_response
            else:
                # Weighted fallback  
                if config.risk_weight > config.cautious_weight:
                    final_action = risk_response
                else:
                    final_action = cautious_response
                    
            decision_sequence.append(final_action)
            
            # Take action in environment
            state, reward, done = env.step(final_action)
            total_reward += reward
            
        return PersonaTrialResult(
            trial_id=trial_id,
            pumps=env.pumps,
            exploded=env.exploded,
            reward=total_reward,
            threshold=env.threshold,
            risk_weight=config.risk_weight,
            cautious_weight=config.cautious_weight,
            temperature=config.temperature,
            seed=seed,
            decision_sequence=decision_sequence,
            persona_responses=persona_responses
        )
        
    def run_grid_search(self, 
                       risk_weights: List[float] = None,
                       cautious_weights: List[float] = None,
                       temperatures: List[float] = None,
                       n_trials: int = 20,
                       n_seeds: int = 2,
                       model: str = "gpt-4o-mini") -> List[Dict]:
        """Run grid search evaluation with persona-based decisions"""
        
        if not self.client:
            print("Warning: No OpenAI client available. Using simulated responses.")
            
        # Default parameter ranges
        if risk_weights is None:
            risk_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
        if cautious_weights is None:
            cautious_weights = [0.9, 0.7, 0.5, 0.3, 0.1]  # Inverse of risk weights
        if temperatures is None:
            temperatures = [0.5, 0.8, 1.0]
            
        results = []
        total_configs = len(risk_weights) * len(temperatures)
        config_idx = 0
        
        print(f"Running persona-based evaluation with {total_configs} configurations...")
        print(f"Each config: {n_trials} trials × {n_seeds} seeds")
        
        for risk_w, temp in itertools.product(risk_weights, temperatures):
            config_idx += 1
            # Cautious weight is complement of risk weight
            cautious_w = 1.0 - risk_w
            
            config = PersonaEvaluationConfig(
                risk_weight=risk_w,
                cautious_weight=cautious_w,
                temperature=temp,
                model=model
            )
            
            print(f"Config {config_idx}/{total_configs}: Risk={risk_w:.1f}, Cautious={cautious_w:.1f}, Temp={temp:.1f}")
            
            all_trials = []
            
            # Run multiple seeds
            for seed_offset in range(n_seeds):
                base_seed = hash(f"{risk_w}_{cautious_w}_{temp}_{seed_offset}") % 2**32
                
                for trial_id in range(n_trials):
                    trial_seed = base_seed + trial_id
                    trial_result = self.run_single_evaluation(config, trial_id, trial_seed)
                    all_trials.append(trial_result)
                    
            # Calculate metrics
            metrics = self._calculate_metrics(all_trials)
            
            result_dict = {
                'config': asdict(config),
                'trials': [asdict(t) for t in all_trials],
                **metrics
            }
            results.append(result_dict)
            
        return results
        
    def _calculate_metrics(self, trials: List[PersonaTrialResult]) -> Dict[str, float]:
        """Calculate performance metrics"""
        total_reward = sum(t.reward for t in trials)
        explosions = sum(1 for t in trials if t.exploded)
        explosion_rate = explosions / len(trials)
        
        avg_pumps = np.mean([t.pumps for t in trials])
        
        # Adjusted pumps (successful trials only)
        successful = [t for t in trials if not t.exploded]
        avg_adjusted_pumps = np.mean([t.pumps for t in successful]) if successful else 0
        
        # Consistency (lower variance = more consistent)
        pump_counts = [t.pumps for t in trials]
        consistency_score = 1.0 / (1.0 + np.var(pump_counts))
        
        # Optimal stopping (distance from threshold-1)
        optimal_distances = []
        for t in trials:
            if not t.exploded:
                optimal_distances.append(abs(t.pumps - (t.threshold - 1)))
            else:
                optimal_distances.append(t.pumps - t.threshold + 1)  # penalty
                
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
            filename = f"bart_persona_evaluation_{timestamp}.json"
            
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {filepath}")
        return filepath
        
    def create_visualizations(self, results: List[Dict]):
        """Create comprehensive visualizations"""
        
        # Convert to DataFrame
        data = []
        for result in results:
            row = result['config'].copy()
            row.update({k: v for k, v in result.items() if k not in ['config', 'trials']})
            data.append(row)
            
        df = pd.DataFrame(data)
        
        self._plot_risk_cautious_effects(df)
        self._plot_temperature_effects(df) 
        self._plot_human_comparison(df)
        self._plot_decision_patterns(results)
        
    def _plot_risk_cautious_effects(self, df: pd.DataFrame):
        """Plot effects of risk vs cautious weighting"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Risk vs Cautious Personality Effects on BART Performance', fontsize=16)
        
        # Average pumps vs risk weight
        ax = axes[0, 0]
        for temp in sorted(df['temperature'].unique()):
            temp_data = df[df['temperature'] == temp]
            ax.plot(temp_data['risk_weight'], temp_data['avg_pumps'], 
                   'o-', label=f'Temp={temp}', alpha=0.8)
        ax.set_xlabel('Risk Taker Weight')
        ax.set_ylabel('Average Pumps')
        ax.set_title('Risk Weight vs Average Pumps')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Explosion rate vs risk weight
        ax = axes[0, 1]
        for temp in sorted(df['temperature'].unique()):
            temp_data = df[df['temperature'] == temp]
            ax.plot(temp_data['risk_weight'], temp_data['explosion_rate'], 
                   's-', label=f'Temp={temp}', alpha=0.8)
        ax.set_xlabel('Risk Taker Weight')
        ax.set_ylabel('Explosion Rate')
        ax.set_title('Risk Weight vs Explosion Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Total reward vs risk weight
        ax = axes[1, 0]
        for temp in sorted(df['temperature'].unique()):
            temp_data = df[df['temperature'] == temp]
            ax.plot(temp_data['risk_weight'], temp_data['total_reward'], 
                   '^-', label=f'Temp={temp}', alpha=0.8)
        ax.set_xlabel('Risk Taker Weight')
        ax.set_ylabel('Total Reward')
        ax.set_title('Risk Weight vs Total Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Consistency vs risk weight
        ax = axes[1, 1]
        for temp in sorted(df['temperature'].unique()):
            temp_data = df[df['temperature'] == temp]
            ax.plot(temp_data['risk_weight'], temp_data['consistency_score'], 
                   'd-', label=f'Temp={temp}', alpha=0.8)
        ax.set_xlabel('Risk Taker Weight')
        ax.set_ylabel('Consistency Score') 
        ax.set_title('Risk Weight vs Consistency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risk_cautious_effects.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_temperature_effects(self, df: pd.DataFrame):
        """Plot temperature effects on decision quality"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Temperature Effects on BART Decision Quality', fontsize=16)
        
        # Temperature vs consistency
        ax = axes[0]
        temp_metrics = df.groupby('temperature').agg({
            'consistency_score': ['mean', 'std'],
            'optimal_stopping_score': ['mean', 'std']
        })
        
        temps = temp_metrics.index
        consistency_mean = temp_metrics[('consistency_score', 'mean')]
        consistency_std = temp_metrics[('consistency_score', 'std')]
        
        ax.errorbar(temps, consistency_mean, yerr=consistency_std, 
                   marker='o', capsize=5, label='Consistency Score')
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Consistency Score')
        ax.set_title('Temperature vs Decision Consistency')
        ax.grid(True, alpha=0.3)
        
        # Temperature vs optimal stopping
        ax = axes[1]
        optimal_mean = temp_metrics[('optimal_stopping_score', 'mean')]
        optimal_std = temp_metrics[('optimal_stopping_score', 'std')]
        
        ax.errorbar(temps, optimal_mean, yerr=optimal_std,
                   marker='s', capsize=5, label='Optimal Stopping', color='orange')
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Optimal Stopping Score')
        ax.set_title('Temperature vs Optimal Stopping')
        ax.grid(True, alpha=0.3)
        
        # Combined effect visualization
        ax = axes[2]
        scatter = ax.scatter(df['temperature'], df['consistency_score'], 
                           c=df['risk_weight'], s=60, alpha=0.7, cmap='RdYlBu')
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Consistency Score')
        ax.set_title('Temperature vs Consistency (colored by Risk Weight)')
        plt.colorbar(scatter, ax=ax, label='Risk Weight')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temperature_effects.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_human_comparison(self, df: pd.DataFrame):
        """Compare AI performance with human baselines"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('AI vs Human BART Performance', fontsize=16)
        
        metrics = ['avg_pumps', 'explosion_rate', 'avg_adjusted_pumps']
        human_values = [self.human_baseline[m] for m in metrics]
        titles = ['Average Pumps', 'Explosion Rate', 'Average Adjusted Pumps']
        
        for i, (metric, human_val, title) in enumerate(zip(metrics, human_values, titles)):
            ax = axes[i]
            
            # Plot AI distribution
            ax.hist(df[metric], bins=15, alpha=0.7, color='skyblue', edgecolor='black', label='AI Configurations')
            
            # Plot human baseline
            ax.axvline(human_val, color='red', linestyle='--', linewidth=3, label='Human Baseline')
            
            # Find closest AI configuration
            closest_idx = (df[metric] - human_val).abs().idxmin()
            closest_val = df.loc[closest_idx, metric]
            ax.axvline(closest_val, color='green', linestyle=':', linewidth=2, label='Closest AI Match')
            
            ax.set_xlabel(title)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{title} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'human_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print best human match
        self._find_best_human_match(df)
        
    def _find_best_human_match(self, df: pd.DataFrame):
        """Find configuration that best matches human performance"""
        
        # Calculate similarity to human baseline
        df = df.copy()
        similarity_scores = []
        
        for _, row in df.iterrows():
            score = 0
            for metric in ['avg_pumps', 'explosion_rate', 'avg_adjusted_pumps']:
                normalized_diff = abs(row[metric] - self.human_baseline[metric]) / self.human_baseline[metric]
                score += normalized_diff
            similarity_scores.append(score)
            
        df['human_similarity'] = similarity_scores
        best_match_idx = df['human_similarity'].idxmin()
        best_match = df.loc[best_match_idx]
        
        print("\n" + "="*60)
        print("BEST HUMAN-MATCHING CONFIGURATION")
        print("="*60)
        print(f"Risk Taker Weight: {best_match['risk_weight']:.2f}")
        print(f"Cautious Weight: {best_match['cautious_weight']:.2f}")
        print(f"Temperature: {best_match['temperature']:.2f}")
        print(f"Model: {best_match['model']}")
        print("\nPerformance Comparison:")
        print(f"Average Pumps:      AI={best_match['avg_pumps']:.2f} | Human={self.human_baseline['avg_pumps']:.2f}")
        print(f"Explosion Rate:     AI={best_match['explosion_rate']:.3f} | Human={self.human_baseline['explosion_rate']:.3f}")
        print(f"Avg Adjusted Pumps: AI={best_match['avg_adjusted_pumps']:.2f} | Human={self.human_baseline['avg_adjusted_pumps']:.2f}")
        print(f"Similarity Score: {best_match['human_similarity']:.3f} (lower is better)")
        
    def _plot_decision_patterns(self, results: List[Dict]):
        """Analyze decision patterns across configurations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Decision Pattern Analysis', fontsize=16)
        
        # Collect decision data
        pump_counts_by_turn = {}
        explosion_positions = []
        
        for result in results:
            config = result['config']
            key = f"R{config['risk_weight']:.1f}_T{config['temperature']:.1f}"
            
            for trial in result['trials']:
                # Count pumps by decision turn
                for turn, decision in enumerate(trial['decision_sequence']):
                    if turn not in pump_counts_by_turn:
                        pump_counts_by_turn[turn] = []
                    pump_counts_by_turn[turn].append(1 if decision == "PUMP" else 0)
                    
                # Track explosion positions
                if trial['exploded']:
                    explosion_positions.append(trial['pumps'])
                    
        # Plot pump probability by turn
        ax = axes[0, 0]
        turns = sorted(pump_counts_by_turn.keys())[:20]  # First 20 turns
        pump_probs = [np.mean(pump_counts_by_turn[turn]) for turn in turns]
        
        ax.plot(turns, pump_probs, 'bo-', alpha=0.7)
        ax.set_xlabel('Decision Turn')
        ax.set_ylabel('Probability of PUMP')
        ax.set_title('PUMP Probability by Decision Turn')
        ax.grid(True, alpha=0.3)
        
        # Plot explosion distribution
        ax = axes[0, 1]
        if explosion_positions:
            ax.hist(explosion_positions, bins=15, alpha=0.7, color='red', edgecolor='black')
            ax.set_xlabel('Pumps When Exploded')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Explosion Points')
            ax.grid(True, alpha=0.3)
        
        # Performance scatter plot
        ax = axes[1, 0]
        scatter_data = []
        for result in results:
            config = result['config']
            scatter_data.append([
                config['risk_weight'],
                result['explosion_rate'],
                result['total_reward']
            ])
            
        scatter_data = np.array(scatter_data)
        scatter = ax.scatter(scatter_data[:, 0], scatter_data[:, 1], 
                           c=scatter_data[:, 2], s=80, alpha=0.7, cmap='viridis')
        ax.set_xlabel('Risk Weight')
        ax.set_ylabel('Explosion Rate')
        ax.set_title('Risk vs Explosion (colored by Reward)')
        plt.colorbar(scatter, ax=ax, label='Total Reward')
        
        # Best configurations highlight
        ax = axes[1, 1]
        # Find Pareto front (low explosion, high reward)
        pareto_configs = []
        for result in results:
            config = result['config']
            is_pareto = True
            for other_result in results:
                if (other_result['explosion_rate'] <= result['explosion_rate'] and 
                    other_result['total_reward'] >= result['total_reward'] and
                    (other_result['explosion_rate'] < result['explosion_rate'] or 
                     other_result['total_reward'] > result['total_reward'])):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_configs.append(result)
                
        # Plot all configs
        for result in results:
            ax.scatter(result['explosion_rate'], result['total_reward'], 
                      c='lightblue', s=40, alpha=0.6)
                      
        # Highlight Pareto optimal
        for result in pareto_configs:
            ax.scatter(result['explosion_rate'], result['total_reward'],
                      c='red', s=100, marker='*', edgecolors='black', linewidth=1)
                      
        ax.set_xlabel('Explosion Rate')
        ax.set_ylabel('Total Reward')
        ax.set_title('Pareto Optimal Configurations (red stars)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'decision_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Run the persona-based BART evaluation"""
    
    print("Starting BART Persona Evaluation...")
    print("This uses real persona files and LLM for realistic decision-making")
    
    evaluator = PersonaEvaluator()
    
    # Check if OpenAI is available
    if not evaluator.client:
        print("\nWARNING: OpenAI client not available.")
        print("Set OPENAI_API_KEY environment variable for full evaluation.")
        print("Proceeding with simulated responses...\n")
    
    # Define evaluation parameters
    risk_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    temperatures = [0.5, 0.8, 1.0]
    
    # Run evaluation
    results = evaluator.run_grid_search(
        risk_weights=risk_weights,
        temperatures=temperatures,
        n_trials=15,  # Fewer trials per config since using LLM
        n_seeds=2,
        model="gpt-4o-mini"
    )
    
    # Save results
    filepath = evaluator.save_results(results)
    
    # Create visualizations
    evaluator.create_visualizations(results)
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {filepath}")
    print(f"Visualizations saved to: {evaluator.output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total configurations tested: {len(results)}")
    
    best_reward = max(results, key=lambda x: x['total_reward'])
    best_consistency = max(results, key=lambda x: x['consistency_score'])
    lowest_explosion = min(results, key=lambda x: x['explosion_rate'])
    
    print(f"\nBest Total Reward: {best_reward['total_reward']:.2f}")
    print(f"  Config: Risk={best_reward['config']['risk_weight']:.1f}, Temp={best_reward['config']['temperature']:.1f}")
    
    print(f"\nMost Consistent: {best_consistency['consistency_score']:.3f}")
    print(f"  Config: Risk={best_consistency['config']['risk_weight']:.1f}, Temp={best_consistency['config']['temperature']:.1f}")
    
    print(f"\nLowest Explosion Rate: {lowest_explosion['explosion_rate']:.3f}")
    print(f"  Config: Risk={lowest_explosion['config']['risk_weight']:.1f}, Temp={lowest_explosion['config']['temperature']:.1f}")

if __name__ == "__main__":
    main()