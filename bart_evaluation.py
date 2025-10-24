"""
BART Empirical Evaluation System
Grid search through personality quality proportions with comprehensive metrics and visualization.
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
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# Set up matplotlib for better plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

@dataclass
class TrialResult:
    """Results from a single BART trial"""
    trial_id: int
    pumps: int
    exploded: bool
    reward: float
    threshold: int
    risk_taker_weight: float
    cautious_weight: float
    temperature: float
    seed: int
    decisions: List[str]  # sequence of PUMP/COLLECT decisions
    
@dataclass 
class EvaluationResult:
    """Aggregated results from multiple trials"""
    config: Dict[str, Any]
    trials: List[TrialResult]
    total_reward: float
    explosion_rate: float
    avg_pumps: float
    avg_adjusted_pumps: float  # only non-exploded balloons
    consistency_score: float  # how consistent the decisions are
    optimal_stopping_score: float  # how close to optimal (threshold-1)
    
class BARTEnvironment:
    """Simplified BART environment for evaluation"""
    
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
        return self._get_state()
        
    def _get_state(self):
        return {
            'pumps': self.pumps,
            'exploded': self.exploded,
            'reward_so_far': self.pumps * self.step_reward
        }
        
    def step(self, action: str) -> Tuple[Dict, float, bool]:
        """Take action (PUMP or COLLECT) and return (state, reward, done)"""
        if self.done:
            raise ValueError("Environment is done, call reset()")
            
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
            
        return self._get_state(), reward, self.done

class PersonaSimulator:
    """Simulates persona decisions based on weights and temperature"""
    
    def __init__(self, risk_weight: float, cautious_weight: float, temperature: float = 0.8):
        self.risk_weight = risk_weight
        self.cautious_weight = cautious_weight 
        self.temperature = temperature
        
        # Normalize weights
        total = risk_weight + cautious_weight
        self.risk_weight = risk_weight / total
        self.cautious_weight = cautious_weight / total
        
    def decide(self, state: Dict, rng: random.Random) -> str:
        """Make PUMP/COLLECT decision based on persona weights"""
        pumps = state['pumps']
        
        # Risk-taker bias: higher probability of PUMP, especially early
        risk_pump_prob = max(0.1, 0.9 - pumps * 0.02)  # decreases slowly
        
        # Cautious bias: lower probability of PUMP, decreases quickly  
        cautious_pump_prob = max(0.05, 0.6 - pumps * 0.05)  # decreases fast
        
        # Weighted combination
        combined_prob = (self.risk_weight * risk_pump_prob + 
                        self.cautious_weight * cautious_pump_prob)
        
        # Apply temperature (higher temp = more randomness)
        if self.temperature > 0:
            # Convert to logits and apply temperature
            pump_logit = np.log(combined_prob / (1 - combined_prob))
            collect_logit = 0  # reference point
            
            pump_logit /= self.temperature
            collect_logit /= self.temperature
            
            # Softmax
            exp_pump = np.exp(pump_logit)
            exp_collect = np.exp(collect_logit) 
            pump_prob = exp_pump / (exp_pump + exp_collect)
        else:
            pump_prob = combined_prob
            
        return "PUMP" if rng.random() < pump_prob else "COLLECT"

class BARTEvaluator:
    """Main evaluation system for BART with grid search"""
    
    def __init__(self, output_dir: str = "bart_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Human baseline data (approximate values from literature)
        self.human_baseline = {
            'avg_pumps': 16.5,
            'explosion_rate': 0.37,
            'avg_adjusted_pumps': 20.2
        }
        
    def run_single_trial(self, config: Dict, trial_id: int, seed: int) -> TrialResult:
        """Run a single BART trial with given configuration"""
        env = BARTEnvironment()
        persona = PersonaSimulator(
            risk_weight=config['risk_weight'],
            cautious_weight=config['cautious_weight'], 
            temperature=config['temperature']
        )
        
        # Use trial-specific random number generator
        rng = random.Random(seed)
        
        state = env.reset(seed)
        decisions = []
        total_reward = 0
        
        while not env.done:
            action = persona.decide(state, rng)
            decisions.append(action)
            state, reward, done = env.step(action)
            total_reward += reward
            
        return TrialResult(
            trial_id=trial_id,
            pumps=env.pumps,
            exploded=env.exploded,
            reward=total_reward,
            threshold=env.threshold,
            risk_taker_weight=config['risk_weight'],
            cautious_weight=config['cautious_weight'],
            temperature=config['temperature'],
            seed=seed,
            decisions=decisions
        )
        
    def calculate_metrics(self, trials: List[TrialResult]) -> Dict[str, float]:
        """Calculate aggregate metrics from trials"""
        total_reward = sum(t.reward for t in trials) 
        explosions = sum(1 for t in trials if t.exploded)
        explosion_rate = explosions / len(trials)
        
        avg_pumps = np.mean([t.pumps for t in trials])
        
        # Adjusted pumps (only successful trials)
        successful_trials = [t for t in trials if not t.exploded]
        avg_adjusted_pumps = np.mean([t.pumps for t in successful_trials]) if successful_trials else 0
        
        # Consistency: variance in pumps (lower = more consistent)
        pump_counts = [t.pumps for t in trials]
        consistency_score = 1.0 / (1.0 + np.var(pump_counts))  # 0-1, higher = more consistent
        
        # Optimal stopping: how close to threshold-1 on average
        optimal_diffs = []
        for t in trials:
            if not t.exploded:
                optimal_diffs.append(abs(t.pumps - (t.threshold - 1)))
            else:
                optimal_diffs.append(t.pumps - t.threshold + 1)  # penalty for explosion
                
        optimal_stopping_score = 1.0 / (1.0 + np.mean(optimal_diffs))
        
        return {
            'total_reward': total_reward,
            'explosion_rate': explosion_rate,
            'avg_pumps': avg_pumps,
            'avg_adjusted_pumps': avg_adjusted_pumps,
            'consistency_score': consistency_score,
            'optimal_stopping_score': optimal_stopping_score
        }
        
    def run_evaluation(self, 
                      risk_weights: List[float] = None,
                      cautious_weights: List[float] = None, 
                      temperatures: List[float] = None,
                      n_trials: int = 50,
                      n_seeds: int = 3,
                      max_workers: int = 4) -> List[EvaluationResult]:
        """Run grid search evaluation"""
        
        # Default parameter ranges
        if risk_weights is None:
            risk_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
        if cautious_weights is None:
            cautious_weights = [0.1, 0.3, 0.5, 0.7, 0.9] 
        if temperatures is None:
            temperatures = [0.5, 0.8, 1.0]
            
        results = []
        total_configs = len(risk_weights) * len(cautious_weights) * len(temperatures)
        config_idx = 0
        
        print(f"Running evaluation with {total_configs} configurations...")
        print(f"Each config: {n_trials} trials × {n_seeds} seeds = {n_trials * n_seeds} total trials")
        
        for risk_w, cautious_w, temp in itertools.product(risk_weights, cautious_weights, temperatures):
            config_idx += 1
            config = {
                'risk_weight': risk_w,
                'cautious_weight': cautious_w,
                'temperature': temp
            }
            
            print(f"Config {config_idx}/{total_configs}: Risk={risk_w:.1f}, Cautious={cautious_w:.1f}, Temp={temp:.1f}")
            
            all_trials = []
            
            # Run multiple seeds for robustness
            for seed_offset in range(n_seeds):
                base_seed = hash(f"{risk_w}_{cautious_w}_{temp}_{seed_offset}") % 2**32
                
                # Run trials in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for trial_id in range(n_trials):
                        trial_seed = base_seed + trial_id
                        future = executor.submit(self.run_single_trial, config, trial_id, trial_seed)
                        futures.append(future)
                        
                    for future in as_completed(futures):
                        trial_result = future.result()
                        all_trials.append(trial_result)
                        
            # Calculate metrics
            metrics = self.calculate_metrics(all_trials)
            
            result = EvaluationResult(
                config=config,
                trials=all_trials,
                **metrics
            )
            results.append(result)
            
        return results
        
    def save_results(self, results: List[EvaluationResult], filename: str = None):
        """Save evaluation results to JSON"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"bart_evaluation_{timestamp}.json"
            
        filepath = self.output_dir / filename
        
        # Convert to serializable format
        data = []
        for result in results:
            result_dict = asdict(result)
            data.append(result_dict)
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Results saved to {filepath}")
        return filepath
        
    def load_results(self, filename: str) -> List[EvaluationResult]:
        """Load evaluation results from JSON"""
        filepath = self.output_dir / filename
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        results = []
        for item in data:
            # Convert trials back to TrialResult objects
            trials = [TrialResult(**trial) for trial in item['trials']]
            item['trials'] = trials
            results.append(EvaluationResult(**item))
            
        return results
        
    def create_visualizations(self, results: List[EvaluationResult]):
        """Create comprehensive visualizations of results"""
        
        # Convert results to DataFrame for easier plotting
        data = []
        for result in results:
            row = result.config.copy()
            row.update({
                'total_reward': result.total_reward,
                'explosion_rate': result.explosion_rate,
                'avg_pumps': result.avg_pumps,
                'avg_adjusted_pumps': result.avg_adjusted_pumps,
                'consistency_score': result.consistency_score,
                'optimal_stopping_score': result.optimal_stopping_score
            })
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Create visualizations
        self._plot_heatmaps(df)
        self._plot_parameter_effects(df)
        self._plot_human_comparison(df)
        self._plot_best_configurations(df)
        
    def _plot_heatmaps(self, df: pd.DataFrame):
        """Create heatmaps for different metrics"""
        
        # Fix temperature for heatmaps
        temp_values = sorted(df['temperature'].unique())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('BART Performance Heatmaps by Personality Weights', fontsize=16)
        
        metrics = ['avg_pumps', 'explosion_rate', 'total_reward', 
                  'consistency_score', 'optimal_stopping_score', 'avg_adjusted_pumps']
        
        for i, metric in enumerate(metrics):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            # Use middle temperature value for heatmap
            mid_temp = temp_values[len(temp_values)//2]
            subset = df[df['temperature'] == mid_temp]
            
            # Create pivot table
            pivot = subset.pivot(index='cautious_weight', 
                               columns='risk_weight', 
                               values=metric)
            
            sns.heatmap(pivot, annot=True, fmt='.3f', ax=ax, cmap='viridis')
            ax.set_title(f'{metric.replace("_", " ").title()} (temp={mid_temp})')
            ax.set_xlabel('Risk Taker Weight')
            ax.set_ylabel('Cautious Weight')
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_parameter_effects(self, df: pd.DataFrame):
        """Plot effects of individual parameters"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Parameter Effects on BART Performance', fontsize=16)
        
        # Risk weight effect (averaged over other parameters)
        ax = axes[0, 0]
        risk_effect = df.groupby('risk_weight').agg({
            'avg_pumps': 'mean',
            'explosion_rate': 'mean'
        }).reset_index()
        
        ax2 = ax.twinx()
        line1 = ax.plot(risk_effect['risk_weight'], risk_effect['avg_pumps'], 
                       'b-o', label='Avg Pumps')
        line2 = ax2.plot(risk_effect['risk_weight'], risk_effect['explosion_rate'], 
                        'r-s', label='Explosion Rate')
        
        ax.set_xlabel('Risk Taker Weight')
        ax.set_ylabel('Average Pumps', color='b')
        ax2.set_ylabel('Explosion Rate', color='r')
        ax.set_title('Risk Weight Effect')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='center right')
        
        # Cautious weight effect
        ax = axes[0, 1] 
        cautious_effect = df.groupby('cautious_weight').agg({
            'avg_pumps': 'mean',
            'explosion_rate': 'mean'
        }).reset_index()
        
        ax2 = ax.twinx()
        line1 = ax.plot(cautious_effect['cautious_weight'], cautious_effect['avg_pumps'], 
                       'b-o', label='Avg Pumps')
        line2 = ax2.plot(cautious_effect['cautious_weight'], cautious_effect['explosion_rate'], 
                        'r-s', label='Explosion Rate')
        
        ax.set_xlabel('Cautious Weight')
        ax.set_ylabel('Average Pumps', color='b')
        ax2.set_ylabel('Explosion Rate', color='r')
        ax.set_title('Cautious Weight Effect')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='center right')
        
        # Temperature effect
        ax = axes[1, 0]
        temp_effect = df.groupby('temperature').agg({
            'consistency_score': 'mean',
            'optimal_stopping_score': 'mean'
        }).reset_index()
        
        ax.plot(temp_effect['temperature'], temp_effect['consistency_score'], 
               'g-o', label='Consistency Score')
        ax.plot(temp_effect['temperature'], temp_effect['optimal_stopping_score'], 
               'm-s', label='Optimal Stopping Score')
        
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Score')
        ax.set_title('Temperature Effect on Decision Quality')
        ax.legend()
        
        # Combined performance metric
        ax = axes[1, 1]
        df['combined_score'] = (df['optimal_stopping_score'] * 0.4 + 
                               df['consistency_score'] * 0.3 + 
                               (1 - df['explosion_rate']) * 0.3)
        
        combined_effect = df.groupby(['risk_weight', 'cautious_weight']).agg({
            'combined_score': 'mean'
        }).reset_index()
        
        scatter = ax.scatter(combined_effect['risk_weight'], 
                           combined_effect['cautious_weight'],
                           c=combined_effect['combined_score'],
                           s=100, cmap='viridis')
        
        ax.set_xlabel('Risk Weight')
        ax.set_ylabel('Cautious Weight')
        ax.set_title('Combined Performance Score')
        plt.colorbar(scatter, ax=ax)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_effects.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_human_comparison(self, df: pd.DataFrame):
        """Compare best performing configurations with human data"""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('AI vs Human Performance Comparison', fontsize=16)
        
        metrics = ['avg_pumps', 'explosion_rate', 'avg_adjusted_pumps']
        human_values = [self.human_baseline[m] for m in metrics]
        metric_names = ['Average Pumps', 'Explosion Rate', 'Avg Adjusted Pumps']
        
        for i, (metric, human_val, name) in enumerate(zip(metrics, human_values, metric_names)):
            ax = axes[i]
            
            # Get top 10 configurations for this metric
            if metric == 'explosion_rate':
                # Lower is better for explosion rate
                top_configs = df.nsmallest(10, metric)
            else:
                # Higher is better for others (or closer to human baseline)
                top_configs = df.nlargest(10, metric)
                
            # Plot AI performance distribution
            ax.hist(df[metric], bins=20, alpha=0.7, label='All Configurations', color='lightblue')
            ax.hist(top_configs[metric], bins=10, alpha=0.8, label='Top 10 Configurations', color='blue')
            
            # Plot human baseline
            ax.axvline(human_val, color='red', linestyle='--', linewidth=2, label='Human Baseline')
            
            ax.set_xlabel(name)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name} Distribution')
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'human_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find best matching configuration to human performance
        df['human_similarity'] = 0
        for metric in ['avg_pumps', 'explosion_rate', 'avg_adjusted_pumps']:
            normalized_diff = abs(df[metric] - self.human_baseline[metric]) / self.human_baseline[metric]
            df['human_similarity'] += normalized_diff
            
        best_match = df.loc[df['human_similarity'].idxmin()]
        print("\nBest human-matching configuration:")
        print(f"Risk weight: {best_match['risk_weight']:.2f}")
        print(f"Cautious weight: {best_match['cautious_weight']:.2f}") 
        print(f"Temperature: {best_match['temperature']:.2f}")
        print(f"Avg pumps: {best_match['avg_pumps']:.2f} (human: {self.human_baseline['avg_pumps']:.2f})")
        print(f"Explosion rate: {best_match['explosion_rate']:.3f} (human: {self.human_baseline['explosion_rate']:.3f})")
        print(f"Avg adjusted pumps: {best_match['avg_adjusted_pumps']:.2f} (human: {self.human_baseline['avg_adjusted_pumps']:.2f})")
        
    def _plot_best_configurations(self, df: pd.DataFrame):
        """Highlight best performing configurations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Best Performing Configurations', fontsize=16)
        
        # Define what "best" means for different objectives
        objectives = {
            'Highest Reward': ('total_reward', True),
            'Lowest Explosion Rate': ('explosion_rate', False), 
            'Most Consistent': ('consistency_score', True),
            'Best Stopping': ('optimal_stopping_score', True)
        }
        
        for i, (obj_name, (metric, maximize)) in enumerate(objectives.items()):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if maximize:
                best_configs = df.nlargest(5, metric)
            else:
                best_configs = df.nsmallest(5, metric)
                
            # Create scatter plot with size based on performance
            scatter = ax.scatter(df['risk_weight'], df['cautious_weight'], 
                               c=df[metric], s=30, alpha=0.6, cmap='viridis')
            
            # Highlight best configurations
            ax.scatter(best_configs['risk_weight'], best_configs['cautious_weight'],
                      c='red', s=100, marker='*', edgecolors='black', linewidth=1,
                      label='Top 5')
            
            ax.set_xlabel('Risk Weight')
            ax.set_ylabel('Cautious Weight')
            ax.set_title(f'{obj_name}')
            ax.legend()
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax)
            
        plt.tight_layout()
        plt.savefig(self.output_dir / 'best_configurations.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Run complete BART evaluation"""
    evaluator = BARTEvaluator()
    
    print("Starting BART empirical evaluation...")
    
    # Define parameter ranges for grid search
    risk_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cautious_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    temperatures = [0.3, 0.5, 0.8, 1.0, 1.2]
    
    # Run evaluation
    results = evaluator.run_evaluation(
        risk_weights=risk_weights,
        cautious_weights=cautious_weights, 
        temperatures=temperatures,
        n_trials=30,  # trials per seed
        n_seeds=3,    # multiple seeds for robustness
        max_workers=6
    )
    
    # Save results
    filepath = evaluator.save_results(results)
    
    # Create visualizations  
    evaluator.create_visualizations(results)
    
    print(f"\nEvaluation complete! Results saved to {filepath}")
    print(f"Visualizations saved to {evaluator.output_dir}")
    
    # Print summary statistics
    df = []
    for result in results:
        row = result.config.copy()
        row.update({
            'total_reward': result.total_reward,
            'explosion_rate': result.explosion_rate,
            'avg_pumps': result.avg_pumps,
            'optimal_stopping_score': result.optimal_stopping_score
        })
        df.append(row)
    
    df = pd.DataFrame(df)
    
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total configurations tested: {len(results)}")
    print(f"Best total reward: {df['total_reward'].max():.2f}")
    print(f"Lowest explosion rate: {df['explosion_rate'].min():.3f}")
    print(f"Highest average pumps: {df['avg_pumps'].max():.2f}")
    print(f"Best optimal stopping score: {df['optimal_stopping_score'].max():.3f}")

if __name__ == "__main__":
    main()