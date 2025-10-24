"""
Quick BART Evaluation - No external dependencies required
Simulates persona behavior based on personality weights for immediate testing
"""

import json
import random
import time
import itertools
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

@dataclass
class QuickTrialResult:
    """Results from a single BART trial"""
    trial_id: int
    pumps: int
    exploded: bool
    reward: float
    threshold: int
    risk_weight: float
    cautious_weight: float
    temperature: float
    decisions: List[str]

class QuickBARTEnvironment:
    """Simplified BART environment"""
    
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

class PersonaAgent:
    """Simulates personality-based decision making"""
    
    def __init__(self, risk_weight: float, cautious_weight: float, temperature: float):
        # Normalize weights
        total = risk_weight + cautious_weight
        self.risk_weight = risk_weight / total
        self.cautious_weight = cautious_weight / total
        self.temperature = temperature
        
    def decide(self, pumps: int, threshold_hint: int = None) -> str:
        """Make PUMP/COLLECT decision based on current state"""
        
        # Risk-taker behavior: aggressive early, still willing to continue
        risk_pump_prob = max(0.05, 0.95 - pumps * 0.025)
        
        # Cautious behavior: conservative throughout, very risk-averse
        cautious_pump_prob = max(0.02, 0.7 - pumps * 0.08)
        
        # Weighted combination
        base_prob = (self.risk_weight * risk_pump_prob + 
                    self.cautious_weight * cautious_pump_prob)
        
        # Apply temperature (exploration vs exploitation)
        if self.temperature > 0:
            # Add temperature-based noise
            noise = np.random.normal(0, self.temperature * 0.2)
            adjusted_prob = max(0.01, min(0.99, base_prob + noise))
        else:
            adjusted_prob = base_prob
            
        return "PUMP" if random.random() < adjusted_prob else "COLLECT"

class QuickBARTEvaluator:
    """Fast BART evaluation system"""
    
    def __init__(self, output_dir: str = "quick_bart_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Human baseline from literature
        self.human_baseline = {
            'avg_pumps': 16.5,
            'explosion_rate': 0.37,
            'avg_adjusted_pumps': 20.2
        }
        
    def run_single_trial(self, risk_weight: float, cautious_weight: float, 
                        temperature: float, trial_id: int, seed: int) -> QuickTrialResult:
        """Run single BART trial"""
        
        env = QuickBARTEnvironment()
        agent = PersonaAgent(risk_weight, cautious_weight, temperature)
        
        env.reset(seed)
        decisions = []
        total_reward = 0
        
        while not env.done:
            action = agent.decide(env.pumps)
            decisions.append(action)
            reward, done = env.step(action)
            total_reward += reward
            
        return QuickTrialResult(
            trial_id=trial_id,
            pumps=env.pumps,
            exploded=env.exploded,
            reward=total_reward,
            threshold=env.threshold,
            risk_weight=risk_weight,
            cautious_weight=cautious_weight,
            temperature=temperature,
            decisions=decisions
        )
        
    def run_grid_search(self, n_trials: int = 25, n_seeds: int = 3) -> List[Dict]:
        """Run comprehensive grid search evaluation"""
        
        # Parameter ranges
        risk_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        temperatures = [0.3, 0.5, 0.8, 1.0, 1.2]
        
        results = []
        total_configs = len(risk_weights) * len(temperatures)
        
        print(f"Running quick evaluation: {total_configs} configurations")
        print(f"Each config: {n_trials} trials × {n_seeds} seeds = {n_trials * n_seeds} total trials")
        
        config_idx = 0
        for risk_w, temp in itertools.product(risk_weights, temperatures):
            config_idx += 1
            cautious_w = 1.0 - risk_w  # Complementary weights
            
            print(f"Config {config_idx}/{total_configs}: Risk={risk_w:.1f}, Temp={temp:.1f}")
            
            all_trials = []
            
            # Multiple seeds for robustness
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
        
    def _calculate_metrics(self, trials: List[QuickTrialResult]) -> Dict:
        """Calculate performance metrics"""
        
        total_reward = sum(t.reward for t in trials)
        explosions = sum(1 for t in trials if t.exploded)
        explosion_rate = explosions / len(trials)
        
        avg_pumps = np.mean([t.pumps for t in trials])
        
        # Adjusted pumps (only successful balloons)
        successful = [t for t in trials if not t.exploded]
        avg_adjusted_pumps = np.mean([t.pumps for t in successful]) if successful else 0
        
        # Consistency score (inverse of variance)
        pump_variance = np.var([t.pumps for t in trials])
        consistency_score = 1.0 / (1.0 + pump_variance)
        
        # Optimal stopping score
        optimal_distances = []
        for t in trials:
            if not t.exploded:
                # Distance from optimal stopping point (threshold - 1)
                optimal_distances.append(abs(t.pumps - (t.threshold - 1)))
            else:
                # Penalty for explosion
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
        """Save results to JSON"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"quick_bart_results_{timestamp}.json"
            
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {filepath}")
        return filepath
        
    def create_comprehensive_visualizations(self, results: List[Dict]):
        """Create all visualizations at once"""
        
        # Convert to DataFrame
        data = []
        for result in results:
            row = result['config'].copy()
            row.update({k: v for k, v in result.items() if k not in ['config', 'trials']})
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Create 6-panel comprehensive visualization
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Risk weight effects
        ax1 = plt.subplot(2, 3, 1)
        grouped = df.groupby('risk_weight').agg({
            'avg_pumps': 'mean',
            'explosion_rate': 'mean'
        })
        
        ax1_twin = ax1.twinx()
        line1 = ax1.plot(grouped.index, grouped['avg_pumps'], 'b-o', linewidth=2, markersize=8, label='Avg Pumps')
        line2 = ax1_twin.plot(grouped.index, grouped['explosion_rate'], 'r-s', linewidth=2, markersize=8, label='Explosion Rate')
        
        ax1.set_xlabel('Risk Taker Weight', fontsize=12)
        ax1.set_ylabel('Average Pumps', color='blue', fontsize=12)
        ax1_twin.set_ylabel('Explosion Rate', color='red', fontsize=12)
        ax1.set_title('Risk Weight Effects', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
        
        # 2. Temperature effects heatmap
        ax2 = plt.subplot(2, 3, 2)
        pivot = df.pivot(index='temperature', columns='risk_weight', values='consistency_score')
        im = ax2.imshow(pivot, cmap='viridis', aspect='auto')
        ax2.set_xticks(range(len(pivot.columns)))
        ax2.set_xticklabels([f'{x:.1f}' for x in pivot.columns])
        ax2.set_yticks(range(len(pivot.index)))
        ax2.set_yticklabels([f'{y:.1f}' for y in pivot.index])
        ax2.set_xlabel('Risk Weight', fontsize=12)
        ax2.set_ylabel('Temperature', fontsize=12)
        ax2.set_title('Consistency Score Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax2)
        
        # 3. Human comparison
        ax3 = plt.subplot(2, 3, 3)
        metrics = ['avg_pumps', 'explosion_rate', 'avg_adjusted_pumps']
        human_vals = [self.human_baseline[m] for m in metrics]
        ai_means = [df[m].mean() for m in metrics]
        ai_stds = [df[m].std() for m in metrics]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax3.bar(x_pos - width/2, human_vals, width, label='Human Baseline', color='red', alpha=0.7)
        bars2 = ax3.bar(x_pos + width/2, ai_means, width, yerr=ai_stds, label='AI Mean ± Std', 
                       color='blue', alpha=0.7, capsize=5)
        
        ax3.set_xlabel('Metrics', fontsize=12)
        ax3.set_ylabel('Values', fontsize=12)
        ax3.set_title('AI vs Human Performance', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(['Avg Pumps', 'Explosion Rate', 'Adj Pumps'], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance scatter
        ax4 = plt.subplot(2, 3, 4)
        scatter = ax4.scatter(df['explosion_rate'], df['total_reward'], 
                             c=df['risk_weight'], s=100, alpha=0.7, cmap='RdYlBu', edgecolors='black')
        ax4.set_xlabel('Explosion Rate', fontsize=12)
        ax4.set_ylabel('Total Reward', fontsize=12)
        ax4.set_title('Risk vs Reward Trade-off', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Risk Weight')
        
        # 5. Temperature effects on decision quality
        ax5 = plt.subplot(2, 3, 5)
        temp_grouped = df.groupby('temperature').agg({
            'consistency_score': ['mean', 'std'],
            'optimal_stopping_score': ['mean', 'std']
        })
        
        temps = temp_grouped.index
        consistency_mean = temp_grouped[('consistency_score', 'mean')]
        consistency_std = temp_grouped[('consistency_score', 'std')]
        optimal_mean = temp_grouped[('optimal_stopping_score', 'mean')]
        optimal_std = temp_grouped[('optimal_stopping_score', 'std')]
        
        ax5.errorbar(temps, consistency_mean, yerr=consistency_std, 
                    marker='o', capsize=5, linewidth=2, markersize=8, label='Consistency')
        ax5.errorbar(temps, optimal_mean, yerr=optimal_std,
                    marker='s', capsize=5, linewidth=2, markersize=8, label='Optimal Stopping')
        
        ax5.set_xlabel('Temperature', fontsize=12)
        ax5.set_ylabel('Score', fontsize=12)
        ax5.set_title('Temperature vs Decision Quality', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Best configurations
        ax6 = plt.subplot(2, 3, 6)
        
        # Find Pareto optimal configurations
        pareto_configs = []
        for i, result in enumerate(results):
            is_pareto = True
            for j, other_result in enumerate(results):
                if i != j:
                    if (other_result['explosion_rate'] <= result['explosion_rate'] and 
                        other_result['total_reward'] >= result['total_reward'] and
                        (other_result['explosion_rate'] < result['explosion_rate'] or 
                         other_result['total_reward'] > result['total_reward'])):
                        is_pareto = False
                        break
            if is_pareto:
                pareto_configs.append((i, result))
        
        # Plot all configurations
        ax6.scatter(df['explosion_rate'], df['total_reward'], 
                   c='lightblue', s=60, alpha=0.6, label='All Configs')
        
        # Highlight Pareto optimal
        pareto_explosions = [results[i]['explosion_rate'] for i, _ in pareto_configs]
        pareto_rewards = [results[i]['total_reward'] for i, _ in pareto_configs]
        ax6.scatter(pareto_explosions, pareto_rewards, 
                   c='red', s=150, marker='*', edgecolors='black', linewidth=2, label='Pareto Optimal')
        
        ax6.set_xlabel('Explosion Rate', fontsize=12)
        ax6.set_ylabel('Total Reward', fontsize=12)
        ax6.set_title('Pareto Optimal Configurations', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('BART Personality Grid Search Results', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(self.output_dir / 'comprehensive_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print analysis
        self._print_analysis(df, pareto_configs, results)
        
    def _print_analysis(self, df: pd.DataFrame, pareto_configs: List, results: List[Dict]):
        """Print detailed analysis of results"""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE BART EVALUATION RESULTS")
        print("="*80)
        
        print(f"\nDataset Summary:")
        print(f"- Total configurations tested: {len(results)}")
        print(f"- Risk weight range: {df['risk_weight'].min():.1f} - {df['risk_weight'].max():.1f}")
        print(f"- Temperature range: {df['temperature'].min():.1f} - {df['temperature'].max():.1f}")
        print(f"- Total trials per config: {results[0]['n_trials']}")
        
        print(f"\nPerformance Statistics:")
        print(f"- Average pumps: {df['avg_pumps'].mean():.2f} ± {df['avg_pumps'].std():.2f}")
        print(f"- Explosion rate: {df['explosion_rate'].mean():.3f} ± {df['explosion_rate'].std():.3f}")
        print(f"- Total reward: {df['total_reward'].mean():.2f} ± {df['total_reward'].std():.2f}")
        print(f"- Consistency: {df['consistency_score'].mean():.3f} ± {df['consistency_score'].std():.3f}")
        
        print(f"\nBest Individual Performers:")
        
        best_reward = df.loc[df['total_reward'].idxmax()]
        print(f"- Highest reward ({best_reward['total_reward']:.2f}): "
              f"Risk={best_reward['risk_weight']:.1f}, Temp={best_reward['temperature']:.1f}")
        
        best_consistency = df.loc[df['consistency_score'].idxmax()]
        print(f"- Most consistent ({best_consistency['consistency_score']:.3f}): "
              f"Risk={best_consistency['risk_weight']:.1f}, Temp={best_consistency['temperature']:.1f}")
        
        lowest_explosion = df.loc[df['explosion_rate'].idxmin()]
        print(f"- Lowest explosions ({lowest_explosion['explosion_rate']:.3f}): "
              f"Risk={lowest_explosion['risk_weight']:.1f}, Temp={lowest_explosion['temperature']:.1f}")
        
        print(f"\nPareto Optimal Configurations ({len(pareto_configs)} found):")
        for i, (idx, config) in enumerate(pareto_configs[:5]):  # Show top 5
            print(f"  {i+1}. Risk={config['config']['risk_weight']:.1f}, "
                  f"Temp={config['config']['temperature']:.1f} -> "
                  f"Explosion={config['explosion_rate']:.3f}, Reward={config['total_reward']:.2f}")
        
        # Human comparison
        print(f"\nComparison with Human Baseline:")
        for metric in ['avg_pumps', 'explosion_rate', 'avg_adjusted_pumps']:
            human_val = self.human_baseline[metric]
            ai_mean = df[metric].mean()
            diff_pct = ((ai_mean - human_val) / human_val) * 100
            closest_idx = (df[metric] - human_val).abs().idxmin()
            closest_config = df.loc[closest_idx]
            
            print(f"- {metric}: Human={human_val:.3f}, AI_avg={ai_mean:.3f} ({diff_pct:+.1f}%)")
            print(f"  Closest match: Risk={closest_config['risk_weight']:.1f}, "
                  f"Temp={closest_config['temperature']:.1f}, Value={closest_config[metric]:.3f}")
        
        print(f"\nKey Insights:")
        print(f"- Risk-reward correlation: {df['risk_weight'].corr(df['total_reward']):.3f}")
        print(f"- Risk-explosion correlation: {df['risk_weight'].corr(df['explosion_rate']):.3f}")
        print(f"- Temperature-consistency correlation: {df['temperature'].corr(df['consistency_score']):.3f}")
        
        # Find best human match overall
        df_copy = df.copy()
        similarity_scores = []
        for _, row in df_copy.iterrows():
            score = 0
            for metric in ['avg_pumps', 'explosion_rate', 'avg_adjusted_pumps']:
                normalized_diff = abs(row[metric] - self.human_baseline[metric]) / self.human_baseline[metric]
                score += normalized_diff
            similarity_scores.append(score)
        
        df_copy['human_similarity'] = similarity_scores
        best_human_match = df_copy.loc[df_copy['human_similarity'].idxmin()]
        
        print(f"\nBest Overall Human Match:")
        print(f"- Configuration: Risk={best_human_match['risk_weight']:.1f}, "
              f"Temp={best_human_match['temperature']:.1f}")
        print(f"- Similarity score: {best_human_match['human_similarity']:.3f} (lower is better)")
        print(f"- Performance: Pumps={best_human_match['avg_pumps']:.2f}, "
              f"Explosions={best_human_match['explosion_rate']:.3f}, "
              f"Adjusted={best_human_match['avg_adjusted_pumps']:.2f}")

def main():
    """Run quick BART evaluation"""
    
    print("Starting Quick BART Evaluation...")
    print("This runs entirely offline with simulated persona behavior")
    print("No external APIs or models required!\n")
    
    evaluator = QuickBARTEvaluator()
    
    start_time = time.time()
    
    # Run evaluation
    results = evaluator.run_grid_search(n_trials=30, n_seeds=3)
    
    end_time = time.time()
    print(f"\nEvaluation completed in {end_time - start_time:.1f} seconds")
    
    # Save results
    filepath = evaluator.save_results(results)
    
    # Create comprehensive visualization
    evaluator.create_comprehensive_visualizations(results)
    
    print(f"\n🎉 Quick BART evaluation complete!")
    print(f"📁 Results saved to: {filepath}")
    print(f"📊 Visualization saved to: {evaluator.output_dir / 'comprehensive_results.png'}")

if __name__ == "__main__":
    main()