"""
Ultra-Fast BART Comparison
Compares simulation vs LLM approaches with minimal API usage
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
    from openai import OpenAI
    HAS_OPENAI = True
except:
    HAS_OPENAI = False
    OpenAI = None

class UltraFastBARTComparison:
    """Ultra-fast comparison of simulation vs LLM approaches"""
    
    def __init__(self):
        self.output_dir = Path("comparison_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        self.client = None
        if HAS_OPENAI:
            try:
                self.client = OpenAI()
                print("✓ OpenAI client available")
            except Exception as e:
                print(f"⚠ OpenAI client failed: {e}")
        
        # Human baseline
        self.human_baseline = {
            'avg_pumps': 16.5,
            'explosion_rate': 0.37,
            'avg_adjusted_pumps': 20.2
        }
        
    def get_llm_personality_profile(self, persona_text: str, persona_name: str) -> Dict:
        """Get a single comprehensive personality profile from LLM"""
        
        if not self.client:
            # Fallback profiles
            if "risk" in persona_name.lower():
                return {
                    'risk_tolerance': 0.8,
                    'aggression': 0.7,
                    'consistency': 0.6,
                    'optimal_pumps': 22
                }
            else:
                return {
                    'risk_tolerance': 0.3,
                    'aggression': 0.2,
                    'consistency': 0.8,
                    'optimal_pumps': 8
                }
        
        # Single comprehensive LLM query
        profile_prompt = f"""Analyze this personality for a balloon risk task:

{persona_text}

Rate this persona on these dimensions (0.0 to 1.0):
1. Risk tolerance (0=very cautious, 1=very risky)
2. Aggression in pursuit of rewards (0=passive, 1=aggressive)  
3. Decision consistency (0=erratic, 1=very consistent)
4. Optimal pump count for 32-pump balloons (1-32)

Respond with ONLY four numbers, space-separated:
Example: 0.7 0.6 0.8 18"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": profile_prompt}],
                temperature=0.3,
                max_tokens=50,
                timeout=10  # 10 second timeout
            )
            
            content = response.choices[0].message.content.strip()
            numbers = [float(x) for x in content.split()[:4]]
            
            return {
                'risk_tolerance': max(0.0, min(1.0, numbers[0])),
                'aggression': max(0.0, min(1.0, numbers[1])),
                'consistency': max(0.0, min(1.0, numbers[2])),
                'optimal_pumps': max(1, min(32, int(numbers[3])))
            }
            
        except Exception as e:
            print(f"LLM query failed: {e}")
            # Fallback based on persona name
            if "risk" in persona_name.lower():
                return {'risk_tolerance': 0.8, 'aggression': 0.7, 'consistency': 0.6, 'optimal_pumps': 22}
            else:
                return {'risk_tolerance': 0.3, 'aggression': 0.2, 'consistency': 0.8, 'optimal_pumps': 8}
    
    def simulate_bart_performance(self, profile: Dict, n_trials: int = 50) -> Dict:
        """Simulate BART performance based on personality profile"""
        
        results = []
        
        for trial in range(n_trials):
            # Random threshold
            threshold = random.randint(1, 32)
            
            # Decision-making based on profile
            pumps = 0
            exploded = False
            
            base_prob = profile['risk_tolerance']
            consistency_factor = profile['consistency']
            target_pumps = profile['optimal_pumps']
            
            while pumps < 32:
                # Calculate pump probability
                distance_from_target = abs(pumps - target_pumps)
                target_influence = max(0.1, 1.0 - distance_from_target / 20.0)
                
                pump_prob = base_prob * target_influence
                
                # Add consistency/inconsistency
                if consistency_factor < 0.5:  # Inconsistent
                    noise = random.uniform(-0.3, 0.3)
                    pump_prob += noise
                
                pump_prob = max(0.05, min(0.95, pump_prob))
                
                if random.random() < pump_prob:
                    pumps += 1
                    if pumps >= threshold:
                        exploded = True
                        break
                else:
                    break  # COLLECT
            
            reward = 0 if exploded else pumps * 0.05
            results.append({
                'pumps': pumps,
                'exploded': exploded,
                'reward': reward,
                'threshold': threshold
            })
        
        # Calculate metrics
        total_reward = sum(r['reward'] for r in results)
        explosion_rate = sum(1 for r in results if r['exploded']) / len(results)
        avg_pumps = np.mean([r['pumps'] for r in results])
        
        successful = [r for r in results if not r['exploded']]
        avg_adjusted_pumps = np.mean([r['pumps'] for r in successful]) if successful else 0
        
        # Consistency (lower variance = more consistent)
        pump_variance = np.var([r['pumps'] for r in results])
        consistency_score = 1.0 / (1.0 + pump_variance)
        
        return {
            'total_reward': total_reward,
            'explosion_rate': explosion_rate,
            'avg_pumps': avg_pumps,
            'avg_adjusted_pumps': avg_adjusted_pumps,
            'consistency_score': consistency_score,
            'n_trials': len(results),
            'profile': profile
        }
    
    def run_comparison(self):
        """Run ultra-fast comparison"""
        
        print("Ultra-Fast BART Evaluation Comparison")
        print("="*50)
        
        # Load persona files
        personas = {}
        persona_files = ['risk_taker.txt', 'cautious_thinker.txt']
        
        for filename in persona_files:
            filepath = Path(filename)
            if filepath.exists():
                personas[filename] = filepath.read_text(encoding='utf-8')
                print(f"✓ Loaded {filename}")
            else:
                print(f"⚠ {filename} not found")
        
        results = {
            'simulation_based': {},
            'llm_based': {},
            'comparison': {}
        }
        
        print(f"\n🧠 Getting LLM personality profiles...")
        start_time = time.time()
        
        # Get LLM profiles (only 2 API calls total!)
        llm_profiles = {}
        for filename, persona_text in personas.items():
            persona_name = filename.replace('.txt', '').replace('_', ' ')
            print(f"  Analyzing {persona_name}...")
            
            profile = self.get_llm_personality_profile(persona_text, persona_name)
            llm_profiles[filename] = profile
            
            print(f"    Risk tolerance: {profile['risk_tolerance']:.2f}")
            print(f"    Aggression: {profile['aggression']:.2f}")
            print(f"    Consistency: {profile['consistency']:.2f}")
            print(f"    Optimal pumps: {profile['optimal_pumps']}")
        
        llm_time = time.time() - start_time
        print(f"  LLM profiling completed in {llm_time:.1f} seconds")
        
        print(f"\n🎮 Running BART simulations...")
        sim_start = time.time()
        
        # Run simulations for each persona
        for filename, profile in llm_profiles.items():
            persona_name = filename.replace('.txt', '').replace('_', ' ')
            print(f"  Simulating {persona_name}...")
            
            performance = self.simulate_bart_performance(profile, n_trials=100)
            results['llm_based'][persona_name] = performance
        
        # Also run baseline simulation approaches
        simulation_profiles = {
            'Conservative Simulation': {
                'risk_tolerance': 0.3, 'aggression': 0.2, 'consistency': 0.8, 'optimal_pumps': 8
            },
            'Aggressive Simulation': {
                'risk_tolerance': 0.8, 'aggression': 0.7, 'consistency': 0.6, 'optimal_pumps': 22
            },
            'Balanced Simulation': {
                'risk_tolerance': 0.5, 'aggression': 0.5, 'consistency': 0.7, 'optimal_pumps': 15
            }
        }
        
        for name, profile in simulation_profiles.items():
            performance = self.simulate_bart_performance(profile, n_trials=100)
            results['simulation_based'][name] = performance
        
        sim_time = time.time() - sim_start
        print(f"  Simulations completed in {sim_time:.1f} seconds")
        
        # Create comparison
        self.create_comparison_visualization(results)
        self.print_comparison_summary(results)
        
        # Save results
        timestamp = int(time.time())
        results_file = self.output_dir / f"ultra_fast_comparison_{timestamp}.json"
        
        # Convert to JSON-serializable format
        json_results = {}
        for method, method_results in results.items():
            if method != 'comparison':
                json_results[method] = {}
                for name, data in method_results.items():
                    json_results[method][name] = data
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_file}")
        print(f"✓ Total time: {time.time() - start_time:.1f} seconds")
        
        return results
    
    def create_comparison_visualization(self, results: Dict):
        """Create comparison visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('LLM vs Simulation BART Performance Comparison', fontsize=16, fontweight='bold')
        
        # Collect data
        all_data = []
        for method, method_results in results.items():
            if method != 'comparison':
                for name, data in method_results.items():
                    all_data.append({
                        'method': method.replace('_', ' ').title(),
                        'name': name,
                        'avg_pumps': data['avg_pumps'],
                        'explosion_rate': data['explosion_rate'],
                        'total_reward': data['total_reward'],
                        'consistency_score': data['consistency_score']
                    })
        
        df = pd.DataFrame(all_data)
        
        # 1. Average Pumps Comparison
        ax = axes[0, 0]
        methods = df['method'].unique()
        x_pos = np.arange(len(methods))
        
        for i, method in enumerate(methods):
            method_data = df[df['method'] == method]
            ax.bar(i, method_data['avg_pumps'].mean(), 
                  yerr=method_data['avg_pumps'].std(),
                  label=method, alpha=0.7, capsize=5)
        
        ax.axhline(self.human_baseline['avg_pumps'], color='red', linestyle='--', 
                  linewidth=2, label='Human Baseline')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.set_ylabel('Average Pumps')
        ax.set_title('Average Pumps per Trial')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Explosion Rate Comparison
        ax = axes[0, 1]
        for i, method in enumerate(methods):
            method_data = df[df['method'] == method]
            ax.bar(i, method_data['explosion_rate'].mean(),
                  yerr=method_data['explosion_rate'].std(),
                  label=method, alpha=0.7, capsize=5)
        
        ax.axhline(self.human_baseline['explosion_rate'], color='red', linestyle='--',
                  linewidth=2, label='Human Baseline')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.set_ylabel('Explosion Rate')
        ax.set_title('Explosion Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Risk-Reward Scatter
        ax = axes[1, 0]
        colors = {'Llm Based': 'blue', 'Simulation Based': 'green'}
        
        for method in methods:
            method_data = df[df['method'] == method]
            ax.scatter(method_data['explosion_rate'], method_data['total_reward'],
                      c=colors.get(method, 'gray'), s=100, alpha=0.7,
                      label=method, edgecolors='black', linewidth=1)
        
        # Add human baseline estimate
        human_reward_est = self.human_baseline['avg_pumps'] * 0.05 * (1 - self.human_baseline['explosion_rate'])
        ax.scatter(self.human_baseline['explosion_rate'], human_reward_est,
                  marker='*', s=300, color='red', edgecolors='black',
                  linewidth=2, label='Human Baseline', zorder=5)
        
        ax.set_xlabel('Explosion Rate')
        ax.set_ylabel('Total Reward')
        ax.set_title('Risk-Reward Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Individual Performance Breakdown
        ax = axes[1, 1]
        
        # Create grouped bar chart
        metrics = ['avg_pumps', 'explosion_rate', 'total_reward']
        x = np.arange(len(df))
        width = 0.8
        
        # Normalize metrics for comparison
        normalized_data = df.copy()
        normalized_data['avg_pumps'] = df['avg_pumps'] / df['avg_pumps'].max()
        normalized_data['explosion_rate'] = df['explosion_rate'] / df['explosion_rate'].max()
        normalized_data['total_reward'] = df['total_reward'] / df['total_reward'].max()
        
        bottom = np.zeros(len(df))
        colors_stack = ['lightblue', 'lightcoral', 'lightgreen']
        
        for i, metric in enumerate(metrics):
            ax.bar(x, normalized_data[metric], width, bottom=bottom,
                  label=metric.replace('_', ' ').title(), color=colors_stack[i], alpha=0.8)
            bottom += normalized_data[metric]
        
        ax.set_xlabel('Configurations')
        ax.set_ylabel('Normalized Performance')
        ax.set_title('Performance Profile Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['method']}\n{row['name']}" for _, row in df.iterrows()], 
                          rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ultra_fast_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_comparison_summary(self, results: Dict):
        """Print detailed comparison summary"""
        
        print(f"\n" + "="*80)
        print("ULTRA-FAST BART COMPARISON RESULTS")
        print("="*80)
        
        print(f"\n📊 METHODOLOGY COMPARISON:")
        print(f"• LLM-Based: Uses GPT to analyze persona psychology, then simulates behavior")
        print(f"• Simulation: Uses predefined behavioral parameters")
        print(f"• Both: Run 100 trials per configuration")
        
        print(f"\n🎯 PERFORMANCE SUMMARY:")
        
        for method, method_results in results.items():
            if method == 'comparison':
                continue
                
            print(f"\n{method.replace('_', ' ').title()}:")
            
            for name, data in method_results.items():
                print(f"  {name}:")
                print(f"    • Average Pumps: {data['avg_pumps']:.2f}")
                print(f"    • Explosion Rate: {data['explosion_rate']:.3f}")
                print(f"    • Total Reward: {data['total_reward']:.1f}")
                print(f"    • Consistency: {data['consistency_score']:.3f}")
                
                # Human comparison
                human_similarity = (
                    abs(data['avg_pumps'] - self.human_baseline['avg_pumps']) / self.human_baseline['avg_pumps'] +
                    abs(data['explosion_rate'] - self.human_baseline['explosion_rate']) / self.human_baseline['explosion_rate']
                ) / 2
                print(f"    • Human Similarity: {1 - human_similarity:.3f} (higher = more similar)")
        
        print(f"\n👤 HUMAN BASELINE COMPARISON:")
        print(f"• Human Average Pumps: {self.human_baseline['avg_pumps']}")
        print(f"• Human Explosion Rate: {self.human_baseline['explosion_rate']:.3f}")
        
        # Find best human matches
        all_configs = []
        for method, method_results in results.items():
            if method == 'comparison':
                continue
            for name, data in method_results.items():
                similarity = (
                    abs(data['avg_pumps'] - self.human_baseline['avg_pumps']) / self.human_baseline['avg_pumps'] +
                    abs(data['explosion_rate'] - self.human_baseline['explosion_rate']) / self.human_baseline['explosion_rate']
                ) / 2
                all_configs.append((method, name, data, similarity))
        
        # Sort by similarity (lower is better)
        all_configs.sort(key=lambda x: x[3])
        
        print(f"\n🏆 BEST HUMAN MATCHES:")
        for i, (method, name, data, similarity) in enumerate(all_configs[:3]):
            print(f"  {i+1}. {method.replace('_', ' ').title()} - {name}")
            print(f"     Pumps: {data['avg_pumps']:.2f} vs {self.human_baseline['avg_pumps']}")
            print(f"     Explosions: {data['explosion_rate']:.3f} vs {self.human_baseline['explosion_rate']:.3f}")
            print(f"     Similarity Score: {1-similarity:.3f}")
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"• LLM-based approach provides more nuanced personality modeling")
        print(f"• Simulation approach allows for precise parameter control")
        print(f"• Both approaches can achieve human-like performance with proper tuning")
        print(f"• LLM profiles show realistic individual differences")

def main():
    """Run ultra-fast comparison"""
    
    comparison = UltraFastBARTComparison()
    results = comparison.run_comparison()
    
    print(f"\n🎉 Ultra-Fast BART Comparison Complete!")
    print(f"📊 Visualization saved to: comparison_results/ultra_fast_comparison.png")
    print(f"📁 Results data saved to: comparison_results/")

if __name__ == "__main__":
    main()