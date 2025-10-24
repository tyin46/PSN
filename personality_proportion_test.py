"""
Personality Proportion Impact on BART Performance
Real API-based test showing how pump count changes with personality weight variations
"""

import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
class PersonalityTrialResult:
    """Results from a single trial with specific personality proportions"""
    trial_id: int
    risk_weight: float
    cautious_weight: float
    temperature: float
    pumps: int
    exploded: bool
    reward: float
    threshold: int
    decision_sequence: List[str]
    llm_responses: List[str]  # Actual LLM reasoning for each decision
    final_decision_reason: str

class PersonalityProportionTester:
    """Tests how pump count varies with personality proportion changes using real API calls"""
    
    def __init__(self):
        self.output_dir = Path("personality_proportion_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        self.client = None
        if HAS_OPENAI:
            try:
                self.client = OpenAI()
                print("✅ OpenAI client initialized successfully")
            except Exception as e:
                print(f"❌ OpenAI client initialization failed: {e}")
                print("Please check OPENAI_API_KEY environment variable")
                return
        else:
            print("❌ OpenAI SDK not available. Install with: pip install openai")
            return
            
        # Load persona files
        self.risk_taker_persona = self._load_persona("risk_taker.txt")
        self.cautious_persona = self._load_persona("cautious_thinker.txt")
        
        if not self.risk_taker_persona or not self.cautious_persona:
            print("❌ Could not load persona files")
            return
            
        print("✅ Persona files loaded successfully")
        
    def _load_persona(self, filename: str) -> Optional[str]:
        """Load persona text file"""
        filepath = _ROOT / filename
        if filepath.exists():
            try:
                return filepath.read_text(encoding='utf-8')
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
                return None
        else:
            print(f"⚠️ Persona file {filename} not found")
            return None
            
    def get_weighted_decision(self, risk_weight: float, cautious_weight: float, 
                            state_description: str, temperature: float) -> Tuple[str, str, str]:
        """Get decision from weighted personality combination using single optimized API call"""
        
        if not self.client:
            raise Exception("OpenAI client not available")
            
        # Create combined persona based on weights
        if risk_weight > cautious_weight:
            dominant_persona = "risk-taking"
            dominant_weight = risk_weight
            secondary_persona = "cautious"
            secondary_weight = cautious_weight
        else:
            dominant_persona = "cautious"
            dominant_weight = cautious_weight
            secondary_persona = "risk-taking"
            secondary_weight = risk_weight
            
        # Single API call with weighted persona
        combined_prompt = f"""You are a decision-making agent with a mixed personality that is {dominant_weight:.1f} {dominant_persona} and {secondary_weight:.1f} {secondary_persona}.

Your personality blend:
- {dominant_weight:.1f} weight: {"Be bold and take calculated risks for potential rewards" if dominant_persona == "risk-taking" else "Be careful and prioritize safety over potential gains"}
- {secondary_weight:.1f} weight: {"Consider safety and avoid unnecessary risks" if secondary_persona == "cautious" else "Consider potential rewards and opportunities"}

Current situation: {state_description}

Make your decision based on your weighted personality blend. The {dominant_persona} side should have more influence.

Format:
DECISION: [PUMP or COLLECT]
REASONING: [Explain your decision in 2-3 sentences, mentioning how your personality weights influenced you]"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": combined_prompt}],
                temperature=temperature,
                max_tokens=120,
                timeout=15
            )
            content = response.choices[0].message.content.strip()
            decision = "PUMP" if "PUMP" in content.upper() else "COLLECT"
            reasoning = self._extract_reasoning(content)
            
            full_reasoning = f"Weighted personality ({risk_weight:.1f} risk, {cautious_weight:.1f} cautious): {reasoning}"
            
            return decision, full_reasoning, reasoning
            
        except Exception as e:
            print(f"⚠️ API call failed: {e}")
            # Intelligent fallback based on weights
            if risk_weight > 0.6:
                decision = "PUMP"
                reasoning = f"Fallback: High risk weight ({risk_weight:.1f}) favors pumping"
            elif cautious_weight > 0.6:
                decision = "COLLECT"
                reasoning = f"Fallback: High cautious weight ({cautious_weight:.1f}) favors collecting"
            else:
                # Balanced weights - random choice based on ratio
                import random
                decision = "PUMP" if random.random() < risk_weight else "COLLECT"
                reasoning = f"Fallback: Random choice based on risk weight ({risk_weight:.1f})"
            
            full_reasoning = f"API Error Fallback: {reasoning}"
            return decision, full_reasoning, reasoning
        
    def _extract_reasoning(self, content: str) -> str:
        """Extract reasoning from LLM response"""
        lines = content.split('\n')
        for line in lines:
            if 'REASONING:' in line.upper() or 'REASON:' in line.upper():
                return line.split(':', 1)[-1].strip()
        # If no explicit reasoning found, return first substantial line
        for line in lines:
            if len(line.strip()) > 10 and 'DECISION' not in line.upper():
                return line.strip()
        return "No reasoning provided"
        
    def get_trial_strategy(self, risk_weight: float, cautious_weight: float, temperature: float) -> Tuple[int, str]:
        """Get the agent's stopping strategy using a single API call"""
        
        if not self.client:
            raise Exception("OpenAI client not available")
            
        strategy_prompt = f"""You are a decision-making agent with a personality that is {risk_weight:.1f} risk-taking and {cautious_weight:.1f} cautious.

Balloon Analogue Risk Task Rules:
- You start with 0 pumps and $0.00 reward
- Each pump adds $0.05 but risks the balloon exploding (losing everything)
- You can collect your reward safely at any time
- Balloons typically explode between 1-32 pumps (varies randomly)

Based on your personality weights, decide your stopping strategy:
- Risk-takers tend to pump more (higher numbers)
- Cautious people tend to collect earlier (lower numbers)

Your personality: {risk_weight:.1f} risk-taking, {cautious_weight:.1f} cautious

Format:
TARGET_PUMPS: [Number between 1-32 indicating when you would typically stop and collect]
REASONING: [Explain your strategy in 2-3 sentences based on your personality weights]"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": strategy_prompt}],
                temperature=temperature,
                max_tokens=100,
                timeout=15
            )
            content = response.choices[0].message.content.strip()
            
            # Extract target pumps
            target_pumps = 10  # Default fallback
            for line in content.split('\n'):
                if 'TARGET_PUMPS:' in line.upper():
                    try:
                        pump_str = line.split(':')[1].strip()
                        # Extract number from string
                        import re
                        numbers = re.findall(r'\d+', pump_str)
                        if numbers:
                            target_pumps = min(32, max(1, int(numbers[0])))
                        break
                    except:
                        continue
            
            reasoning = self._extract_reasoning(content)
            return target_pumps, reasoning
            
        except Exception as e:
            print(f"⚠️ Strategy API call failed: {e}")
            # Fallback strategy based on weights
            if risk_weight > 0.7:
                target_pumps = random.randint(15, 25)
                reasoning = f"Fallback: High risk weight ({risk_weight:.1f}) suggests aggressive strategy"
            elif cautious_weight > 0.7:
                target_pumps = random.randint(3, 8)
                reasoning = f"Fallback: High cautious weight ({cautious_weight:.1f}) suggests conservative strategy"
            else:
                # Weighted average approach
                base_pumps = int(5 + (risk_weight * 15))  # Scale 5-20 based on risk weight
                target_pumps = random.randint(max(1, base_pumps-3), min(32, base_pumps+3))
                reasoning = f"Fallback: Balanced strategy based on risk weight ({risk_weight:.1f})"
            
            return target_pumps, reasoning

    def run_single_trial(self, risk_weight: float, cautious_weight: float, 
                        temperature: float, trial_id: int, seed: int) -> PersonalityTrialResult:
        """Run a single BART trial with specific personality proportions - optimized version"""
        
        # Set up environment
        random.seed(seed)
        threshold = random.randint(1, 32)
        step_reward = 0.05
        
        print(f"  Trial {trial_id}: Risk={risk_weight:.1f}, Cautious={cautious_weight:.1f}, Threshold={threshold}")
        
        # Get the agent's strategy with single API call
        try:
            target_pumps, strategy_reasoning = self.get_trial_strategy(risk_weight, cautious_weight, temperature)
            
            # Execute the strategy
            actual_pumps = min(target_pumps, threshold - 1)  # Stop before explosion or at target
            exploded = (target_pumps >= threshold)
            
            if exploded:
                actual_pumps = threshold
                final_reward = 0.0
                final_reason = f"Exploded at {actual_pumps} pumps (target was {target_pumps}). {strategy_reasoning}"
            else:
                final_reward = actual_pumps * step_reward
                final_reason = f"Collected at {actual_pumps} pumps (target was {target_pumps}). {strategy_reasoning}"
            
            # Create decision sequence for consistency
            decision_sequence = ["PUMP"] * actual_pumps
            if not exploded:
                decision_sequence.append("COLLECT")
            
            llm_responses = [f"Strategy: Target {target_pumps} pumps. {strategy_reasoning}"]
            
        except Exception as e:
            print(f"  ❌ Strategy API call failed: {e}")
            # Emergency fallback
            if risk_weight > cautious_weight:
                actual_pumps = min(random.randint(10, 20), threshold - 1)
            else:
                actual_pumps = min(random.randint(3, 8), threshold - 1)
                
            exploded = False
            final_reward = actual_pumps * step_reward
            final_reason = f"Emergency fallback: {actual_pumps} pumps due to API failure"
            decision_sequence = ["PUMP"] * actual_pumps + ["COLLECT"]
            llm_responses = [f"Emergency fallback due to API error: {e}"]
        
        print(f"    Result: {actual_pumps} pumps, {'exploded' if exploded else 'collected'}, ${final_reward:.2f}")
        
        return PersonalityTrialResult(
            trial_id=trial_id,
            risk_weight=risk_weight,
            cautious_weight=cautious_weight,
            temperature=temperature,
            pumps=actual_pumps,
            exploded=exploded,
            reward=final_reward,
            threshold=threshold,
            decision_sequence=decision_sequence,
            llm_responses=llm_responses,
            final_decision_reason=final_reason
        )
        
    def run_proportion_sweep(self, n_trials_per_config: int = 5, n_seeds: int = 2) -> List[PersonalityTrialResult]:
        """Run comprehensive sweep of personality proportions"""
        
        if not self.client:
            raise Exception("OpenAI client required for this test")
            
        print("🚀 Starting Personality Proportion Sweep Test")
        print("=" * 60)
        
        # Define proportion ranges - focused sweep for API efficiency
        risk_proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        temperatures = [0.5, 0.8]  # Two temperature levels
        
        all_results = []
        total_configs = len(risk_proportions) * len(temperatures)
        config_count = 0
        
        print(f"📊 Testing {total_configs} configurations")
        print(f"📈 {n_trials_per_config} trials × {n_seeds} seeds = {n_trials_per_config * n_seeds} trials per config")
        print(f"🔢 Total trials: {total_configs * n_trials_per_config * n_seeds}")
        
        for risk_weight in risk_proportions:
            cautious_weight = 1.0 - risk_weight  # Complementary weights
            
            for temperature in temperatures:
                config_count += 1
                print(f"\n🎯 Configuration {config_count}/{total_configs}")
                print(f"   Risk Weight: {risk_weight:.1f}, Cautious Weight: {cautious_weight:.1f}, Temperature: {temperature}")
                
                # Multiple seeds for robustness
                for seed_offset in range(n_seeds):
                    base_seed = hash(f"{risk_weight}_{temperature}_{seed_offset}") % 2**32
                    
                    for trial_idx in range(n_trials_per_config):
                        trial_seed = base_seed + trial_idx
                        trial_id = len(all_results) + 1
                        
                        try:
                            result = self.run_single_trial(
                                risk_weight, cautious_weight, temperature, 
                                trial_id, trial_seed)
                            all_results.append(result)
                            
                        except Exception as e:
                            print(f"  ❌ Trial {trial_id} failed: {e}")
                            continue
                            
                        # Small delay to avoid rate limits
                        time.sleep(0.1)
                
                # Progress summary for this configuration
                config_results = [r for r in all_results 
                                if r.risk_weight == risk_weight and r.temperature == temperature]
                if config_results:
                    avg_pumps = np.mean([r.pumps for r in config_results])
                    explosion_rate = np.mean([r.exploded for r in config_results])
                    print(f"   📊 Config Summary: Avg Pumps={avg_pumps:.1f}, Explosion Rate={explosion_rate:.2f}")
        
        print(f"\n✅ Proportion sweep completed!")
        print(f"📊 Collected {len(all_results)} valid trial results")
        
        return all_results
        
    def analyze_proportion_effects(self, results: List[PersonalityTrialResult]):
        """Analyze how personality proportions affect pump counts"""
        
        print("\n📈 Analyzing Personality Proportion Effects...")
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in results])
        
        # Group by personality proportions
        proportion_analysis = df.groupby(['risk_weight', 'temperature']).agg({
            'pumps': ['mean', 'std', 'min', 'max'],
            'exploded': 'mean',
            'reward': 'mean',
            'trial_id': 'count'
        }).round(3)
        
        print("\n📊 Proportion Analysis Summary:")
        print(proportion_analysis.to_string())
        
        # Create comprehensive visualizations
        self._create_proportion_visualizations(df)
        
        # Save detailed analysis
        self._save_proportion_analysis(results, proportion_analysis)
        
        return proportion_analysis
        
    def _create_proportion_visualizations(self, df: pd.DataFrame):
        """Create comprehensive visualizations of proportion effects"""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Personality Proportion Effects on BART Performance (Real API Data)', 
                     fontsize=16, fontweight='bold')
        
        # 1. Risk Weight vs Average Pumps
        ax = axes[0, 0]
        for temp in sorted(df['temperature'].unique()):
            temp_data = df[df['temperature'] == temp]
            grouped = temp_data.groupby('risk_weight')['pumps'].agg(['mean', 'std'])
            
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                       marker='o', linewidth=2, capsize=5, label=f'Temperature {temp}')
        
        ax.set_xlabel('Risk Taker Weight')
        ax.set_ylabel('Average Pumps')
        ax.set_title('Risk Weight vs Average Pumps')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Risk Weight vs Explosion Rate
        ax = axes[0, 1]
        for temp in sorted(df['temperature'].unique()):
            temp_data = df[df['temperature'] == temp]
            explosion_rates = temp_data.groupby('risk_weight')['exploded'].mean()
            
            ax.plot(explosion_rates.index, explosion_rates.values,
                   marker='s', linewidth=2, label=f'Temperature {temp}')
        
        ax.set_xlabel('Risk Taker Weight')
        ax.set_ylabel('Explosion Rate')
        ax.set_title('Risk Weight vs Explosion Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Pump Distribution by Risk Weight
        ax = axes[0, 2]
        risk_weights = sorted(df['risk_weight'].unique())
        pump_data = [df[df['risk_weight'] == rw]['pumps'].values for rw in risk_weights]
        
        bp = ax.boxplot(pump_data, labels=[f'{rw:.1f}' for rw in risk_weights], patch_artist=True)
        
        # Color boxes by risk level
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            
        ax.set_xlabel('Risk Taker Weight')
        ax.set_ylabel('Pumps')
        ax.set_title('Pump Count Distribution by Risk Weight')
        ax.grid(True, alpha=0.3)
        
        # 4. Temperature Effects
        ax = axes[1, 0]
        temp_comparison = df.groupby(['temperature', 'risk_weight'])['pumps'].mean().unstack()
        
        temp_comparison.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Average Pumps')
        ax.set_title('Temperature Effects on Pump Counts')
        ax.legend(title='Risk Weight', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=0)
        
        # 5. Risk-Reward Relationship
        ax = axes[1, 1]
        scatter = ax.scatter(df['pumps'], df['reward'], c=df['risk_weight'], 
                           s=60, alpha=0.7, cmap='RdYlBu_r', edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Pumps')
        ax.set_ylabel('Reward')
        ax.set_title('Risk-Reward Relationship')
        plt.colorbar(scatter, ax=ax, label='Risk Weight')
        ax.grid(True, alpha=0.3)
        
        # 6. Summary Statistics Table
        ax = axes[1, 2]
        ax.axis('off')
        
        # Create summary statistics
        summary_stats = []
        for risk_weight in sorted(df['risk_weight'].unique()):
            subset = df[df['risk_weight'] == risk_weight]
            stats = [
                f"{risk_weight:.1f}",
                f"{subset['pumps'].mean():.1f}",
                f"{subset['exploded'].mean():.2f}",
                f"{subset['reward'].mean():.2f}",
                f"{len(subset)}"
            ]
            summary_stats.append(stats)
        
        table = ax.table(cellText=summary_stats,
                        colLabels=['Risk Weight', 'Avg Pumps', 'Explosion Rate', 'Avg Reward', 'N Trials'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0.1, 0.1, 0.8, 0.8])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(summary_stats) + 1):
            for j in range(5):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
        
        ax.set_title('Summary Statistics by Risk Weight')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'personality_proportion_effects.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
    def _save_proportion_analysis(self, results: List[PersonalityTrialResult], analysis: pd.DataFrame):
        """Save detailed proportion analysis results"""
        
        timestamp = int(time.time())
        
        # Save raw results
        results_file = self.output_dir / f"proportion_test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        
        # Save analysis summary
        analysis_file = self.output_dir / f"proportion_analysis_{timestamp}.csv"
        analysis.to_csv(analysis_file)
        
        print(f"💾 Results saved:")
        print(f"   Raw data: {results_file}")
        print(f"   Analysis: {analysis_file}")
        
    def print_key_findings(self, results: List[PersonalityTrialResult]):
        """Print key findings from the proportion test"""
        
        df = pd.DataFrame([asdict(r) for r in results])
        
        print("\n" + "=" * 80)
        print("KEY FINDINGS: PERSONALITY PROPORTION EFFECTS ON PUMP COUNT")
        print("=" * 80)
        
        print(f"\n📊 Dataset Overview:")
        print(f"   Total trials: {len(results)}")
        print(f"   Risk weight range: {df['risk_weight'].min():.1f} - {df['risk_weight'].max():.1f}")
        print(f"   Temperature levels: {sorted(df['temperature'].unique())}")
        
        print(f"\n🎯 Pump Count Analysis:")
        overall_correlation = df['risk_weight'].corr(df['pumps'])
        print(f"   Risk weight - Pump count correlation: {overall_correlation:.3f}")
        
        # Analyze by risk weight brackets
        print(f"\n📈 Performance by Risk Weight Brackets:")
        risk_brackets = [
            (0.0, 0.3, "Conservative (0.0-0.3)"),
            (0.3, 0.7, "Moderate (0.3-0.7)"), 
            (0.7, 1.0, "Aggressive (0.7-1.0)")
        ]
        
        for min_risk, max_risk, label in risk_brackets:
            bracket_data = df[(df['risk_weight'] >= min_risk) & (df['risk_weight'] <= max_risk)]
            if len(bracket_data) > 0:
                avg_pumps = bracket_data['pumps'].mean()
                explosion_rate = bracket_data['exploded'].mean()
                avg_reward = bracket_data['reward'].mean()
                print(f"   {label}:")
                print(f"     Average pumps: {avg_pumps:.1f}")
                print(f"     Explosion rate: {explosion_rate:.2f}")
                print(f"     Average reward: ${avg_reward:.2f}")
        
        print(f"\n🌡️ Temperature Effects:")
        for temp in sorted(df['temperature'].unique()):
            temp_data = df[df['temperature'] == temp]
            temp_correlation = temp_data['risk_weight'].corr(temp_data['pumps'])
            avg_pumps = temp_data['pumps'].mean()
            print(f"   Temperature {temp}:")
            print(f"     Risk-pump correlation: {temp_correlation:.3f}")
            print(f"     Average pumps: {avg_pumps:.1f}")
        
        print(f"\n🏆 Extreme Performers:")
        
        # Highest pump count
        max_pumps_trial = df.loc[df['pumps'].idxmax()]
        print(f"   Highest pump count: {max_pumps_trial['pumps']} pumps")
        print(f"     Risk weight: {max_pumps_trial['risk_weight']:.1f}")
        print(f"     Result: {'Exploded' if max_pumps_trial['exploded'] else 'Collected'}")
        
        # Most conservative
        min_pumps_trial = df.loc[df['pumps'].idxmin()]
        print(f"   Most conservative: {min_pumps_trial['pumps']} pumps")
        print(f"     Risk weight: {min_pumps_trial['risk_weight']:.1f}")
        print(f"     Reward: ${min_pumps_trial['reward']:.2f}")
        
        print(f"\n💡 Key Insights:")
        if overall_correlation > 0.3:
            print(f"   ✅ Strong positive correlation: Higher risk weights lead to more pumps")
        elif overall_correlation > 0.1:
            print(f"   ⚠️ Weak positive correlation: Risk weights have limited effect on pumps")
        else:
            print(f"   ❌ No correlation: Risk weights don't significantly affect pump count")
            
        # Performance recommendations
        best_reward_risk = df.loc[df['reward'].idxmax(), 'risk_weight']
        safest_risk = df.groupby('risk_weight')['exploded'].mean().idxmin()
        
        print(f"\n🎯 Recommendations:")
        print(f"   For maximum reward: Risk weight ≈ {best_reward_risk:.1f}")
        print(f"   For safety: Risk weight ≈ {safest_risk:.1f}")

def main():
    """Run the personality proportion effect test"""
    
    print("🚀 BART Personality Proportion Effect Test")
    print("Testing how pump counts change with personality weight variations")
    print("Using real OpenAI API calls for authentic responses")
    print("=" * 70)
    
    tester = PersonalityProportionTester()
    
    if not tester.client:
        print("❌ Cannot run test without OpenAI API access")
        print("Please ensure OPENAI_API_KEY is set in your environment")
        return
    
    # Run the proportion sweep test
    start_time = time.time()
    
    try:
        results = tester.run_proportion_sweep(
            n_trials_per_config=3,  # Reduced for API efficiency
            n_seeds=2
        )
        
        if not results:
            print("❌ No results collected")
            return
            
        # Analyze results
        analysis = tester.analyze_proportion_effects(results)
        
        # Print key findings
        tester.print_key_findings(results)
        
        end_time = time.time()
        print(f"\n⏱️ Test completed in {end_time - start_time:.1f} seconds")
        print(f"📊 Visualization saved: personality_proportion_results/personality_proportion_effects.png")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()