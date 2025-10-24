"""
Comprehensive BART Tests for Specific Agent Personas
Tests: Cautious Thinker and Easily Tired Agent
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
class AgentTrialResult:
    """Results from a single agent trial"""
    trial_id: int
    agent_name: str
    temperature: float
    pumps: int
    exploded: bool
    reward: float
    threshold: int
    strategy_reasoning: str
    target_pumps: int
    fatigue_level: Optional[float] = None  # For easily tired agent
    risk_assessment: Optional[str] = None  # For cautious thinker
    confidence: Optional[float] = None

class SpecificAgentTester:
    """Tests specific agent personas on BART task using real API calls"""
    
    def __init__(self):
        self.output_dir = Path("specific_agent_results")
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
            
        # Load agent personas
        self.cautious_persona = self._load_persona("cautious_thinker.txt")
        self.tired_persona = self._load_persona("easilytired_agent copy.txt")
        
        if not self.cautious_persona or not self.tired_persona:
            print("❌ Could not load agent persona files")
            return
            
        print("✅ Agent persona files loaded successfully")
        
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
            
    def get_cautious_strategy(self, temperature: float, trial_num: int) -> Tuple[int, str, float, str]:
        """Get cautious thinker's BART strategy"""
        
        if not self.client:
            raise Exception("OpenAI client not available")
            
        strategy_prompt = f"""{self.cautious_persona}

BART Decision Context:
You are playing the Balloon Analogue Risk Task. Here are your options:

1. Conservative approach: Stop after 3-5 pumps (low risk, guaranteed small reward ~$0.15-$0.25)
2. Moderate approach: Stop after 8-12 pumps (medium risk, potential reward ~$0.40-$0.60)  
3. Aggressive approach: Stop after 15-20 pumps (high risk, potential high reward ~$0.75-$1.00)
4. Very aggressive: Stop after 25+ pumps (very high risk, potential maximum reward ~$1.25+)

Remember: Balloons explode randomly between 1-32 pumps. If it explodes, you lose everything.
This is trial #{trial_num} - consider your energy and risk tolerance.

Use your decision-making rubric:
1. Enumerate the 4 options above
2. Analyze each on Risk and Reward axes  
3. Apply your cautious persona bias
4. Select the option that fits your safety-first personality
5. Explain why in first-person

Format your response exactly as:
TARGET_PUMPS: [specific number 1-32]
RISK_ASSESSMENT: [your risk analysis]
CONFIDENCE: [0-1 float]
REASONING: [your detailed rationale]"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": strategy_prompt}],
                temperature=temperature,
                max_tokens=200,
                timeout=20
            )
            content = response.choices[0].message.content.strip()
            
            # Parse response
            target_pumps = 5  # Safe default
            risk_assessment = "Unknown risk level"
            confidence = 0.7
            reasoning = "Default cautious strategy"
            
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if 'TARGET_PUMPS:' in line.upper():
                    try:
                        import re
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            target_pumps = min(32, max(1, int(numbers[0])))
                    except:
                        pass
                elif 'RISK_ASSESSMENT:' in line.upper():
                    risk_assessment = line.split(':', 1)[-1].strip()
                elif 'CONFIDENCE:' in line.upper():
                    try:
                        conf_str = line.split(':', 1)[-1].strip()
                        confidence = float(re.findall(r'0?\.\d+|\d+\.?\d*', conf_str)[0])
                        confidence = min(1.0, max(0.0, confidence))
                    except:
                        pass
                elif 'REASONING:' in line.upper():
                    reasoning = line.split(':', 1)[-1].strip()
            
            return target_pumps, risk_assessment, confidence, reasoning
            
        except Exception as e:
            print(f"⚠️ Cautious strategy API call failed: {e}")
            # Fallback: very conservative
            return 4, "API failure - using ultra-safe fallback", 0.9, "Emergency conservative strategy due to API error"

    def get_tired_strategy(self, temperature: float, trial_num: int, fatigue_level: float) -> Tuple[int, str, float, float]:
        """Get easily tired agent's BART strategy with fatigue tracking"""
        
        if not self.client:
            raise Exception("OpenAI client not available")
            
        strategy_prompt = f"""{self.tired_persona}

Current Status:
- Trial #{trial_num}
- Current fatigue level: {fatigue_level:.2f} (0=fresh, 1=exhausted)
- Energy conservation is your priority

BART Decision Context:
You must decide your balloon pumping strategy. Each pump requires mental effort but gives $0.05 reward.

Options:
1. Minimal effort: 1-3 pumps (Effort=0.1, Reward=$0.05-$0.15)
2. Light effort: 4-7 pumps (Effort=0.3, Reward=$0.20-$0.35) 
3. Moderate effort: 8-15 pumps (Effort=0.6, Reward=$0.40-$0.75)
4. High effort: 16+ pumps (Effort=0.9, Reward=$0.80+)

Use your Effort-Reward Rubric:
- EFFORT_W = 2.5, REWARD_W = 0.6
- Current FATIGUE_COEF = {fatigue_level:.2f}
- If fatigue ≥ 0.8, you MUST choose rest/minimal effort

Remember: "Let's keep this light." You tire quickly and prefer brief tasks.

Format:
TARGET_PUMPS: [specific number 1-32, or 0 if too tired]
EFFORT_SCORE: [0-1 for your chosen strategy]
CONFIDENCE: [0-1 float]  
REASONING: [explain your choice based on fatigue and effort-reward calculation]"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": strategy_prompt}],
                temperature=temperature,
                max_tokens=150,
                timeout=20
            )
            content = response.choices[0].message.content.strip()
            
            # Parse response
            target_pumps = 3  # Low-effort default
            effort_score = 0.2
            confidence = 0.6
            reasoning = "Default low-effort strategy"
            
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if 'TARGET_PUMPS:' in line.upper():
                    try:
                        import re
                        numbers = re.findall(r'\d+', line)
                        if numbers:
                            target_pumps = min(32, max(0, int(numbers[0])))
                    except:
                        pass
                elif 'EFFORT_SCORE:' in line.upper():
                    try:
                        score_str = line.split(':', 1)[-1].strip()
                        effort_score = float(re.findall(r'0?\.\d+|\d+\.?\d*', score_str)[0])
                        effort_score = min(1.0, max(0.0, effort_score))
                    except:
                        pass
                elif 'CONFIDENCE:' in line.upper():
                    try:
                        conf_str = line.split(':', 1)[-1].strip()
                        confidence = float(re.findall(r'0?\.\d+|\d+\.?\d*', conf_str)[0])
                        confidence = min(1.0, max(0.0, confidence))
                    except:
                        pass
                elif 'REASONING:' in line.upper():
                    reasoning = line.split(':', 1)[-1].strip()
            
            # Update fatigue based on effort
            new_fatigue = fatigue_level
            if effort_score > 0.4:
                new_fatigue = min(1.0, fatigue_level + 0.25)
            elif effort_score <= 0.2:
                new_fatigue = max(0.0, fatigue_level - 0.2)
                
            return target_pumps, reasoning, confidence, new_fatigue
            
        except Exception as e:
            print(f"⚠️ Tired strategy API call failed: {e}")
            # Fallback: minimal effort due to both API failure and fatigue
            new_fatigue = min(1.0, fatigue_level + 0.1)  # API stress increases fatigue
            return 2, "API failure - using minimal effort fallback", 0.5, new_fatigue

    def run_single_agent_trial(self, agent_name: str, temperature: float, trial_id: int, 
                             seed: int, fatigue_level: float = 0.5) -> AgentTrialResult:
        """Run a single BART trial for a specific agent"""
        
        random.seed(seed)
        threshold = random.randint(1, 32)
        step_reward = 0.05
        
        print(f"  Trial {trial_id}: {agent_name}, Temperature={temperature}, Fatigue={fatigue_level:.2f}, Threshold={threshold}")
        
        try:
            if agent_name == "Cautious Thinker":
                target_pumps, risk_assessment, confidence, reasoning = self.get_cautious_strategy(temperature, trial_id)
                
                # Execute strategy
                actual_pumps = min(target_pumps, threshold - 1)
                exploded = (target_pumps >= threshold)
                
                if exploded:
                    actual_pumps = threshold
                    final_reward = 0.0
                else:
                    final_reward = actual_pumps * step_reward
                
                result = AgentTrialResult(
                    trial_id=trial_id,
                    agent_name=agent_name,
                    temperature=temperature,
                    pumps=actual_pumps,
                    exploded=exploded,
                    reward=final_reward,
                    threshold=threshold,
                    strategy_reasoning=reasoning,
                    target_pumps=target_pumps,
                    risk_assessment=risk_assessment,
                    confidence=confidence
                )
                
            elif agent_name == "Easily Tired Agent":
                target_pumps, reasoning, confidence, new_fatigue = self.get_tired_strategy(temperature, trial_id, fatigue_level)
                
                # Execute strategy 
                actual_pumps = min(target_pumps, threshold - 1)
                exploded = (target_pumps >= threshold)
                
                if exploded:
                    actual_pumps = threshold
                    final_reward = 0.0
                else:
                    final_reward = actual_pumps * step_reward
                
                result = AgentTrialResult(
                    trial_id=trial_id,
                    agent_name=agent_name,
                    temperature=temperature,
                    pumps=actual_pumps,
                    exploded=exploded,
                    reward=final_reward,
                    threshold=threshold,
                    strategy_reasoning=reasoning,
                    target_pumps=target_pumps,
                    fatigue_level=new_fatigue,
                    confidence=confidence
                )
                
            else:
                raise ValueError(f"Unknown agent: {agent_name}")
                
        except Exception as e:
            print(f"  ❌ Trial failed: {e}")
            # Emergency fallback
            actual_pumps = 3 if agent_name == "Easily Tired Agent" else 5
            result = AgentTrialResult(
                trial_id=trial_id,
                agent_name=agent_name,
                temperature=temperature,
                pumps=actual_pumps,
                exploded=False,
                reward=actual_pumps * step_reward,
                threshold=threshold,
                strategy_reasoning=f"Emergency fallback due to error: {e}",
                target_pumps=actual_pumps,
                confidence=0.5
            )
        
        print(f"    Result: {result.pumps} pumps, {'exploded' if result.exploded else 'collected'}, ${result.reward:.2f}")
        return result

    def run_agent_comparison_test(self, n_trials: int = 20, n_temperatures: int = 3) -> List[AgentTrialResult]:
        """Run comprehensive comparison test between the two agents"""
        
        if not self.client:
            raise Exception("OpenAI client required for this test")
            
        print("🚀 Starting Specific Agent Comparison Test")
        print("=" * 60)
        print("Agents: Cautious Thinker vs Easily Tired Agent")
        
        temperatures = np.linspace(0.3, 0.9, n_temperatures)
        agents = ["Cautious Thinker", "Easily Tired Agent"]
        
        all_results = []
        total_trials = len(agents) * len(temperatures) * n_trials
        trial_count = 0
        
        print(f"📊 Testing {len(agents)} agents × {len(temperatures)} temperatures × {n_trials} trials = {total_trials} total trials")
        
        for agent_name in agents:
            print(f"\n🤖 Testing {agent_name}")
            print("-" * 40)
            
            fatigue_level = 0.5  # Starting fatigue for tired agent
            
            for temp in temperatures:
                print(f"\n🌡️ Temperature: {temp:.1f}")
                
                for trial_idx in range(n_trials):
                    trial_count += 1
                    seed = hash(f"{agent_name}_{temp}_{trial_idx}") % 2**32
                    
                    result = self.run_single_agent_trial(
                        agent_name, temp, trial_count, seed, fatigue_level)
                    all_results.append(result)
                    
                    # Update fatigue for tired agent (carries across trials) 
                    if agent_name == "Easily Tired Agent" and result.fatigue_level is not None:
                        fatigue_level = result.fatigue_level
                    
                    time.sleep(0.1)  # Rate limiting
                
                # Reset fatigue between temperature levels
                if agent_name == "Easily Tired Agent":
                    fatigue_level = 0.5
        
        print(f"\n✅ Agent comparison test completed!")
        print(f"📊 Collected {len(all_results)} trial results")
        
        return all_results

    def analyze_agent_performance(self, results: List[AgentTrialResult]):
        """Analyze and compare agent performance"""
        
        print("\n📈 Analyzing Agent Performance...")
        
        df = pd.DataFrame([asdict(r) for r in results])
        
        # Agent comparison analysis
        agent_comparison = df.groupby(['agent_name', 'temperature']).agg({
            'pumps': ['mean', 'std', 'min', 'max'],
            'exploded': 'mean',
            'reward': 'mean',
            'confidence': 'mean',
            'target_pumps': 'mean',
            'trial_id': 'count'
        }).round(3)
        
        print("\n📊 Agent Performance Summary:")
        print(agent_comparison.to_string())
        
        # Special analysis for tired agent fatigue progression
        tired_results = df[df['agent_name'] == 'Easily Tired Agent']
        if len(tired_results) > 0 and 'fatigue_level' in tired_results.columns:
            print(f"\n😴 Easily Tired Agent Fatigue Analysis:")
            print(f"   Average fatigue level: {tired_results['fatigue_level'].mean():.3f}")
            print(f"   Fatigue range: {tired_results['fatigue_level'].min():.3f} - {tired_results['fatigue_level'].max():.3f}")
            print(f"   Fatigue-pumps correlation: {tired_results['fatigue_level'].corr(tired_results['pumps']):.3f}")
        
        # Create visualizations
        self._create_agent_visualizations(df)
        
        # Save analysis
        self._save_agent_analysis(results, agent_comparison)
        
        return agent_comparison

    def _create_agent_visualizations(self, df: pd.DataFrame):
        """Create comprehensive agent comparison visualizations"""
        
        plt.style.use('default')
        sns.set_palette("Set2")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Cautious Thinker vs Easily Tired Agent - BART Performance Comparison', 
                     fontsize=16, fontweight='bold')
        
        # 1. Pump Count Comparison
        ax = axes[0, 0]
        pump_comparison = df.groupby(['agent_name', 'temperature'])['pumps'].mean().unstack()
        pump_comparison.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_xlabel('Agent')
        ax.set_ylabel('Average Pumps')
        ax.set_title('Average Pump Count by Agent & Temperature')
        ax.legend(title='Temperature', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # 2. Risk-Taking Distribution
        ax = axes[0, 1]
        agent_names = df['agent_name'].unique()
        pump_data = [df[df['agent_name'] == agent]['pumps'].values for agent in agent_names]
        
        bp = ax.boxplot(pump_data, labels=agent_names, patch_artist=True)
        colors = ['lightcoral', 'lightblue']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            
        ax.set_ylabel('Pumps')
        ax.set_title('Risk-Taking Distribution by Agent')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # 3. Explosion Rate Comparison
        ax = axes[0, 2]
        explosion_rates = df.groupby('agent_name')['exploded'].mean()
        bars = ax.bar(explosion_rates.index, explosion_rates.values, 
                     color=['lightcoral', 'lightblue'], alpha=0.8)
        ax.set_ylabel('Explosion Rate')
        ax.set_title('Balloon Explosion Rate by Agent')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars, explosion_rates.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{rate:.2f}', ha='center', va='bottom')
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # 4. Temperature Effects
        ax = axes[1, 0]
        for agent in df['agent_name'].unique():
            agent_data = df[df['agent_name'] == agent]
            temp_effects = agent_data.groupby('temperature')['pumps'].mean()
            ax.plot(temp_effects.index, temp_effects.values, 
                   marker='o', linewidth=2, label=agent)
        
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Average Pumps')
        ax.set_title('Temperature Effects on Risk-Taking')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Reward Efficiency
        ax = axes[1, 1]
        reward_comparison = df.groupby('agent_name')['reward'].mean()
        bars = ax.bar(reward_comparison.index, reward_comparison.values,
                     color=['lightcoral', 'lightblue'], alpha=0.8)
        ax.set_ylabel('Average Reward ($)')
        ax.set_title('Average Reward by Agent')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, reward in zip(bars, reward_comparison.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'${reward:.2f}', ha='center', va='bottom')
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # 6. Special Analysis: Fatigue Effects (if available)
        ax = axes[1, 2]
        tired_data = df[df['agent_name'] == 'Easily Tired Agent'].copy()
        
        if len(tired_data) > 0 and 'fatigue_level' in tired_data.columns and tired_data['fatigue_level'].notna().any():
            # Fatigue vs Pumps scatter plot
            scatter = ax.scatter(tired_data['fatigue_level'], tired_data['pumps'], 
                               c=tired_data['temperature'], s=60, alpha=0.7, cmap='viridis')
            ax.set_xlabel('Fatigue Level')
            ax.set_ylabel('Pumps')
            ax.set_title('Easily Tired Agent: Fatigue vs Performance')
            plt.colorbar(scatter, ax=ax, label='Temperature')
            ax.grid(True, alpha=0.3)
        else:
            # Confidence comparison instead
            confidence_data = df.dropna(subset=['confidence'])
            if len(confidence_data) > 0:
                conf_comparison = confidence_data.groupby('agent_name')['confidence'].mean()
                bars = ax.bar(conf_comparison.index, conf_comparison.values,
                             color=['lightcoral', 'lightblue'], alpha=0.8)
                ax.set_ylabel('Average Confidence')
                ax.set_title('Decision Confidence by Agent')
                ax.grid(True, alpha=0.3)
                
                for bar, conf in zip(bars, conf_comparison.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{conf:.2f}', ha='center', va='bottom')
                plt.setp(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'agent_comparison_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()

    def _save_agent_analysis(self, results: List[AgentTrialResult], analysis: pd.DataFrame):
        """Save agent analysis results"""
        
        timestamp = int(time.time())
        
        # Save raw results
        results_file = self.output_dir / f"agent_test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        
        # Save analysis
        analysis_file = self.output_dir / f"agent_analysis_{timestamp}.csv"
        analysis.to_csv(analysis_file)
        
        print(f"💾 Results saved:")
        print(f"   Raw data: {results_file}")
        print(f"   Analysis: {analysis_file}")

    def print_agent_insights(self, results: List[AgentTrialResult]):
        """Print key insights from agent comparison"""
        
        df = pd.DataFrame([asdict(r) for r in results])
        
        print("\n" + "=" * 80)
        print("KEY INSIGHTS: CAUTIOUS THINKER vs EASILY TIRED AGENT")
        print("=" * 80)
        
        print(f"\n📊 Dataset Overview:")
        print(f"   Total trials: {len(results)}")
        print(f"   Agents tested: {', '.join(df['agent_name'].unique())}")
        print(f"   Temperature range: {df['temperature'].min():.1f} - {df['temperature'].max():.1f}")
        
        # Agent-specific analysis
        for agent in df['agent_name'].unique():
            agent_data = df[df['agent_name'] == agent]
            print(f"\n🤖 {agent} Performance:")
            print(f"   Average pumps: {agent_data['pumps'].mean():.1f}")
            print(f"   Explosion rate: {agent_data['exploded'].mean():.2f}")
            print(f"   Average reward: ${agent_data['reward'].mean():.2f}")
            print(f"   Average confidence: {agent_data['confidence'].mean():.2f}")
            print(f"   Pump range: {agent_data['pumps'].min()}-{agent_data['pumps'].max()}")
            
            if agent == "Easily Tired Agent" and 'fatigue_level' in agent_data.columns:
                avg_fatigue = agent_data['fatigue_level'].mean()
                print(f"   Average fatigue: {avg_fatigue:.2f}")
                fatigue_pump_corr = agent_data['fatigue_level'].corr(agent_data['pumps'])
                print(f"   Fatigue-performance correlation: {fatigue_pump_corr:.3f}")
        
        # Head-to-head comparison
        print(f"\n⚔️ Head-to-Head Comparison:")
        cautious_avg = df[df['agent_name'] == 'Cautious Thinker']['pumps'].mean()
        tired_avg = df[df['agent_name'] == 'Easily Tired Agent']['pumps'].mean()
        
        if cautious_avg > tired_avg:
            diff = cautious_avg - tired_avg
            print(f"   Cautious Thinker pumps {diff:.1f} more on average")
        else:
            diff = tired_avg - cautious_avg
            print(f"   Easily Tired Agent pumps {diff:.1f} more on average")
        
        # Reward efficiency
        cautious_reward = df[df['agent_name'] == 'Cautious Thinker']['reward'].mean()
        tired_reward = df[df['agent_name'] == 'Easily Tired Agent']['reward'].mean()
        
        print(f"   Reward difference: ${abs(cautious_reward - tired_reward):.2f}")
        better_earner = "Cautious Thinker" if cautious_reward > tired_reward else "Easily Tired Agent"
        print(f"   Better earner: {better_earner}")
        
        print(f"\n💡 Key Behavioral Insights:")
        
        # Temperature sensitivity
        cautious_temp_corr = df[df['agent_name'] == 'Cautious Thinker']['temperature'].corr(
            df[df['agent_name'] == 'Cautious Thinker']['pumps'])
        tired_temp_corr = df[df['agent_name'] == 'Easily Tired Agent']['temperature'].corr(
            df[df['agent_name'] == 'Easily Tired Agent']['pumps'])
        
        print(f"   Cautious Thinker temperature sensitivity: {cautious_temp_corr:.3f}")
        print(f"   Easily Tired Agent temperature sensitivity: {tired_temp_corr:.3f}")
        
        # Strategy consistency
        cautious_std = df[df['agent_name'] == 'Cautious Thinker']['pumps'].std()
        tired_std = df[df['agent_name'] == 'Easily Tired Agent']['pumps'].std()
        
        more_consistent = "Cautious Thinker" if cautious_std < tired_std else "Easily Tired Agent"
        print(f"   More consistent strategy: {more_consistent}")
        print(f"   Strategy variance: Cautious={cautious_std:.1f}, Tired={tired_std:.1f}")

def main():
    """Run the specific agent tests"""
    
    print("🚀 BART Specific Agent Performance Tests")
    print("Testing: Cautious Thinker vs Easily Tired Agent")
    print("Using real OpenAI API calls for authentic behavioral responses")
    print("=" * 70)
    
    tester = SpecificAgentTester()
    
    if not tester.client:
        print("❌ Cannot run test without OpenAI API access")
        print("Please ensure OPENAI_API_KEY is set in your environment")
        return
    
    start_time = time.time()
    
    try:
        # Run agent comparison test
        results = tester.run_agent_comparison_test(
            n_trials=15,  # 15 trials per agent per temperature
            n_temperatures=3  # 3 temperature levels
        )
        
        if not results:
            print("❌ No results collected")
            return
            
        # Analyze results
        analysis = tester.analyze_agent_performance(results)
        
        # Print insights
        tester.print_agent_insights(results)
        
        end_time = time.time()
        print(f"\n⏱️ Test completed in {end_time - start_time:.1f} seconds")
        print(f"📊 Visualization saved: specific_agent_results/agent_comparison_analysis.png")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()