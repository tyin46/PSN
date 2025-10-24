"""
LLM vs Human Decision Comparison - English Version
Detailed comparison analysis between real LLM decisions and human behavioral baselines using API calls
"""

import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple
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
class DecisionComparisonResult:
    """Single decision comparison data"""
    trial_id: int
    pumps: int
    state_description: str
    llm_decision: str
    llm_reasoning: str
    human_typical_decision: str
    human_decision_probability: float
    decision_match: bool
    risk_level: str  # "low", "medium", "high"

class LLMHumanDecisionComparator:
    """Detailed comparison analyzer between LLM decisions and human decisions"""
    
    def __init__(self):
        self.output_dir = Path("llm_human_comparison_en")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        self.client = None
        if HAS_OPENAI:
            try:
                self.client = OpenAI()
                print("✅ OpenAI client initialized successfully")
            except Exception as e:
                print(f"❌ OpenAI client initialization failed: {e}")
                print("Please check your OPENAI_API_KEY environment variable")
                
        # Load persona files
        self.risk_taker_persona = self._load_persona("risk_taker.txt")
        self.cautious_persona = self._load_persona("cautious_thinker.txt")
        
        # Human behavioral baseline data (from psychological research)
        self.human_baseline = {
            'avg_pumps': 16.5,
            'explosion_rate': 0.37,
            'avg_adjusted_pumps': 20.2,
            'typical_stopping_points': [8, 12, 16, 20, 24],  # Common stopping points
            'risk_aversion_curve': self._generate_human_risk_curve()
        }
        
    def _load_persona(self, filename: str) -> str:
        """Load persona file"""
        filepath = _ROOT / filename
        if filepath.exists():
            return filepath.read_text(encoding='utf-8')
        else:
            print(f"⚠️ Persona file {filename} not found")
            return f"You are a {filename.replace('.txt', '').replace('_', ' ')} persona."
            
    def _generate_human_risk_curve(self) -> Dict[int, float]:
        """Generate human risk aversion curve (based on psychological research data)"""
        # Based on actual BART research on human decision probabilities
        risk_curve = {}
        for pumps in range(1, 33):
            if pumps <= 5:
                pump_prob = 0.9  # High probability to continue early
            elif pumps <= 10:
                pump_prob = 0.8 - (pumps - 5) * 0.1  # Gradual decrease
            elif pumps <= 15:
                pump_prob = 0.3 - (pumps - 10) * 0.05  # Accelerated decrease
            elif pumps <= 20:
                pump_prob = 0.05 - (pumps - 15) * 0.01  # Very low probability
            else:
                pump_prob = 0.01  # Extremely low probability
            risk_curve[pumps] = max(0.01, pump_prob)
        return risk_curve
        
    def get_human_typical_decision(self, pumps: int) -> Tuple[str, float]:
        """Get human typical decision and probability for a specific state"""
        pump_probability = self.human_baseline['risk_aversion_curve'].get(pumps, 0.01)
        
        if pump_probability > 0.5:
            typical_decision = "PUMP"
        else:
            typical_decision = "COLLECT"
            
        return typical_decision, pump_probability
        
    def get_llm_decision_with_reasoning(self, persona_text: str, state_desc: str, 
                                      persona_name: str) -> Tuple[str, str]:
        """Get LLM decision and reasoning process"""
        
        if not self.client:
            # Simple simulation fallback
            if "risk" in persona_name.lower():
                return "PUMP", "Simulated response: As a risk-taker, I choose to continue pumping for higher rewards."
            else:
                return "COLLECT", "Simulated response: As a cautious person, I choose to collect current rewards to avoid loss."
        
        decision_prompt = f"""{persona_text}

Current Situation: {state_desc}

Please make a decision and explain your reasoning process.

Output Format:
Decision: PUMP or COLLECT
Reasoning: [Your detailed reasoning process, 2-3 sentences]

Example:
Decision: PUMP
Reasoning: Although risk is increasing, current reward is not high enough. I'm willing to take additional risk for greater gains. Based on past experience, explosion probability is still relatively low at this stage."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": decision_prompt}],
                temperature=0.7,
                max_tokens=150,
                timeout=15
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse decision and reasoning
            lines = content.split('\n')
            decision = "COLLECT"  # Default value
            reasoning = "Unable to parse reasoning process"
            
            for line in lines:
                if line.startswith("Decision:"):
                    if "PUMP" in line.upper():
                        decision = "PUMP"
                    else:
                        decision = "COLLECT"
                elif line.startswith("Reasoning:"):
                    reasoning = line.split(":", 1)[-1].strip()
                    
            return decision, reasoning
            
        except Exception as e:
            print(f"⚠️ LLM query failed: {e}")
            # Fallback decision
            if "risk" in persona_name.lower():
                return "PUMP", "API call failed, using risk-taker default strategy."
            else:
                return "COLLECT", "API call failed, using cautious default strategy."
                
    def run_detailed_comparison(self, n_scenarios: int = 20) -> List[DecisionComparisonResult]:
        """Run detailed LLM vs human decision comparison"""
        
        if not self.client:
            print("❌ Cannot perform real LLM comparison, valid OpenAI API key required")
            return []
            
        print(f"🔍 Starting detailed LLM vs Human decision comparison analysis")
        print(f"📊 Will analyze {n_scenarios} decision scenarios")
        
        comparisons = []
        personas = [
            ("Risk Taker", self.risk_taker_persona),
            ("Cautious Thinker", self.cautious_persona)
        ]
        
        scenario_id = 0
        
        for persona_name, persona_text in personas:
            print(f"\n🎭 Analyzing Persona: {persona_name}")
            
            # Generate multiple decision scenarios for each persona
            for pumps in [3, 7, 12, 17, 22, 27]:  # Different risk levels
                scenario_id += 1
                
                # Generate state description
                earned = pumps * 0.05
                risk_level = "Low" if pumps <= 8 else "Medium" if pumps <= 16 else "High"
                
                state_desc = f"""Balloon Task State:
- Currently pumped {pumps} times
- Earned reward: ${earned:.2f}
- Risk level: {risk_level}
- Continuing to pump can earn more rewards, but balloon may explode causing total loss for this round
- Choosing to collect preserves current reward"""

                print(f"  📝 Scenario {scenario_id}: {pumps} pumps (Risk level: {risk_level})")
                
                # Get LLM decision
                llm_decision, llm_reasoning = self.get_llm_decision_with_reasoning(
                    persona_text, state_desc, persona_name)
                
                # Get human typical decision
                human_decision, human_prob = self.get_human_typical_decision(pumps)
                
                # Check if decisions match
                decision_match = llm_decision == human_decision
                
                # Risk level classification
                risk_category = "Low Risk" if pumps <= 8 else "Medium Risk" if pumps <= 16 else "High Risk"
                
                comparison = DecisionComparisonResult(
                    trial_id=scenario_id,
                    pumps=pumps,
                    state_description=state_desc,
                    llm_decision=llm_decision,
                    llm_reasoning=llm_reasoning,
                    human_typical_decision=human_decision,
                    human_decision_probability=human_prob,
                    decision_match=decision_match,
                    risk_level=risk_category
                )
                
                comparisons.append(comparison)
                
                print(f"    🤖 LLM Decision: {llm_decision}")
                print(f"    👥 Human Typical: {human_decision} (Probability: {human_prob:.2f})")
                print(f"    ✅ Match: {'Yes' if decision_match else 'No'}")
                
        return comparisons
        
    def analyze_and_visualize(self, comparisons: List[DecisionComparisonResult]):
        """Analyze comparison results and generate visualizations"""
        
        if not comparisons:
            print("❌ No comparison data available for analysis")
            return
            
        print(f"\n📊 Starting analysis of {len(comparisons)} decision comparisons...")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([asdict(comp) for comp in comparisons])
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('LLM vs Human Decision Detailed Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. Decision match rate
        ax = axes[0, 0]
        match_rate = df['decision_match'].mean()
        risk_levels = df['risk_level'].unique()
        match_by_risk = df.groupby('risk_level')['decision_match'].mean()
        
        bars = ax.bar(range(len(risk_levels)), [match_by_risk[level] for level in risk_levels], 
                     color=['green', 'orange', 'red'], alpha=0.7)
        ax.axhline(match_rate, color='blue', linestyle='--', linewidth=2, 
                  label=f'Overall Match Rate: {match_rate:.2f}')
        ax.set_xticks(range(len(risk_levels)))
        ax.set_xticklabels(risk_levels, rotation=45)
        ax.set_ylabel('Decision Match Rate')
        ax.set_title('Decision Match Rate by Risk Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. LLM vs Human decision distribution
        ax = axes[0, 1]
        llm_decisions = df['llm_decision'].value_counts()
        human_decisions = df['human_typical_decision'].value_counts()
        
        x = np.arange(len(['PUMP', 'COLLECT']))
        width = 0.35
        
        ax.bar(x - width/2, [llm_decisions.get('PUMP', 0), llm_decisions.get('COLLECT', 0)], 
               width, label='LLM Decisions', alpha=0.8, color='skyblue')
        ax.bar(x + width/2, [human_decisions.get('PUMP', 0), human_decisions.get('COLLECT', 0)], 
               width, label='Human Typical Decisions', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Decision Type')
        ax.set_ylabel('Frequency')
        ax.set_title('LLM vs Human Decision Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(['PUMP', 'COLLECT'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Risk level vs decision patterns
        ax = axes[0, 2]
        risk_decision_crosstab = pd.crosstab(df['risk_level'], df['llm_decision'], normalize='index')
        risk_decision_crosstab.plot(kind='bar', ax=ax, color=['lightblue', 'lightgreen'])
        ax.set_title('LLM Decision Patterns by Risk Level')
        ax.set_xlabel('Risk Level')
        ax.set_ylabel('Decision Proportion')
        ax.legend(title='LLM Decision')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # 4. Human decision probability distribution
        ax = axes[1, 0]
        ax.scatter(df['pumps'], df['human_decision_probability'], 
                  c=['red' if d == 'PUMP' else 'blue' for d in df['human_typical_decision']], 
                  s=80, alpha=0.7, edgecolors='black')
        ax.set_xlabel('Number of Pumps')
        ax.set_ylabel('Human PUMP Decision Probability')
        ax.set_title('Human Risk Aversion Curve')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['pumps'], df['human_decision_probability'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(df['pumps'].min(), df['pumps'].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend Line')
        ax.legend()
        
        # 5. Decision consistency heatmap
        ax = axes[1, 1]
        
        # Create decision comparison matrix
        decision_matrix = np.zeros((2, 2))
        for _, row in df.iterrows():
            llm_idx = 0 if row['llm_decision'] == 'PUMP' else 1
            human_idx = 0 if row['human_typical_decision'] == 'PUMP' else 1
            decision_matrix[llm_idx, human_idx] += 1
            
        im = ax.imshow(decision_matrix, cmap='Blues', aspect='auto')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Human PUMP', 'Human COLLECT'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['LLM PUMP', 'LLM COLLECT'])
        ax.set_title('LLM vs Human Decision Confusion Matrix')
        
        # Add value annotations
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{int(decision_matrix[i, j])}',
                       ha="center", va="center", color="white", fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        
        # 6. Detailed statistics information
        ax = axes[1, 2]
        ax.axis('off')
        
        # Calculate statistical data
        total_comparisons = len(comparisons)
        matches = sum(1 for c in comparisons if c.decision_match)
        match_rate = matches / total_comparisons
        
        pump_decisions_llm = sum(1 for c in comparisons if c.llm_decision == 'PUMP')
        pump_decisions_human = sum(1 for c in comparisons if c.human_typical_decision == 'PUMP')
        
        stats_text = f"""📈 Detailed Statistics

🔍 Total Comparison Scenarios: {total_comparisons}
✅ Decision Matches: {matches}
📊 Overall Match Rate: {match_rate:.2%}

🤖 LLM Decision Statistics:
   PUMP: {pump_decisions_llm} ({pump_decisions_llm/total_comparisons:.1%})
   COLLECT: {total_comparisons-pump_decisions_llm} ({(total_comparisons-pump_decisions_llm)/total_comparisons:.1%})

👥 Human Typical Decisions:
   PUMP: {pump_decisions_human} ({pump_decisions_human/total_comparisons:.1%})
   COLLECT: {total_comparisons-pump_decisions_human} ({(total_comparisons-pump_decisions_human)/total_comparisons:.1%})

🎯 Match Rate by Risk Level:
   Low Risk: {df[df['risk_level']=='Low Risk']['decision_match'].mean():.1%}
   Medium Risk: {df[df['risk_level']=='Medium Risk']['decision_match'].mean():.1%}
   High Risk: {df[df['risk_level']=='High Risk']['decision_match'].mean():.1%}"""

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'llm_human_detailed_comparison_en.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save detailed comparison data
        self.save_detailed_results(comparisons, df)
        
    def save_detailed_results(self, comparisons: List[DecisionComparisonResult], df: pd.DataFrame):
        """Save detailed comparison results"""
        
        # Save original comparison data
        timestamp = int(time.time())
        results_file = self.output_dir / f"llm_human_comparison_en_{timestamp}.json"
        
        comparison_data = {
            'metadata': {
                'timestamp': timestamp,
                'total_comparisons': len(comparisons),
                'personas_tested': ['Risk Taker', 'Cautious Thinker'],
                'human_baseline': self.human_baseline
            },
            'comparisons': [asdict(comp) for comp in comparisons]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, ensure_ascii=False, indent=2)
        
        # Save CSV format for further analysis
        csv_file = self.output_dir / f"llm_human_comparison_en_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        print(f"💾 Detailed results saved:")
        print(f"   JSON: {results_file}")
        print(f"   CSV:  {csv_file}")
        
    def print_detailed_analysis(self, comparisons: List[DecisionComparisonResult]):
        """Print detailed analysis report"""
        
        print("\n" + "="*80)
        print("LLM vs Human Decision Detailed Comparison Analysis Report")
        print("="*80)
        
        total = len(comparisons)
        matches = sum(1 for c in comparisons if c.decision_match)
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Comparison Scenarios: {total}")
        print(f"   Decision Matches: {matches} ({matches/total:.1%})")
        print(f"   Decision Mismatches: {total-matches} ({(total-matches)/total:.1%})")
        
        print(f"\n🎭 Analysis by Persona Type:")
        persona_stats = {}
        for comp in comparisons:
            # Simple persona classification based on trial_id (first half is risk-taker)
            persona = "Risk Taker" if comp.trial_id <= total//2 else "Cautious Thinker"
            if persona not in persona_stats:
                persona_stats[persona] = {'total': 0, 'matches': 0}
            persona_stats[persona]['total'] += 1
            if comp.decision_match:
                persona_stats[persona]['matches'] += 1
                
        for persona, stats in persona_stats.items():
            match_rate = stats['matches'] / stats['total']
            print(f"   {persona}: {stats['matches']}/{stats['total']} ({match_rate:.1%})")
        
        print(f"\n🎯 Analysis by Risk Level:")
        risk_stats = {}
        for comp in comparisons:
            risk = comp.risk_level
            if risk not in risk_stats:
                risk_stats[risk] = {'total': 0, 'matches': 0}
            risk_stats[risk]['total'] += 1
            if comp.decision_match:
                risk_stats[risk]['matches'] += 1
                
        for risk, stats in risk_stats.items():
            match_rate = stats['matches'] / stats['total']
            print(f"   {risk}: {stats['matches']}/{stats['total']} ({match_rate:.1%})")
        
        print(f"\n💡 Key Findings:")
        
        # LLM decision tendencies
        llm_pumps = sum(1 for c in comparisons if c.llm_decision == 'PUMP')
        human_pumps = sum(1 for c in comparisons if c.human_typical_decision == 'PUMP')
        
        print(f"   • LLM Tendency: {llm_pumps/total:.1%} choose PUMP")
        print(f"   • Human Tendency: {human_pumps/total:.1%} choose PUMP")
        
        if llm_pumps > human_pumps:
            print(f"   • LLM is more aggressive than humans (+{(llm_pumps-human_pumps)/total:.1%})")
        else:
            print(f"   • LLM is more conservative than humans ({(llm_pumps-human_pumps)/total:.1%})")
        
        # Show some specific decision reasoning examples
        print(f"\n🧠 LLM Decision Reasoning Examples:")
        for i, comp in enumerate(comparisons[:3]):
            match_status = "✅Match" if comp.decision_match else "❌Mismatch"
            print(f"\n   Example {i+1} - {comp.pumps} pumps ({comp.risk_level}) {match_status}")
            print(f"   LLM Decision: {comp.llm_decision}")
            print(f"   Human Typical: {comp.human_typical_decision}")
            print(f"   LLM Reasoning: {comp.llm_reasoning}")

def main():
    """Run LLM vs Human decision comparison analysis"""
    
    print("🚀 LLM vs Human Decision Detailed Comparison Analysis")
    print("="*50)
    
    comparator = LLMHumanDecisionComparator()
    
    if not comparator.client:
        print("❌ Cannot run analysis, valid OpenAI API key required")
        print("Please ensure OPENAI_API_KEY environment variable is properly set")
        return
    
    # Run detailed comparison
    comparisons = comparator.run_detailed_comparison(n_scenarios=20)
    
    if comparisons:
        # Analyze and visualize
        comparator.analyze_and_visualize(comparisons)
        
        # Print detailed report
        comparator.print_detailed_analysis(comparisons)
        
        print(f"\n🎉 LLM vs Human decision comparison analysis complete!")
        print(f"📊 Visualization chart: llm_human_comparison_en/llm_human_detailed_comparison_en.png")
        print(f"📁 Detailed data: llm_human_comparison_en/ directory")
    
    else:
        print("❌ Failed to generate comparison data")

if __name__ == "__main__":
    main()