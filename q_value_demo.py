"""
Q-Value Evolution Demonstration
Shows how Q(A) and Q(B) change during PRLT with different personality types
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prlt_personality_proportions import ParamGenerator, AgentParams, PRLTSimulator

def demo_q_values():
    """Demonstrate Q-value evolution for different agent types"""
    
    # Define three different agent types
    agents = {
        'Fast Learner': AgentParams(learning_rate=0.8, epsilon=0.1, perseveration=0.0, 
                                   decision_noise=0.0, patience=15, rationale='Fast learning'),
        'Cautious': AgentParams(learning_rate=0.2, epsilon=0.05, perseveration=0.3, 
                               decision_noise=0.0, patience=25, rationale='Cautious learning'), 
        'Explorer': AgentParams(learning_rate=0.4, epsilon=0.3, perseveration=0.1, 
                               decision_noise=0.0, patience=20, rationale='High exploration')
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (agent_name, params) in enumerate(agents.items()):
        sim = PRLTSimulator(params, rng_seed=42)
        result, history = sim.run(pre_reversal_trials=150)
        
        # Extract data
        trials = [h['trial'] for h in history]
        QA_vals = [h['QA'] for h in history]
        QB_vals = [h['QB'] for h in history]
        phases = [h['phase'] for h in history]
        
        # Find reversal point
        reversal_trial = None
        for j, phase in enumerate(phases):
            if phase == 'post':
                reversal_trial = trials[j]
                break
        
        ax = axes[i]
        
        # Plot Q-values
        ax.plot(trials, QA_vals, 'b-', linewidth=2, label='Q(A)', alpha=0.8)
        ax.plot(trials, QB_vals, 'r-', linewidth=2, label='Q(B)', alpha=0.8)
        
        # Mark reversal
        if reversal_trial:
            ax.axvline(reversal_trial, color='orange', linestyle='--', alpha=0.7, 
                      label='Reversal')
        
        # Mark convergence points
        if result.pre_rev_trials_to_converge:
            ax.axvline(result.pre_rev_trials_to_converge, color='green', 
                      linestyle=':', alpha=0.7, label='Pre-Converge')
        
        if result.post_rev_trials_to_switch and reversal_trial:
            switch_trial = reversal_trial + result.post_rev_trials_to_switch
            if switch_trial <= max(trials):
                ax.axvline(switch_trial, color='purple', linestyle=':', 
                          alpha=0.7, label='Post-Switch')
        
        ax.set_xlabel('Trial')
        ax.set_ylabel('Q-Value')
        ax.set_title(f'{agent_name}\n(α={params.learning_rate}, ε={params.epsilon})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add performance text
        perf_text = f"Pre: {result.pre_rev_trials_to_converge} trials\n"
        perf_text += f"Post: {result.post_rev_trials_to_switch} trials"
        ax.text(0.02, 0.98, perf_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('Q-Value Evolution in PRLT: Different Agent Types', y=1.02, fontsize=14)
    
    # Save and show
    plt.savefig(ROOT / 'q_value_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print interpretation
    print("\n" + "="*60)
    print("Q-VALUE INTERPRETATION")
    print("="*60)
    print("Q(A) and Q(B) represent the agent's learned estimates of reward probability for each option.")
    print()
    print("Key Observations:")
    print("1. PRE-REVERSAL (trials 1 to reversal point):")
    print("   - Task: A=75% reward, B=25% reward")
    print("   - Expected: Q(A) should rise above Q(B)")
    print("   - Fast Learner: Quick rise due to high learning rate (α=0.8)")
    print("   - Cautious: Slower rise due to low learning rate (α=0.2) + perseveration")  
    print("   - Explorer: Noisy learning due to high exploration (ε=0.3)")
    print()
    print("2. REVERSAL POINT (orange dashed line):")
    print("   - Task switches: A=25% reward, B=75% reward")
    print("   - Agent doesn't know about the switch!")
    print()
    print("3. POST-REVERSAL (after reversal to end):")
    print("   - Expected: Q(B) should eventually overtake Q(A)")
    print("   - Fast Learner: Quick adaptation")
    print("   - Cautious: Slower switching due to perseveration bias")
    print("   - Explorer: May switch faster due to more exploration of B")
    print()
    print("4. CONVERGENCE MARKERS:")
    print("   - Green dotted: Agent learned initial contingency (90% A choices)")
    print("   - Purple dotted: Agent adapted to reversal (90% B choices)")
    print()
    print("Clinical Relevance:")
    print("- Fast switching = good cognitive flexibility")
    print("- Slow switching = possible impulsivity or mood impairment")
    print("- No switching = severe perseveration/set-shifting deficit")

if __name__ == '__main__':
    demo_q_values()