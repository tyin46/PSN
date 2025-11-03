"""
Quick test to verify BART improvements are working
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from streamlit_app import CustomBARTSimulator
from prlt_personality_proportions import AgentParams

def test_bart_pumping():
    # Test with aggressive pumping parameters
    params = AgentParams(
        learning_rate=0.8,  # High learning rate
        epsilon=0.1,        # Low exploration (more exploitation)
        perseveration=0.1,  # Low perseveration
        decision_noise=0.05,
        patience=10,
        rationale='aggressive_pumper'
    )
    
    simulator = CustomBARTSimulator(
        params,
        num_balloons=10,
        max_pumps=32,
        curve=1.0,
        rng_seed=42
    )
    
    result, history = simulator.run()
    
    pumps_per_balloon = [h['pumps'] for h in history]
    avg_pumps = result.avg_pumps
    exploded_count = result.exploded_count
    
    print(f"🎈 BART Test Results:")
    print(f"   Average pumps per balloon: {avg_pumps:.2f}")
    print(f"   Exploded balloons: {exploded_count}/{result.total_balloons}")
    print(f"   Pumps per balloon: {pumps_per_balloon}")
    print(f"   Q_pump final: {history[-1]['Q_pump']:.2f}")
    print(f"   Q_cash final: {history[-1]['Q_cash']:.2f}")
    
    if avg_pumps > 5:
        print("✅ SUCCESS: Agent is pumping more than 5 times on average!")
    else:
        print("❌ ISSUE: Agent still pumping too conservatively")
    
    return result, history

if __name__ == "__main__":
    test_bart_pumping()