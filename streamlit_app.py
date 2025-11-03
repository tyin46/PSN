"""
Streamlit Web App for BART and PRLT Personality Testing
Features:
- Combined BART and PRLT tests in tabbed interface
- Temperature control for LLM parameter generation
- Graph storage and download functionality
- User API key input for OpenAI access
- Online deployment ready
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import io
import base64
import time
from dataclasses import dataclass, asdict
import json

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prlt_personality_proportions import ParamGenerator, AgentParams, PERSONA_FILES
import os
import tempfile

# Configure Streamlit page
st.set_page_config(
    page_title="Personality Testing Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_personalities():
    """Load personality files from personalities folder and uploaded files"""
    personalities = {}
    
    # Load default personalities from personalities folder
    personalities_folder = Path(__file__).parent / "personalities"
    if personalities_folder.exists():
        for file_path in personalities_folder.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    personalities[file_path.stem] = f.read()
            except Exception as e:
                st.error(f"Error loading {file_path.name}: {e}")
    
    # Add uploaded personalities from session state
    if 'uploaded_personalities' in st.session_state:
        personalities.update(st.session_state.uploaded_personalities)
    
    return personalities

def initialize_session_state():
    """Initialize session state variables"""
    if 'stored_graphs' not in st.session_state:
        st.session_state.stored_graphs = []
    if 'bart_history' not in st.session_state:
        st.session_state.bart_history = None
    if 'prlt_history' not in st.session_state:
        st.session_state.prlt_history = None
    if 'uploaded_personalities' not in st.session_state:
        st.session_state.uploaded_personalities = {}

def create_sidebar():
    """Create sidebar with common controls"""
    st.sidebar.title("🧠 Personality Testing Suite")
    
    # Initialize normalized weights if needed
    if 'weights_reset' in st.session_state and st.session_state.weights_reset:
        # Clear the reset flag
        st.session_state.weights_reset = False
        # Clear normalized weights
        if 'normalized_weights' in st.session_state:
            del st.session_state.normalized_weights
    
    # API Key Input
    st.sidebar.subheader("🔑 OpenAI Configuration")
    api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        help="Required for LLM-based parameter generation. Your key is not stored."
    )
    
    # Temperature Control
    st.sidebar.subheader("🌡️ LLM Temperature")
    temperature = st.sidebar.slider(
        "Temperature (creativity vs consistency):",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more consistent, Higher = more creative parameter generation"
    )
    
    # Use API toggle
    use_api = st.sidebar.checkbox(
        "Use OpenAI API",
        value=bool(api_key),
        help="Uncheck to use heuristic parameter generation"
    )
    
    # File Upload Section
    st.sidebar.subheader("📁 Upload Custom Personalities")
    uploaded_files = st.sidebar.file_uploader(
        "Upload personality .txt files:",
        type=['txt'],
        accept_multiple_files=True,
        help="Upload your own personality prompt files"
    )
    
    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_personalities:
                try:
                    content = uploaded_file.read().decode('utf-8')
                    # Use filename without extension as personality name
                    personality_name = Path(uploaded_file.name).stem
                    st.session_state.uploaded_personalities[personality_name] = content
                    st.sidebar.success(f"✅ Added: {personality_name}")
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading {uploaded_file.name}: {e}")
    
    # Personality Selection
    st.sidebar.subheader("👤 Personality Mixing")
    st.sidebar.write("Select and weight personalities:")
    
    personality_weights = {}
    available_personalities = load_personalities()
    available_personas = list(available_personalities.keys())[:10]  # Increased limit for more personalities
    
    for persona in available_personas:
        col1, col2 = st.sidebar.columns([3, 2])
        with col1:
            enabled = st.checkbox(
                persona.replace('_', ' ').title(),
                key=f"enable_{persona}",
                value=(persona == 'risk_taker')  # Default selection
            )
        with col2:
            if enabled:
                # Use normalized weight if available, otherwise default to 1.0
                default_weight = 1.0
                if ('normalized_weights' in st.session_state and 
                    persona in st.session_state.normalized_weights):
                    default_weight = st.session_state.normalized_weights[persona]
                
                weight = st.slider(
                    "Weight",
                    min_value=0.0,
                    max_value=2.0,
                    value=default_weight,
                    step=0.1,
                    key=f"weight_{persona}",
                    label_visibility="collapsed"
                )
                personality_weights[persona] = weight
    
    # Control buttons
    col_norm, col_clear = st.sidebar.columns(2)
    
    with col_norm:
        if st.button("🔄 Normalize", help="Normalize weights to sum to 1.0"):
            if personality_weights:
                total = sum(personality_weights.values())
                if total > 0:
                    # Store normalized values
                    st.session_state.normalized_weights = {}
                    for persona in personality_weights:
                        normalized_weight = personality_weights[persona] / total
                        st.session_state.normalized_weights[persona] = normalized_weight
                    st.sidebar.success("✅ Weights calculated!")
                    st.sidebar.info("💡 Refresh page to apply normalized values")
                else:
                    st.sidebar.error("❌ All weights are zero!")
            else:
                st.sidebar.error("❌ No personalities selected!")
    
    with col_clear:
        if st.button("🗑️ Clear", help="Clear all weights and selections"):
            # Set flag to reset weights on next run
            st.session_state.weights_reset = True
            st.rerun()
    
    # Display current mix
    if personality_weights:
        total = sum(personality_weights.values())
        if total > 0:
            norm_mix = {k: v/total for k, v in personality_weights.items()}
            mix_display = ", ".join([f"{k}: {v:.1%}" for k, v in norm_mix.items()])
            st.sidebar.info(f"**Current Mix:** {mix_display}")
            
            # Show normalized values if they exist
            if 'normalized_weights' in st.session_state and st.session_state.normalized_weights:
                st.sidebar.write("**Suggested Normalized Values:**")
                for persona, norm_val in st.session_state.normalized_weights.items():
                    if persona in personality_weights:
                        st.sidebar.write(f"• {persona.replace('_', ' ').title()}: {norm_val:.3f}")
    
    # Clear uploaded personalities button
    if st.session_state.uploaded_personalities:
        st.sidebar.write("**Uploaded Personalities:**")
        for name in st.session_state.uploaded_personalities.keys():
            st.sidebar.write(f"• {name}")
        if st.sidebar.button("🗑️ Clear Uploaded Files"):
            st.session_state.uploaded_personalities = {}
            st.rerun()
    
    return api_key, temperature, use_api, personality_weights

def get_parameter_generator(api_key, use_api, personalities=None):
    """Create parameter generator with API key and custom personalities"""
    if use_api and api_key:
        param_gen = ParamGenerator()
        # Set the API key dynamically
        if hasattr(param_gen, 'client') and param_gen.client:
            param_gen.client.api_key = api_key
        # Update persona files if custom personalities provided
        if personalities:
            param_gen.persona_files = personalities
        return param_gen
    return ParamGenerator()  # Will fallback to heuristic

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string for download"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    buffer.close()
    return img_base64

def store_graph(title, fig, test_type, parameters):
    """Store graph in session state"""
    img_base64 = plot_to_base64(fig)
    graph_data = {
        'title': title,
        'image': img_base64,
        'test_type': test_type,
        'parameters': parameters,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.stored_graphs.append(graph_data)

class CustomBARTSimulator:
    """BART simulator for Streamlit app"""
    
    def __init__(self, params: AgentParams, num_balloons=30, max_pumps=64, curve=1.0, rng_seed=None):
        self.params = params
        self.num_balloons = int(num_balloons)
        self.max_pumps = int(max_pumps)
        self.curve = float(curve)
        self.rng = np.random.RandomState(rng_seed)

    def explosion_probability(self, pump_count):
        # More realistic BART explosion probability - very low for first 10-15 pumps
        if pump_count <= 10:
            return 0.01  # Very low risk for first 10 pumps
        elif pump_count <= 20:
            return 0.05 + (pump_count - 10) * 0.01  # Gradual increase
        else:
            return min(0.999, (pump_count / float(self.max_pumps)) ** self.curve)

    def choose(self, Q_pump, Q_cash, prev_action):
        pump_val = Q_pump + (self.params.perseveration if prev_action == 'pump' else 0.0)
        cash_val = Q_cash + (self.params.perseveration if prev_action == 'cash' else 0.0)

        # Add very strong optimism bias toward pumping
        pump_val += 5.0  # Very strong pumping bias
        
        if self.rng.random() < self.params.epsilon:
            return 'pump' if self.rng.random() < 0.8 else 'cash'  # Even exploration favors pumping
        return 'pump' if pump_val > cash_val else 'cash'

    def update(self, Q, reward):
        return Q + self.params.learning_rate * (reward - Q)

    def run(self):
        history = []
        # Start with more optimistic Q-values to encourage pumping
        Q_pump = 3.0  # Higher initial expectation for pumping
        Q_cash = 1.0  # Lower initial expectation for cashing out early
        prev_action = None

        for b in range(1, self.num_balloons + 1):
            pumps = 0
            exploded = False

            while True:
                action = self.choose(Q_pump, Q_cash, prev_action)
                prev_action = action

                if action == 'pump':
                    pumps += 1
                    explosion_prob = self.explosion_probability(pumps)
                    
                    if self.rng.random() < explosion_prob:
                        exploded = True
                        reward = 0
                        # Moderate penalty for explosions (not too harsh)
                        Q_pump = self.update(Q_pump, reward - 1.0)  # Reduced penalty
                        break
                    else:
                        # Very strong positive reward for successful pumps
                        success_reward = 5.0 + (pumps * 0.3)  # Strongly increasing reward per pump
                        Q_pump = self.update(Q_pump, success_reward)
                        
                        # Force cash if at max pumps
                        if pumps >= self.max_pumps:
                            action = 'cash'

                if action == 'cash':
                    # Reward based on pumps achieved
                    reward = pumps + (pumps * 0.1)  # Bonus for more pumps
                    Q_cash = self.update(Q_cash, reward)
                    break

            history.append({
                'balloon': b, 
                'pumps': pumps, 
                'exploded': exploded, 
                'Q_pump': Q_pump, 
                'Q_cash': Q_cash
            })

        @dataclass
        class BARTResult:
            persona_mix_name: str
            params: dict
            total_balloons: int
            avg_pumps: float
            exploded_count: int
            trial_history: list

        avg_pumps = float(np.mean([h['pumps'] for h in history])) if history else 0.0
        exploded_count = sum(1 for h in history if h['exploded'])

        result = BARTResult(
            persona_mix_name='streamlit',
            params=asdict(self.params),
            total_balloons=self.num_balloons,
            avg_pumps=avg_pumps,
            exploded_count=exploded_count,
            trial_history=history
        )

        return result, history

class CustomPRLTSimulator:
    """PRLT simulator for Streamlit app"""
    
    def __init__(self, params: AgentParams, pA_pre=0.75, pB_pre=0.25, pA_post=0.25, pB_post=0.75, rng_seed=None):
        self.params = params
        self.pA_pre = pA_pre
        self.pB_pre = pB_pre
        self.pA_post = pA_post
        self.pB_post = pB_post
        self.rng = np.random.RandomState(rng_seed)

    def choose(self, QA, QB, prev_choice):
        valA = QA + (self.params.perseveration if prev_choice == 'A' else 0.0)
        valB = QB + (self.params.perseveration if prev_choice == 'B' else 0.0)
        
        if self.rng.random() < self.params.epsilon:
            return 'A' if self.rng.random() < 0.5 else 'B'
        return 'A' if valA >= valB else 'B'

    def update(self, QA, QB, choice, reward):
        if choice == 'A':
            QA = QA + self.params.learning_rate * (reward - QA)
        else:
            QB = QB + self.params.learning_rate * (reward - QB)
        return QA, QB

    def run(self, pre_reversal_trials=200):
        pA = self.pA_pre
        pB = self.pB_pre
        QA, QB = 0.5, 0.5
        prev_choice = None
        trial_history = []

        def check_convergence(history, correct_option, window=20, threshold=0.9):
            if len(history) < window:
                return False
            recent = history[-window:]
            picks = sum(1 for t in recent if t['choice'] == correct_option)
            return (picks / window) >= threshold

        # Pre-reversal phase
        pre_converge_trial = None
        correct_pre = 'A' if pA > pB else 'B'
        
        for t in range(1, pre_reversal_trials + 1):
            choice = self.choose(QA, QB, prev_choice)
            reward = 1 if self.rng.random() < (pA if choice == 'A' else pB) else 0
            QA, QB = self.update(QA, QB, choice, reward)
            prev_choice = choice
            trial_history.append({
                'phase': 'pre', 'trial': t, 'choice': choice, 
                'reward': reward, 'QA': QA, 'QB': QB
            })
            
            if check_convergence(trial_history, correct_pre, window=self.params.patience, threshold=0.9):
                pre_converge_trial = t
                break

        if pre_converge_trial is None:
            pre_converge_trial = pre_reversal_trials

        # Post-reversal phase
        pA = self.pA_post
        pB = self.pB_post
        correct_post = 'A' if pA > pB else 'B'

        post_converge_trial = None
        for t2 in range(1, 1001):
            t_index = pre_converge_trial + t2
            choice = self.choose(QA, QB, prev_choice)
            reward = 1 if self.rng.random() < (pA if choice == 'A' else pB) else 0
            QA, QB = self.update(QA, QB, choice, reward)
            prev_choice = choice
            trial_history.append({
                'phase': 'post', 'trial': t_index, 'choice': choice, 
                'reward': reward, 'QA': QA, 'QB': QB
            })
            
            if check_convergence(trial_history, correct_post, window=self.params.patience, threshold=0.9):
                post_converge_trial = t2
                break

        if post_converge_trial is None:
            post_converge_trial = 1000

        @dataclass
        class PRLTResult:
            persona_mix_name: str
            params: dict
            pre_rev_trials_to_converge: int
            post_rev_trials_to_switch: int
            total_trials_run: int
            trial_history: list

        result = PRLTResult(
            persona_mix_name='streamlit',
            params=asdict(self.params),
            pre_rev_trials_to_converge=pre_converge_trial,
            post_rev_trials_to_switch=post_converge_trial,
            total_trials_run=len(trial_history),
            trial_history=trial_history
        )

        return result, trial_history

def bart_test_interface(api_key, temperature, use_api, personality_weights):
    """BART test interface"""
    st.header("🎈 Balloon Analog Risk Task (BART)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Task Parameters")
        
        num_balloons = st.slider("Number of Balloons:", 5, 100, 30)
        max_pumps = st.slider("Max Pumps per Balloon:", 8, 128, 64)
        explosion_curve = st.slider("Explosion Curve:", 0.2, 2.0, 1.0, 0.1,
                                   help="Higher = balloon explodes faster with more pumps")
        
        if st.button("🚀 Run BART Simulation", key="bart_run"):
            if not personality_weights:
                st.error("Please select at least one personality in the sidebar!")
                return
                
            with st.spinner("Running BART simulation..."):
                # Normalize personality mix
                total = sum(personality_weights.values())
                norm_mix = {k: v/total for k, v in personality_weights.items()}
                
                # Get parameters
                personalities = load_personalities()
                param_gen = get_parameter_generator(api_key, use_api, personalities)
                
                if use_api and api_key:
                    try:
                        params = param_gen.get_params_from_llm(norm_mix, temperature=temperature)
                    except Exception as e:
                        st.error(f"API Error: {e}")
                        return
                else:
                    # Enhanced heuristic fallback for BART
                    # More aggressive defaults to encourage pumping
                    base_lr = 0.3 + 0.4 * norm_mix.get('risk_taker', 0) + 0.3 * norm_mix.get('bold_pumper', 0)
                    base_eps = 0.1 * (1 - norm_mix.get('risk_taker', 0) - norm_mix.get('bold_pumper', 0))
                    base_pers = 0.1 + 0.4 * norm_mix.get('cautious_thinker', 0)
                    
                    # Moderate pumper influence
                    if 'moderate_pumper' in norm_mix:
                        base_lr += 0.2 * norm_mix['moderate_pumper']
                        base_eps += 0.15 * norm_mix['moderate_pumper']
                    
                    lr = min(0.9, base_lr)  # Cap at 0.9
                    eps = min(0.5, max(0.05, base_eps))  # Keep between 0.05-0.5
                    pers = min(0.5, base_pers)  # Cap at 0.5
                    patience = int(5 + 15 * norm_mix.get('cautious_thinker', 0))
                    params = AgentParams(lr, eps, pers, 0.05, patience, rationale='enhanced_heuristic')
                
                # Run simulation
                simulator = CustomBARTSimulator(
                    params, num_balloons, max_pumps, explosion_curve, 
                    rng_seed=int(time.time()) % 2**32
                )
                result, history = simulator.run()
                
                # Store results
                st.session_state.bart_history = history
                
                st.success(f"✅ Simulation complete! Avg pumps: {result.avg_pumps:.2f}, "
                          f"Exploded: {result.exploded_count}/{result.total_balloons}")
    
    with col2:
        st.subheader("Results Visualization")
        
        if st.session_state.bart_history:
            history = st.session_state.bart_history
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot 1: Q-values over balloons
            balloons = [h['balloon'] for h in history]
            Q_pump = [h['Q_pump'] for h in history]
            Q_cash = [h['Q_cash'] for h in history]
            
            ax1.plot(balloons, Q_pump, 'b-', label='Q(Pump)', linewidth=2)
            ax1.plot(balloons, Q_cash, 'g-', label='Q(Cash)', linewidth=2)
            ax1.set_xlabel('Balloon')
            ax1.set_ylabel('Estimated Value')
            ax1.set_title('Agent Value Estimates Across Balloons')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Pumps per balloon
            pumps = [h['pumps'] for h in history]
            exploded = [h['exploded'] for h in history]
            
            exploded_x = [balloons[i] for i, e in enumerate(exploded) if e]
            exploded_y = [pumps[i] for i, e in enumerate(exploded) if e]
            cashed_x = [balloons[i] for i, e in enumerate(exploded) if not e]
            cashed_y = [pumps[i] for i, e in enumerate(exploded) if not e]
            
            if cashed_x:
                ax2.scatter(cashed_x, cashed_y, c='green', label='Cashed', alpha=0.7)
            if exploded_x:
                ax2.scatter(exploded_x, exploded_y, c='red', label='Exploded', alpha=0.7)
            
            ax2.set_xlabel('Balloon')
            ax2.set_ylabel('Pumps')
            ax2.set_title('Pumps per Balloon (Green=cash, Red=exploded)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Store graph button
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("💾 Store Graph", key="store_bart"):
                    store_graph(
                        f"BART Results - {len(st.session_state.stored_graphs)+1}",
                        fig,
                        "BART",
                        {"balloons": num_balloons, "max_pumps": max_pumps, "curve": explosion_curve}
                    )
                    st.success("Graph stored!")
            
            with col_b:
                # Download button
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                st.download_button(
                    "📥 Download PNG",
                    img_buffer.getvalue(),
                    f"bart_results_{int(time.time())}.png",
                    "image/png"
                )
        else:
            st.info("👆 Run a simulation to see results here")

def prlt_test_interface(api_key, temperature, use_api, personality_weights):
    """PRLT test interface"""
    st.header("🔄 Probabilistic Reversal Learning Task (PRLT)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Task Parameters")
        
        st.write("**Pre-Reversal Phase:**")
        pA_pre = st.slider("P(Reward|A):", 0.1, 0.9, 0.75, 0.05)
        pB_pre = 1.0 - pA_pre
        st.write(f"P(Reward|B): {pB_pre:.2f}")
        
        st.write("**Post-Reversal Phase:**")
        pA_post = st.slider("P(Reward|A) after reversal:", 0.1, 0.9, 0.25, 0.05)
        pB_post = 1.0 - pA_post
        st.write(f"P(Reward|B) after reversal: {pB_post:.2f}")
        
        pre_trials = st.slider("Pre-reversal trials:", 50, 500, 200)
        
        if st.button("🚀 Run PRLT Simulation", key="prlt_run"):
            if not personality_weights:
                st.error("Please select at least one personality in the sidebar!")
                return
                
            with st.spinner("Running PRLT simulation..."):
                # Normalize personality mix
                total = sum(personality_weights.values())
                norm_mix = {k: v/total for k, v in personality_weights.items()}
                
                # Get parameters
                personalities = load_personalities()
                param_gen = get_parameter_generator(api_key, use_api, personalities)
                
                if use_api and api_key:
                    try:
                        params = param_gen.get_params_from_llm(norm_mix, temperature=temperature)
                    except Exception as e:
                        st.error(f"API Error: {e}")
                        return
                else:
                    # Heuristic fallback for PRLT
                    lr = 0.1 + 0.4 * norm_mix.get('risk_taker', 0)
                    eps = 0.15 + 0.4 * (1 - norm_mix.get('cautious_thinker', 0))
                    pers = 0.05 + 0.4 * norm_mix.get('cautious_thinker', 0)
                    patience = int(10 + 30 * norm_mix.get('cautious_thinker', 0))
                    params = AgentParams(lr, eps, pers, 0.05, patience, rationale='heuristic')
                
                # Run simulation
                simulator = CustomPRLTSimulator(
                    params, pA_pre, pB_pre, pA_post, pB_post,
                    rng_seed=int(time.time()) % 2**32
                )
                result, history = simulator.run(pre_trials)
                
                # Store results
                st.session_state.prlt_history = history
                
                st.success(f"✅ Simulation complete! Pre-converge: {result.pre_rev_trials_to_converge} trials, "
                          f"Post-switch: {result.post_rev_trials_to_switch} trials")
    
    with col2:
        st.subheader("Results Visualization")
        
        if st.session_state.prlt_history:
            history = st.session_state.prlt_history
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot 1: Q-values over time
            trials = [h['trial'] for h in history]
            QA_vals = [h['QA'] for h in history]
            QB_vals = [h['QB'] for h in history]
            phases = [h['phase'] for h in history]
            
            # Find reversal point
            reversal_trial = None
            for i, phase in enumerate(phases):
                if phase == 'post':
                    reversal_trial = trials[i]
                    break
            
            ax1.plot(trials, QA_vals, 'b-', label='Q(A)', linewidth=2)
            ax1.plot(trials, QB_vals, 'r-', label='Q(B)', linewidth=2)
            
            if reversal_trial:
                ax1.axvline(reversal_trial, color='orange', linestyle='--', alpha=0.7, label='Reversal')
            
            ax1.set_xlabel('Trial')
            ax1.set_ylabel('Q-Value')
            ax1.set_title('Agent Q-Values Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Choice sequence
            choices = [1 if h['choice'] == 'A' else 0 for h in history]
            choice_A = [trials[i] for i, c in enumerate(choices) if c == 1]
            choice_B = [trials[i] for i, c in enumerate(choices) if c == 0]
            
            if choice_A:
                ax2.scatter(choice_A, [1]*len(choice_A), c='blue', alpha=0.6, s=20, label='Choice A')
            if choice_B:
                ax2.scatter(choice_B, [0]*len(choice_B), c='red', alpha=0.6, s=20, label='Choice B')
            
            if reversal_trial:
                ax2.axvline(reversal_trial, color='orange', linestyle='--', alpha=0.7)
            
            ax2.set_xlabel('Trial')
            ax2.set_ylabel('Choice')
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['B', 'A'])
            ax2.set_title('Choice Sequence')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Store graph button
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("💾 Store Graph", key="store_prlt"):
                    store_graph(
                        f"PRLT Results - {len(st.session_state.stored_graphs)+1}",
                        fig,
                        "PRLT",
                        {"pA_pre": pA_pre, "pB_pre": pB_pre, "pA_post": pA_post, "pB_post": pB_post}
                    )
                    st.success("Graph stored!")
            
            with col_b:
                # Download button
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                st.download_button(
                    "📥 Download PNG",
                    img_buffer.getvalue(),
                    f"prlt_results_{int(time.time())}.png",
                    "image/png"
                )
        else:
            st.info("👆 Run a simulation to see results here")

def stored_graphs_interface():
    """Interface for viewing and managing stored graphs"""
    st.header("💾 Stored Graphs")
    
    if not st.session_state.stored_graphs:
        st.info("No graphs stored yet. Run simulations and use the 'Store Graph' button to save results.")
        return
    
    # Display stored graphs
    for i, graph in enumerate(st.session_state.stored_graphs):
        with st.expander(f"{graph['title']} - {graph['timestamp']} ({graph['test_type']})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display image
                img_data = base64.b64decode(graph['image'])
                st.image(img_data, caption=graph['title'])
            
            with col2:
                st.write("**Parameters:**")
                for key, value in graph['parameters'].items():
                    st.write(f"- {key}: {value}")
                
                # Download button
                st.download_button(
                    "📥 Download",
                    img_data,
                    f"{graph['title'].replace(' ', '_').lower()}_{graph['timestamp'].replace(':', '-')}.png",
                    "image/png",
                    key=f"download_{i}"
                )
    
    # Clear all button
    if st.button("🗑️ Clear All Stored Graphs"):
        st.session_state.stored_graphs = []
        st.success("All stored graphs cleared!")

def main():
    """Main Streamlit app"""
    initialize_session_state()
    
    # Create sidebar
    api_key, temperature, use_api, personality_weights = create_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🎈 BART Test", "🔄 PRLT Test", "💾 Stored Graphs"])
    
    with tab1:
        bart_test_interface(api_key, temperature, use_api, personality_weights)
    
    with tab2:
        prlt_test_interface(api_key, temperature, use_api, personality_weights)
    
    with tab3:
        stored_graphs_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Personality Testing Suite** - Built with Streamlit | "
        "[GitHub](https://github.com) | [Documentation](https://docs.streamlit.io)"
    )

if __name__ == "__main__":
    main()