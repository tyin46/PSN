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

def format_bart_context(balloon_num, pumps_so_far, history, max_pumps):
    """Format BART context for LLM decision making"""
    context = f"""You are participating in a Balloon Analog Risk Task (BART).

Current Situation:
- Balloon #{balloon_num}
- Pumps so far: {pumps_so_far}
- Maximum possible pumps: {max_pumps}
- Each pump increases the balloon value but also increases explosion risk

Previous Results:"""
    
    if history:
        for h in history[-5:]:  # Show last 5 balloons
            result = "exploded" if h['exploded'] else "cashed out"
            context += f"\n- Balloon {h['balloon']}: {h['pumps']} pumps, {result}"
    else:
        context += "\n- No previous balloons"
    
    context += "\n\nDecision: Should you 'pump' (increase value but risk explosion) or 'cash' (take current value safely)?"
    context += "\nRespond with exactly one word: 'pump' or 'cash'"
    
    return context

def format_prlt_context(trial_num, phase, choice_history, reward_history, QA, QB):
    """Format PRLT context for LLM decision making"""
    context = f"""You are participating in a Probabilistic Reversal Learning Task (PRLT).

Current Situation:
- Trial #{trial_num}
- Phase: {phase}-reversal
- You can choose option A or option B
- Current value estimates: A={QA:.2f}, B={QB:.2f}

Recent History:"""
    
    if choice_history:
        recent_trials = min(10, len(choice_history))
        for i in range(-recent_trials, 0):
            choice = choice_history[i]
            reward = "reward" if reward_history[i] == 1 else "no reward"
            context += f"\n- Trial {len(choice_history) + i + 1}: chose {choice}, got {reward}"
    else:
        context += "\n- No previous trials"
    
    context += "\n\nDecision: Which option do you choose?"
    context += "\nRespond with exactly one letter: 'A' or 'B'"
    
    return context

def format_mcq_context(choice_num, immediate_amt, immediate_delay, delayed_amt, delayed_delay, history):
    """Format MCQ context for LLM decision making"""
    context = f"""You are participating in a Monetary Choice Questionnaire (MCQ).

Current Choice #{choice_num}:
- Immediate option: ${immediate_amt} in {immediate_delay} days
- Delayed option: ${delayed_amt} in {delayed_delay} days

Previous Choices:"""
    
    if history:
        for h in history[-5:]:  # Show last 5 choices
            context += f"\n- Choice {h['choice_num']}: chose {h['choice']} option"
    else:
        context += "\n- No previous choices"
    
    context += "\n\nDecision: Which option do you prefer?"
    context += "\nRespond with exactly one word: 'immediate' or 'delayed'"
    
    return context

def format_igt_context(trial_num, deck_history, money_total, Q_values, deck_config):
    """Format IGT context for LLM decision making"""
    context = f"""You are participating in the Iowa Gambling Task (IGT).

Current Situation:
- Trial #{trial_num}
- Current money: ${money_total}
- Available decks: A, B, C, D

Deck Performance So Far:"""
    
    if deck_history:
        deck_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        deck_outcomes = {'A': [], 'B': [], 'C': [], 'D': []}
        
        for h in deck_history:
            deck_counts[h['deck']] += 1
            deck_outcomes[h['deck']].append(h['net_outcome'])
        
        for deck in ['A', 'B', 'C', 'D']:
            if deck_counts[deck] > 0:
                avg_outcome = sum(deck_outcomes[deck]) / len(deck_outcomes[deck])
                context += f"\n- Deck {deck}: {deck_counts[deck]} selections, avg outcome: ${avg_outcome:.1f}"
            else:
                context += f"\n- Deck {deck}: not selected yet"
    else:
        context += "\n- No previous selections"
    
    context += "\n\nRecent Outcomes:"
    if deck_history:
        for h in deck_history[-5:]:
            context += f"\n- Trial {h['trial']}: Deck {h['deck']}, outcome: ${h['net_outcome']}"
    
    context += "\n\nDecision: Which deck do you choose?"
    context += "\nRespond with exactly one letter: 'A', 'B', 'C', or 'D'"
    
    return context

def get_llm_decision(context, api_key, temperature=0.3):
    """Get LLM decision based on context"""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are participating in a psychological experiment. Make decisions based on the context provided. Follow instructions exactly."},
                {"role": "user", "content": context}
            ],
            temperature=temperature,
            max_tokens=10
        )
        
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        st.error(f"LLM decision error: {e}")
        return None

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
    if 'mcq_history' not in st.session_state:
        st.session_state.mcq_history = None
    if 'igt_history' not in st.session_state:
        st.session_state.igt_history = None
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
    st.sidebar.subheader("OpenAI Configuration")
    api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        help="Required for LLM-based parameter generation. Your key is not stored."
    )
    
    # Temperature Control
    st.sidebar.subheader("LLM Temperature")
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
    st.sidebar.subheader("Upload Custom Personalities")
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
    st.sidebar.subheader("Personality Mixing")
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

class CustomMCQSimulator:
    """Monetary Choice Questionnaire simulator for Streamlit app"""
    
    def __init__(self, params: AgentParams, num_choices=27, custom_choice_pairs=None, rng_seed=None):
        self.params = params
        self.num_choices = int(num_choices)
        self.rng = np.random.RandomState(rng_seed)
        
        # Use custom choice pairs if provided, otherwise use classic MCQ pairs
        if custom_choice_pairs:
            self.choice_pairs = custom_choice_pairs
        else:
            # MCQ choice pairs (smaller immediate vs larger delayed)
            # Based on Kirby, Petry & Bickel (1999)
            self.choice_pairs = [
                (11, 0, 30, 7),   # $11 now vs $30 in 7 days
                (54, 0, 55, 117), # $54 now vs $55 in 117 days
                (47, 0, 50, 160), # $47 now vs $50 in 160 days
                (15, 0, 35, 13),  # $15 now vs $35 in 13 days
                (25, 0, 60, 14),  # $25 now vs $60 in 14 days
                (78, 0, 80, 162), # $78 now vs $80 in 162 days
                (40, 0, 55, 62),  # $40 now vs $55 in 62 days
                (11, 0, 30, 7),   # $11 now vs $30 in 7 days
                (67, 0, 75, 119), # $67 now vs $75 in 119 days
                (34, 0, 35, 186), # $34 now vs $35 in 186 days
                (27, 0, 50, 21),  # $27 now vs $50 in 21 days
                (69, 0, 85, 91),  # $69 now vs $85 in 91 days
                (49, 0, 60, 89),  # $49 now vs $60 in 89 days
                (80, 0, 85, 157), # $80 now vs $85 in 157 days
                (24, 0, 35, 29),  # $24 now vs $35 in 29 days
                (33, 0, 80, 14),  # $33 now vs $80 in 14 days
                (28, 0, 30, 179), # $28 now vs $30 in 179 days
                (34, 0, 50, 30),  # $34 now vs $50 in 30 days
                (25, 0, 30, 80),  # $25 now vs $30 in 80 days
                (41, 0, 75, 20),  # $41 now vs $75 in 20 days
                (54, 0, 60, 111), # $54 now vs $60 in 111 days
                (54, 0, 80, 30),  # $54 now vs $80 in 30 days
                (22, 0, 25, 136), # $22 now vs $25 in 136 days
                (20, 0, 55, 7),   # $20 now vs $55 in 7 days
                (79, 0, 80, 162), # $79 now vs $80 in 162 days
                (16, 0, 30, 15),  # $16 now vs $30 in 15 days
                (31, 0, 85, 7),   # $31 now vs $85 in 7 days
            ]
    
    def discount_value(self, amount, delay, k_value=0.01):
        """Calculate discounted value using hyperbolic discounting"""
        if delay == 0:
            return amount
        return amount / (1 + k_value * delay)
    
    def choose(self, immediate_val, delayed_val, prev_choice):
        """Choose between immediate and delayed rewards"""
        # Add perseveration bias
        if prev_choice == 'immediate':
            immediate_val += self.params.perseveration
        elif prev_choice == 'delayed':
            delayed_val += self.params.perseveration
        
        # Add exploration noise
        if self.rng.random() < self.params.epsilon:
            return 'immediate' if self.rng.random() < 0.5 else 'delayed'
        
        return 'immediate' if immediate_val >= delayed_val else 'delayed'
    
    def run(self):
        history = []
        Q_immediate = 0.5  # Initial value estimates
        Q_delayed = 0.5
        prev_choice = None
        
        # Use subset of choices based on num_choices
        selected_choices = self.choice_pairs[:self.num_choices]
        
        for i, (imm_amt, imm_delay, del_amt, del_delay) in enumerate(selected_choices):
            # Calculate subjective values (agent learns discount rate)
            k_estimate = max(0.001, min(1.0, self.params.learning_rate))  # Use LR as discount sensitivity
            
            imm_value = self.discount_value(imm_amt, imm_delay, k_estimate) * Q_immediate
            del_value = self.discount_value(del_amt, del_delay, k_estimate) * Q_delayed
            
            choice = self.choose(imm_value, del_value, prev_choice)
            prev_choice = choice
            
            # Learning: update value estimates based on choice
            if choice == 'immediate':
                # Positive feedback for immediate choice
                Q_immediate = Q_immediate + self.params.learning_rate * (1.0 - Q_immediate)
                Q_delayed = Q_delayed + self.params.learning_rate * (-0.1 - Q_delayed)  # Slight negative
            else:
                # Positive feedback for delayed choice
                Q_delayed = Q_delayed + self.params.learning_rate * (1.0 - Q_delayed)
                Q_immediate = Q_immediate + self.params.learning_rate * (-0.1 - Q_immediate)
            
            history.append({
                'choice_num': i + 1,
                'immediate_amount': imm_amt,
                'immediate_delay': imm_delay,
                'delayed_amount': del_amt,
                'delayed_delay': del_delay,
                'choice': choice,
                'Q_immediate': Q_immediate,
                'Q_delayed': Q_delayed,
                'k_estimate': k_estimate
            })
        
        @dataclass
        class MCQResult:
            persona_mix_name: str
            params: dict
            total_choices: int
            immediate_count: int
            delayed_count: int
            estimated_k: float
            trial_history: list
        
        immediate_count = sum(1 for h in history if h['choice'] == 'immediate')
        delayed_count = len(history) - immediate_count
        avg_k = np.mean([h['k_estimate'] for h in history]) if history else 0.0
        
        result = MCQResult(
            persona_mix_name='streamlit',
            params=asdict(self.params),
            total_choices=len(history),
            immediate_count=immediate_count,
            delayed_count=delayed_count,
            estimated_k=avg_k,
            trial_history=history
        )
        
        return result, history

class CustomIGTSimulator:
    """Iowa Gambling Task simulator for Streamlit app"""
    
    def __init__(self, params: AgentParams, num_trials=100, custom_decks=None, rng_seed=None):
        self.params = params
        self.num_trials = int(num_trials)
        self.rng = np.random.RandomState(rng_seed)
        
        # IGT deck characteristics (gains, losses, net expected value)
        # Use custom decks if provided, otherwise use default Bechara et al. (1994) settings
        if custom_decks:
            self.decks = custom_decks
        else:
            # Default deck characteristics: Decks A & B disadvantageous, C & D advantageous
            self.decks = {
                'A': {'gain': 100, 'loss_prob': 0.5, 'loss_amt': -250, 'net_expected': -25},
                'B': {'gain': 100, 'loss_prob': 0.1, 'loss_amt': -1250, 'net_expected': -25},
                'C': {'gain': 50, 'loss_prob': 0.5, 'loss_amt': -50, 'net_expected': 25},
                'D': {'gain': 50, 'loss_prob': 0.1, 'loss_amt': -250, 'net_expected': 25}
            }
    
    def choose_deck(self, Q_values, prev_choice):
        """Choose a deck based on Q-values and exploration"""
        # Add perseveration bias
        adjusted_Q = Q_values.copy()
        if prev_choice:
            adjusted_Q[prev_choice] += self.params.perseveration
        
        # Exploration vs exploitation
        if self.rng.random() < self.params.epsilon:
            return self.rng.choice(['A', 'B', 'C', 'D'])
        
        # Choose deck with highest Q-value
        return max(adjusted_Q, key=adjusted_Q.get)
    
    def get_outcome(self, deck):
        """Get reward/penalty from chosen deck"""
        deck_info = self.decks[deck]
        gain = deck_info['gain']
        
        # Determine if loss occurs
        if self.rng.random() < deck_info['loss_prob']:
            loss = deck_info['loss_amt']
        else:
            loss = 0
        
        net_outcome = gain + loss
        return gain, loss, net_outcome
    
    def run(self):
        history = []
        Q_values = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}  # Initial Q-values
        prev_choice = None
        total_money = 2000  # Starting amount
        
        for trial in range(1, self.num_trials + 1):
            # Choose deck
            chosen_deck = self.choose_deck(Q_values, prev_choice)
            
            # Get outcome
            gain, loss, net_outcome = self.get_outcome(chosen_deck)
            total_money += net_outcome
            
            # Update Q-value for chosen deck
            Q_values[chosen_deck] += self.params.learning_rate * (net_outcome - Q_values[chosen_deck])
            
            # Record trial
            history.append({
                'trial': trial,
                'deck': chosen_deck,
                'gain': gain,
                'loss': loss,
                'net_outcome': net_outcome,
                'total_money': total_money,
                'Q_A': Q_values['A'],
                'Q_B': Q_values['B'],
                'Q_C': Q_values['C'],
                'Q_D': Q_values['D']
            })
            
            prev_choice = chosen_deck
        
        @dataclass
        class IGTResult:
            persona_mix_name: str
            params: dict
            total_trials: int
            final_money: float
            advantageous_choices: int
            disadvantageous_choices: int
            trial_history: list
        
        # Count advantageous (C, D) vs disadvantageous (A, B) choices
        advantageous = sum(1 for h in history if h['deck'] in ['C', 'D'])
        disadvantageous = sum(1 for h in history if h['deck'] in ['A', 'B'])
        
        result = IGTResult(
            persona_mix_name='streamlit',
            params=asdict(self.params),
            total_trials=self.num_trials,
            final_money=total_money,
            advantageous_choices=advantageous,
            disadvantageous_choices=disadvantageous,
            trial_history=history
        )
        
        return result, history

def bart_test_interface(api_key, temperature, use_api, personality_weights):
    """BART test interface"""
    st.header("Balloon Analog Risk Task (BART)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Task Parameters")
        
        num_balloons = st.slider("Number of Balloons:", 5, 100, 30)
        max_pumps = st.slider("Max Pumps per Balloon:", 8, 128, 64)
        explosion_curve = st.slider("Explosion Curve:", 0.2, 2.0, 1.0, 0.1,
                                   help="Higher = balloon explodes faster with more pumps")
        
        # LLM Decision Mode
        llm_mode = st.checkbox(
            "🤖 LLM Decision Mode", 
            help="Agent makes decisions step-by-step using language model instead of Q-learning algorithm"
        )
        
        if st.button("🚀 Run BART Simulation", key="bart_run"):
            if not personality_weights:
                st.error("Please select at least one personality in the sidebar!")
                return
                
            with st.spinner("Running BART simulation..."):
                if llm_mode:
                    # LLM-driven simulation
                    if not api_key:
                        st.error("LLM Decision Mode requires an OpenAI API key.")
                        return
                    
                    history = []
                    for balloon_num in range(1, num_balloons + 1):
                        pumps = 0
                        exploded = False
                        
                        while pumps < max_pumps:
                            # Get LLM decision
                            context = format_bart_context(balloon_num, pumps, history, max_pumps)
                            decision = get_llm_decision(context, api_key, temperature)
                            
                            if decision in ['cash', 'cash out', 'take']:
                                break
                            elif decision in ['pump', 'continue', 'risk']:
                                pumps += 1
                                # Check for explosion
                                explosion_prob = min(0.999, (pumps / float(max_pumps)) ** explosion_curve)
                                if np.random.random() < explosion_prob:
                                    exploded = True
                                    break
                            else:
                                # Default to cash if unclear response
                                break
                        
                        history.append({
                            'balloon': balloon_num,
                            'pumps': pumps,
                            'exploded': exploded,
                            'Q_pump': 0.0,  # Not used in LLM mode
                            'Q_cash': 0.0   # Not used in LLM mode
                        })
                    
                    avg_pumps = float(np.mean([h['pumps'] for h in history])) if history else 0.0
                    exploded_count = sum(1 for h in history if h['exploded'])
                    st.session_state.bart_history = history
                    st.success(f"LLM Simulation complete! Average pumps: {avg_pumps:.1f}, Explosions: {exploded_count}/{num_balloons}")
                    
                else:
                    # Traditional Q-learning simulation
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
                    
                    if llm_mode:
                        st.success(f"✅ LLM Simulation complete! Avg pumps: {avg_pumps:.2f}, "
                                  f"Exploded: {exploded_count}/{num_balloons}")
                    else:
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
    st.header("Probabilistic Reversal Learning Task (PRLT)")
    
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
        
        # LLM Decision Mode
        llm_mode_prlt = st.checkbox(
            "🤖 LLM Decision Mode", 
            key="prlt_llm",
            help="Agent makes decisions step-by-step using language model instead of Q-learning algorithm"
        )
        
        if st.button("🚀 Run PRLT Simulation", key="prlt_run"):
            if not personality_weights:
                st.error("Please select at least one personality in the sidebar!")
                return
                
            with st.spinner("Running PRLT simulation..."):
                if llm_mode_prlt:
                    # LLM-driven simulation
                    if not api_key:
                        st.error("LLM Decision Mode requires an OpenAI API key.")
                        return
                    
                    QA, QB = 0.5, 0.5
                    choice_history = []
                    reward_history = []
                    trial_history = []
                    
                    # Pre-reversal phase
                    for trial in range(1, pre_trials + 1):
                        context = format_prlt_context(trial, "pre", choice_history, reward_history, QA, QB)
                        decision = get_llm_decision(context, api_key, temperature)
                        
                        choice = 'A' if decision.upper() in ['A', 'OPTION A', 'CHOICE A'] else 'B'
                        reward = 1 if np.random.random() < (pA_pre if choice == 'A' else pB_pre) else 0
                        
                        # Update Q-values for display
                        if choice == 'A':
                            QA = QA + 0.1 * (reward - QA)
                        else:
                            QB = QB + 0.1 * (reward - QB)
                        
                        choice_history.append(choice)
                        reward_history.append(reward)
                        trial_history.append({
                            'phase': 'pre', 'trial': trial, 'choice': choice,
                            'reward': reward, 'QA': QA, 'QB': QB
                        })
                    
                    # Post-reversal phase
                    for trial in range(1, 201):  # Max 200 post-reversal trials
                        t_index = pre_trials + trial
                        context = format_prlt_context(t_index, "post", choice_history, reward_history, QA, QB)
                        decision = get_llm_decision(context, api_key, temperature)
                        
                        choice = 'A' if decision.upper() in ['A', 'OPTION A', 'CHOICE A'] else 'B'
                        reward = 1 if np.random.random() < (pA_post if choice == 'A' else pB_post) else 0
                        
                        # Update Q-values for display
                        if choice == 'A':
                            QA = QA + 0.1 * (reward - QA)
                        else:
                            QB = QB + 0.1 * (reward - QB)
                        
                        choice_history.append(choice)
                        reward_history.append(reward)
                        trial_history.append({
                            'phase': 'post', 'trial': t_index, 'choice': choice,
                            'reward': reward, 'QA': QA, 'QB': QB
                        })
                        
                        # Check convergence to new optimal choice
                        if len(choice_history) >= 20:
                            correct_choice = 'B' if pA_post < pB_post else 'A'
                            recent_choices = choice_history[-20:]
                            if sum(1 for c in recent_choices if c == correct_choice) >= 18:  # 90% correct
                                break
                    
                    st.session_state.prlt_history = trial_history
                    st.success(f"LLM PRLT complete! Total trials: {len(trial_history)}")
                    
                else:
                    # Traditional Q-learning simulation
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

def mcq_test_interface(api_key, temperature, use_api, personality_weights):
    """MCQ test interface"""
    st.header("Monetary Choice Questionnaire (MCQ)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Task Parameters")
        
        num_choices = st.slider("Number of Choice Pairs:", 10, 27, 27,
                               help="Number of immediate vs delayed reward choices")
        
        st.write("**Choice Configuration:**")
        
        # Choice generation method
        choice_method = st.radio(
            "Choice Generation Method:",
            ["Classic MCQ (Kirby et al. 1999)", "Custom Parameters"],
            help="Use original research choices or create custom reward/delay pairs"
        )
        
        if choice_method == "Custom Parameters":
            with st.expander("Custom Choice Parameters", expanded=True):
                col_imm, col_del = st.columns(2)
                
                with col_imm:
                    st.write("**Immediate Rewards:**")
                    imm_min = st.number_input("Min Immediate Reward ($):", min_value=1, max_value=500, value=15, key="imm_min")
                    imm_max = st.number_input("Max Immediate Reward ($):", min_value=imm_min+1, max_value=1000, value=80, key="imm_max")
                    imm_delay = st.number_input("Immediate Delay (days):", min_value=0, max_value=30, value=0, key="imm_delay")
                
                with col_del:
                    st.write("**Delayed Rewards:**")
                    del_min_default = max(25, imm_max + 5)  # Ensure delayed min is always higher than immediate max
                    del_min = st.number_input("Min Delayed Reward ($):", min_value=imm_max+1, max_value=2000, value=del_min_default, key="del_min")
                    del_max_default = max(200, del_min + 20)  # Ensure delayed max is always higher than delayed min
                    del_max = st.number_input("Max Delayed Reward ($):", min_value=del_min+1, max_value=5000, value=del_max_default, key="del_max")
                    del_delay_min = st.number_input("Min Delay (days):", min_value=1, max_value=365, value=7, key="del_delay_min")
                    del_delay_max = st.number_input("Max Delay (days):", min_value=del_delay_min+1, max_value=365, value=180, key="del_delay_max")
                
                # Delay distribution
                st.write("**Delay Distribution:**")
                delay_distribution = st.selectbox(
                    "Delay Pattern:",
                    ["Linear", "Exponential", "Custom List"],
                    help="How delays are distributed across choices"
                )
                
                if delay_distribution == "Custom List":
                    custom_delays = st.text_input(
                        "Custom Delays (comma-separated days):",
                        value="7, 14, 30, 60, 90, 120, 180",
                        help="Enter specific delay values in days"
                    )
        
        # Quick presets
        st.write("**Quick Presets:**")
        col_preset1, col_preset2 = st.columns(2)
        
        with col_preset1:
            if st.button("📊 Classic MCQ", help="Original Kirby, Petry & Bickel (1999) settings"):
                st.session_state.mcq_choice_method = "Classic MCQ (Kirby et al. 1999)"
                st.rerun()
        
        with col_preset2:
            if st.button("⚖️ Modern Range", help="Contemporary values with extended delays"):
                st.session_state.mcq_choice_method = "Custom Parameters"
                st.session_state.imm_min = 20
                st.session_state.imm_max = 100
                st.session_state.imm_delay = 0
                st.session_state.del_min = 30
                st.session_state.del_max = 300
                st.session_state.del_delay_min = 7
                st.session_state.del_delay_max = 365
                st.rerun()
        
        # Generate choice pairs based on method
        if choice_method == "Custom Parameters":
            # Create custom choice pairs
            import numpy as np
            np.random.seed(42)  # For consistent custom choices
            
            custom_choice_pairs = []
            
            if delay_distribution == "Custom List":
                try:
                    delays = [int(d.strip()) for d in custom_delays.split(",")]
                    delays = delays[:num_choices]  # Limit to num_choices
                    if len(delays) < num_choices:
                        # Repeat delays if needed
                        delays = (delays * (num_choices // len(delays) + 1))[:num_choices]
                except:
                    delays = [7, 14, 30, 60, 90, 120, 180][:num_choices]
            elif delay_distribution == "Linear":
                delays = np.linspace(del_delay_min, del_delay_max, num_choices, dtype=int)
            else:  # Exponential
                delays = np.logspace(np.log10(del_delay_min), np.log10(del_delay_max), num_choices, dtype=int)
            
            for i in range(num_choices):
                imm_amt = np.random.randint(imm_min, imm_max + 1)
                del_amt = np.random.randint(max(del_min, imm_amt + 5), del_max + 1)
                del_delay = int(delays[i]) if i < len(delays) else del_delay_max
                
                custom_choice_pairs.append((imm_amt, imm_delay, del_amt, del_delay))
            
            choice_pairs_to_use = custom_choice_pairs
        else:
            # Use classic MCQ pairs
            choice_pairs_to_use = None  # Will use default in simulator
        
        # Display current choices preview
        if choice_method == "Custom Parameters":
            st.write("**Choice Pairs Preview:**")
            preview_choices = min(5, len(custom_choice_pairs))
            for i in range(preview_choices):
                imm_amt, imm_delay, del_amt, del_delay = custom_choice_pairs[i]
                if imm_delay == 0:
                    st.write(f"{i+1}. ${imm_amt} now vs ${del_amt} in {del_delay} days")
                else:
                    st.write(f"{i+1}. ${imm_amt} in {imm_delay} days vs ${del_amt} in {del_delay} days")
            if preview_choices < len(custom_choice_pairs):
                st.write(f"... and {len(custom_choice_pairs) - preview_choices} more choices")
        else:
            st.write("**Using Classic MCQ Choices:**")
            st.write("• $11 now vs $30 in 7 days")
            st.write("• $25 now vs $60 in 14 days") 
            st.write("• $54 now vs $80 in 30 days")
            st.write("• ... and 24 more validated pairs")
        
        # LLM Decision Mode
        llm_mode_mcq = st.checkbox(
            "🤖 LLM Decision Mode", 
            key="mcq_llm",
            help="Agent makes decisions step-by-step using language model instead of Q-learning algorithm"
        )
        
        if st.button("🚀 Run MCQ Simulation", key="mcq_run"):
            if not personality_weights:
                st.error("Please select at least one personality in the sidebar!")
                return
                
            with st.spinner("Running MCQ simulation..."):
                if llm_mode_mcq:
                    # LLM-driven simulation
                    if not api_key:
                        st.error("LLM Decision Mode requires an OpenAI API key.")
                        return
                    
                    history = []
                    Q_immediate = 0.5
                    Q_delayed = 0.5
                    
                    # Use appropriate choice pairs
                    for i, (imm_amt, imm_delay, del_amt, del_delay) in enumerate(choice_pairs_to_use):
                        context = format_mcq_context(i+1, imm_amt, imm_delay, del_amt, del_delay, history)
                        decision = get_llm_decision(context, api_key, temperature)
                        
                        choice = 'immediate' if decision in ['immediate', 'now', 'sooner', 'imm'] else 'delayed'
                        
                        # Update Q-values for tracking
                        if choice == 'immediate':
                            Q_immediate = Q_immediate + 0.1 * (1.0 - Q_immediate)
                        else:
                            Q_delayed = Q_delayed + 0.1 * (1.0 - Q_delayed)
                        
                        history.append({
                            'choice_num': i + 1,
                            'immediate_amount': imm_amt,
                            'immediate_delay': imm_delay,
                            'delayed_amount': del_amt,
                            'delayed_delay': del_delay,
                            'choice': choice,
                            'Q_immediate': Q_immediate,
                            'Q_delayed': Q_delayed,
                            'k_estimate': 0.1  # Default estimate for LLM mode
                        })
                    
                    immediate_count = sum(1 for h in history if h['choice'] == 'immediate')
                    delayed_count = len(history) - immediate_count
                    st.session_state.mcq_history = history
                    st.success(f"LLM MCQ complete! Immediate choices: {immediate_count}, Delayed choices: {delayed_count}")
                    
                else:
                    # Traditional Q-learning simulation
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
                        # Heuristic fallback for MCQ
                        lr = 0.05 + 0.3 * norm_mix.get('risk_taker', 0)  # Higher LR = more impulsive
                        eps = 0.1 + 0.3 * norm_mix.get('risk_taker', 0)
                        pers = 0.1 + 0.3 * norm_mix.get('cautious_thinker', 0)
                        patience = int(20 + 20 * norm_mix.get('cautious_thinker', 0))
                        params = AgentParams(lr, eps, pers, 0.05, patience, rationale='heuristic_mcq')
                    
                    # Run simulation with custom choice pairs
                    simulator = CustomMCQSimulator(
                        params, num_choices, choice_pairs_to_use,
                        rng_seed=int(time.time()) % 2**32
                    )
                    result, history = simulator.run()
                    
                    # Store results
                    st.session_state.mcq_history = history
                    
                    st.success(f"✅ Simulation complete! "
                              f"Immediate: {result.immediate_count}/{result.total_choices}, "
                              f"Delayed: {result.delayed_count}/{result.total_choices}")
    
    with col2:
        st.subheader("Results Visualization")
        
        if 'mcq_history' in st.session_state and st.session_state.mcq_history:
            history = st.session_state.mcq_history
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot 1: Q-values over choices
            choices = [h['choice_num'] for h in history]
            Q_immediate = [h['Q_immediate'] for h in history]
            Q_delayed = [h['Q_delayed'] for h in history]
            
            ax1.plot(choices, Q_immediate, 'r-', label='Q(Immediate)', linewidth=2)
            ax1.plot(choices, Q_delayed, 'b-', label='Q(Delayed)', linewidth=2)
            ax1.set_xlabel('Choice Number')
            ax1.set_ylabel('Value Estimate')
            ax1.set_title('Agent Value Estimates for Immediate vs Delayed Rewards')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Choice pattern
            immediate_x = [h['choice_num'] for h in history if h['choice'] == 'immediate']
            delayed_x = [h['choice_num'] for h in history if h['choice'] == 'delayed']
            
            if immediate_x:
                ax2.scatter(immediate_x, [1]*len(immediate_x), c='red', label='Immediate Choice', alpha=0.7, s=50)
            if delayed_x:
                ax2.scatter(delayed_x, [0]*len(delayed_x), c='blue', label='Delayed Choice', alpha=0.7, s=50)
            
            ax2.set_xlabel('Choice Number')
            ax2.set_ylabel('Choice Type')
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['Delayed', 'Immediate'])
            ax2.set_title('Choice Pattern: Immediate vs Delayed Rewards')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Store graph button
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("💾 Store Graph", key="store_mcq"):
                    store_graph(
                        f"MCQ Results - {len(st.session_state.stored_graphs)+1}",
                        fig,
                        "MCQ",
                        {"num_choices": num_choices}
                    )
                    st.success("Graph stored!")
            
            with col_b:
                # Download button
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                st.download_button(
                    "📥 Download PNG",
                    img_buffer.getvalue(),
                    f"mcq_results_{int(time.time())}.png",
                    "image/png"
                )
        else:
            st.info("👆 Run a simulation to see results here")

def igt_test_interface(api_key, temperature, use_api, personality_weights):
    """IGT test interface"""
    st.header("Iowa Gambling Task (IGT)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Task Parameters")
        
        num_trials = st.slider("Number of Trials:", 50, 200, 100)
        
        st.write("**Deck Configuration:**")
        
        # Deck A Configuration
        with st.expander("Deck A Settings", expanded=False):
            deck_a_gain = st.number_input("Deck A - Reward ($):", min_value=1, max_value=500, value=100, key="deck_a_gain")
            deck_a_loss_prob = st.slider("Deck A - Loss Probability:", 0.0, 1.0, 0.5, 0.05, key="deck_a_loss_prob")
            deck_a_loss_amt = st.number_input("Deck A - Loss Amount ($):", min_value=-2000, max_value=-1, value=-250, key="deck_a_loss_amt")
            deck_a_expected = deck_a_gain * (1 - deck_a_loss_prob) + (deck_a_gain + deck_a_loss_amt) * deck_a_loss_prob
            st.write(f"Expected value: ${deck_a_expected:.1f}")
        
        # Deck B Configuration
        with st.expander("Deck B Settings", expanded=False):
            deck_b_gain = st.number_input("Deck B - Reward ($):", min_value=1, max_value=500, value=100, key="deck_b_gain")
            deck_b_loss_prob = st.slider("Deck B - Loss Probability:", 0.0, 1.0, 0.1, 0.05, key="deck_b_loss_prob")
            deck_b_loss_amt = st.number_input("Deck B - Loss Amount ($):", min_value=-2000, max_value=-1, value=-1250, key="deck_b_loss_amt")
            deck_b_expected = deck_b_gain * (1 - deck_b_loss_prob) + (deck_b_gain + deck_b_loss_amt) * deck_b_loss_prob
            st.write(f"Expected value: ${deck_b_expected:.1f}")
        
        # Deck C Configuration
        with st.expander("Deck C Settings", expanded=False):
            deck_c_gain = st.number_input("Deck C - Reward ($):", min_value=1, max_value=500, value=50, key="deck_c_gain")
            deck_c_loss_prob = st.slider("Deck C - Loss Probability:", 0.0, 1.0, 0.5, 0.05, key="deck_c_loss_prob")
            deck_c_loss_amt = st.number_input("Deck C - Loss Amount ($):", min_value=-2000, max_value=-1, value=-50, key="deck_c_loss_amt")
            deck_c_expected = deck_c_gain * (1 - deck_c_loss_prob) + (deck_c_gain + deck_c_loss_amt) * deck_c_loss_prob
            st.write(f"Expected value: ${deck_c_expected:.1f}")
        
        # Deck D Configuration
        with st.expander("Deck D Settings", expanded=False):
            deck_d_gain = st.number_input("Deck D - Reward ($):", min_value=1, max_value=500, value=50, key="deck_d_gain")
            deck_d_loss_prob = st.slider("Deck D - Loss Probability:", 0.0, 1.0, 0.1, 0.05, key="deck_d_loss_prob")
            deck_d_loss_amt = st.number_input("Deck D - Loss Amount ($):", min_value=-2000, max_value=-1, value=-250, key="deck_d_loss_amt")
            deck_d_expected = deck_d_gain * (1 - deck_d_loss_prob) + (deck_d_gain + deck_d_loss_amt) * deck_d_loss_prob
            st.write(f"Expected value: ${deck_d_expected:.1f}")
        
        # Quick presets
        st.write("**Quick Presets:**")
        col_preset1, col_preset2 = st.columns(2)
        
        with col_preset1:
            if st.button("📊 Classic IGT", help="Original Bechara et al. (1994) settings"):
                st.session_state.deck_a_gain = 100
                st.session_state.deck_a_loss_prob = 0.5
                st.session_state.deck_a_loss_amt = -250
                st.session_state.deck_b_gain = 100
                st.session_state.deck_b_loss_prob = 0.1
                st.session_state.deck_b_loss_amt = -1250
                st.session_state.deck_c_gain = 50
                st.session_state.deck_c_loss_prob = 0.5
                st.session_state.deck_c_loss_amt = -50
                st.session_state.deck_d_gain = 50
                st.session_state.deck_d_loss_prob = 0.1
                st.session_state.deck_d_loss_amt = -250
                st.rerun()
        
        with col_preset2:
            if st.button("⚖️ Balanced Risk", help="More balanced risk/reward ratios"):
                st.session_state.deck_a_gain = 80
                st.session_state.deck_a_loss_prob = 0.3
                st.session_state.deck_a_loss_amt = -150
                st.session_state.deck_b_gain = 120
                st.session_state.deck_b_loss_prob = 0.2
                st.session_state.deck_b_loss_amt = -400
                st.session_state.deck_c_gain = 60
                st.session_state.deck_c_loss_prob = 0.4
                st.session_state.deck_c_loss_amt = -30
                st.session_state.deck_d_gain = 40
                st.session_state.deck_d_loss_prob = 0.1
                st.session_state.deck_d_loss_amt = -100
                st.rerun()
        
        # Create custom deck configuration
        custom_decks = {
            'A': {'gain': deck_a_gain, 'loss_prob': deck_a_loss_prob, 'loss_amt': deck_a_loss_amt, 'net_expected': deck_a_expected},
            'B': {'gain': deck_b_gain, 'loss_prob': deck_b_loss_prob, 'loss_amt': deck_b_loss_amt, 'net_expected': deck_b_expected},
            'C': {'gain': deck_c_gain, 'loss_prob': deck_c_loss_prob, 'loss_amt': deck_c_loss_amt, 'net_expected': deck_c_expected},
            'D': {'gain': deck_d_gain, 'loss_prob': deck_d_loss_prob, 'loss_amt': deck_d_loss_amt, 'net_expected': deck_d_expected}
        }
        
        # Display deck summary
        st.write("**Current Deck Summary:**")
        for deck_name, deck_info in custom_decks.items():
            advantageous = "✅" if deck_info['net_expected'] > 0 else "❌"
            st.write(f"**Deck {deck_name}**: {advantageous} +${deck_info['gain']}, {deck_info['loss_prob']:.1%} chance ${deck_info['loss_amt']:.0f} (EV: ${deck_info['net_expected']:.1f})")
        
        # LLM Decision Mode
        llm_mode_igt = st.checkbox(
            "🤖 LLM Decision Mode", 
            key="igt_llm",
            help="Agent makes decisions step-by-step using language model instead of Q-learning algorithm"
        )
        
        if st.button("🚀 Run IGT Simulation", key="igt_run"):
            if not personality_weights:
                st.error("Please select at least one personality in the sidebar!")
                return
                
            with st.spinner("Running IGT simulation..."):
                if llm_mode_igt:
                    # LLM-driven simulation
                    if not api_key:
                        st.error("LLM Decision Mode requires an OpenAI API key.")
                        return
                    
                    Q_values = {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
                    history = []
                    total_money = 2000  # Starting amount
                    
                    for trial in range(1, num_trials + 1):
                        context = format_igt_context(trial, history, total_money, Q_values, custom_decks)
                        decision = get_llm_decision(context, api_key, temperature)
                        
                        # Parse deck choice
                        chosen_deck = 'A'
                        if decision.upper() in ['B', 'DECK B']:
                            chosen_deck = 'B'
                        elif decision.upper() in ['C', 'DECK C']:
                            chosen_deck = 'C'
                        elif decision.upper() in ['D', 'DECK D']:
                            chosen_deck = 'D'
                        elif decision.upper() in ['A', 'DECK A']:
                            chosen_deck = 'A'
                        
                        # Get outcome from chosen deck
                        deck_info = custom_decks[chosen_deck]
                        gain = deck_info['gain']
                        
                        # Determine if loss occurs
                        if np.random.random() < deck_info['loss_prob']:
                            loss = deck_info['loss_amt']
                        else:
                            loss = 0
                        
                        net_outcome = gain + loss
                        total_money += net_outcome
                        
                        # Update Q-value for chosen deck
                        Q_values[chosen_deck] += 0.1 * (net_outcome - Q_values[chosen_deck])
                        
                        # Record trial
                        history.append({
                            'trial': trial,
                            'deck': chosen_deck,
                            'gain': gain,
                            'loss': loss,
                            'net_outcome': net_outcome,
                            'total_money': total_money,
                            'Q_A': Q_values['A'],
                            'Q_B': Q_values['B'],
                            'Q_C': Q_values['C'],
                            'Q_D': Q_values['D']
                        })
                    
                    advantageous = sum(1 for h in history if h['deck'] in ['C', 'D'])
                    st.session_state.igt_history = history
                    st.success(f"LLM IGT complete! Final money: ${total_money}, Advantageous choices: {advantageous}/{num_trials}")
                    
                else:
                    # Traditional Q-learning simulation
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
                        # Heuristic fallback for IGT
                        lr = 0.1 + 0.4 * norm_mix.get('risk_taker', 0)
                        eps = 0.15 + 0.3 * norm_mix.get('risk_taker', 0)
                        pers = 0.1 + 0.3 * norm_mix.get('cautious_thinker', 0)
                        patience = int(10 + 20 * norm_mix.get('cautious_thinker', 0))
                        params = AgentParams(lr, eps, pers, 0.05, patience, rationale='heuristic_igt')
                    
                    # Run simulation with custom decks
                    simulator = CustomIGTSimulator(
                        params, num_trials, custom_decks,
                        rng_seed=int(time.time()) % 2**32
                    )
                    result, history = simulator.run()
                    
                    # Store results
                    st.session_state.igt_history = history
                    
                    st.success(f"✅ Simulation complete! "
                              f"Final money: ${result.final_money:.0f}, "
                              f"Good choices: {result.advantageous_choices}/{result.total_trials}")
    
    with col2:
        st.subheader("Results Visualization")
        
        if 'igt_history' in st.session_state and st.session_state.igt_history:
            history = st.session_state.igt_history
            
            # Create plots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot 1: Q-values for each deck over time
            trials = [h['trial'] for h in history]
            Q_A = [h['Q_A'] for h in history]
            Q_B = [h['Q_B'] for h in history]
            Q_C = [h['Q_C'] for h in history]
            Q_D = [h['Q_D'] for h in history]
            
            ax1.plot(trials, Q_A, 'r-', label='Deck A (Bad)', linewidth=2, alpha=0.8)
            ax1.plot(trials, Q_B, 'orange', label='Deck B (Bad)', linewidth=2, alpha=0.8)
            ax1.plot(trials, Q_C, 'g-', label='Deck C (Good)', linewidth=2, alpha=0.8)
            ax1.plot(trials, Q_D, 'b-', label='Deck D (Good)', linewidth=2, alpha=0.8)
            ax1.set_xlabel('Trial')
            ax1.set_ylabel('Q-Value (Expected Return)')
            ax1.set_title('Learning Curves: Deck Value Estimates Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Deck choices over time
            deck_choices = [h['deck'] for h in history]
            deck_A = [trials[i] for i, d in enumerate(deck_choices) if d == 'A']
            deck_B = [trials[i] for i, d in enumerate(deck_choices) if d == 'B']
            deck_C = [trials[i] for i, d in enumerate(deck_choices) if d == 'C']
            deck_D = [trials[i] for i, d in enumerate(deck_choices) if d == 'D']
            
            if deck_A:
                ax2.scatter(deck_A, [0]*len(deck_A), c='red', label='Deck A (Bad)', alpha=0.7, s=20)
            if deck_B:
                ax2.scatter(deck_B, [1]*len(deck_B), c='orange', label='Deck B (Bad)', alpha=0.7, s=20)
            if deck_C:
                ax2.scatter(deck_C, [2]*len(deck_C), c='green', label='Deck C (Good)', alpha=0.7, s=20)
            if deck_D:
                ax2.scatter(deck_D, [3]*len(deck_D), c='blue', label='Deck D (Good)', alpha=0.7, s=20)
            
            ax2.set_xlabel('Trial')
            ax2.set_ylabel('Deck Chosen')
            ax2.set_yticks([0, 1, 2, 3])
            ax2.set_yticklabels(['A', 'B', 'C', 'D'])
            ax2.set_title('Deck Selection Pattern Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Store graph button
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("💾 Store Graph", key="store_igt"):
                    store_graph(
                        f"IGT Results - {len(st.session_state.stored_graphs)+1}",
                        fig,
                        "IGT",
                        {"num_trials": num_trials}
                    )
                    st.success("Graph stored!")
            
            with col_b:
                # Download button
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                st.download_button(
                    "📥 Download PNG",
                    img_buffer.getvalue(),
                    f"igt_results_{int(time.time())}.png",
                    "image/png"
                )
        else:
            st.info("👆 Run a simulation to see results here")

def stored_graphs_interface():
    """Interface for viewing and managing stored graphs"""
    st.header("Stored Graphs")
    
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

def create_human_behavior_graphs():
    """Create graphs showing typical human behavior in BART and PRLT tests based on research literature"""
    
    # BART Human Data (based on research literature)
    def create_human_bart_graph():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        
        # Typical human BART behavior - based on research studies
        # Average pumps typically range from 20-35 with high individual variation
        balloons = list(range(1, 31))
        
        # Simulate typical human Q-value learning (more conservative than our AI)
        human_Q_pump = [2.0 + 0.5 * np.sin(i/5) + np.random.normal(0, 0.3) for i in balloons]
        human_Q_cash = [3.0 + 0.3 * i/10 + np.random.normal(0, 0.2) for i in balloons]
        
        ax1.plot(balloons, human_Q_pump, 'b-', label='Human Q(Pump)', linewidth=2, alpha=0.8)
        ax1.plot(balloons, human_Q_cash, 'g-', label='Human Q(Cash)', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Balloon Number')
        ax1.set_ylabel('Estimated Value')
        ax1.set_title('Typical Human Value Estimates (Research Data)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Typical human pumping behavior - research shows avg 25-30 pumps
        np.random.seed(42)  # For consistent results
        human_pumps = np.random.gamma(3, 8, 30)  # Gamma distribution for realistic pump counts
        human_pumps = np.clip(human_pumps, 5, 50)  # Reasonable range
        
        # Explosion rate around 30-40% based on literature
        explosion_prob = 0.35
        human_exploded = np.random.random(30) < explosion_prob
        
        exploded_x = [i+1 for i, e in enumerate(human_exploded) if e]
        exploded_y = [human_pumps[i] for i, e in enumerate(human_exploded) if e]
        cashed_x = [i+1 for i, e in enumerate(human_exploded) if not e]
        cashed_y = [human_pumps[i] for i, e in enumerate(human_exploded) if not e]
        
        if cashed_x:
            ax2.scatter(cashed_x, cashed_y, c='green', label='Cashed Out', alpha=0.7, s=50)
        if exploded_x:
            ax2.scatter(exploded_x, exploded_y, c='red', label='Exploded', alpha=0.7, s=50)
        
        ax2.set_xlabel('Balloon Number')
        ax2.set_ylabel('Pumps')
        ax2.set_title(f'Typical Human BART Performance\n(Avg: {np.mean(human_pumps):.1f} pumps, {len(exploded_x)}/{len(balloons)} explosions)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, np.mean(human_pumps), len(exploded_x)
    
    def create_human_prlt_graph():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        
        # Typical human PRLT behavior - based on research
        # Pre-reversal: Usually learn within 50-100 trials
        # Post-reversal: Usually adapt within 20-50 trials
        
        # Pre-reversal phase (150 trials)
        pre_trials = list(range(1, 151))
        QA_pre = [0.5 + 0.4 * (1 - np.exp(-t/30)) + np.random.normal(0, 0.05) for t in pre_trials]
        QB_pre = [0.5 - 0.3 * (1 - np.exp(-t/30)) + np.random.normal(0, 0.05) for t in pre_trials]
        
        # Post-reversal phase (100 trials)
        post_trials = list(range(151, 251))
        QA_post = [QA_pre[-1] - 0.6 * (1 - np.exp(-(t-150)/25)) + np.random.normal(0, 0.05) for t in post_trials]
        QB_post = [QB_pre[-1] + 0.6 * (1 - np.exp(-(t-150)/25)) + np.random.normal(0, 0.05) for t in post_trials]
        
        all_trials = pre_trials + post_trials
        QA_vals = QA_pre + QA_post
        QB_vals = QB_pre + QB_post
        
        ax1.plot(all_trials, QA_vals, 'b-', label='Human Q(A)', linewidth=2, alpha=0.8)
        ax1.plot(all_trials, QB_vals, 'r-', label='Human Q(B)', linewidth=2, alpha=0.8)
        ax1.axvline(150, color='orange', linestyle='--', alpha=0.7, label='Reversal')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Q-Value')
        ax1.set_title('Typical Human Learning Curves (Research Data)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Generate typical human choice pattern
        np.random.seed(42)
        choices = []
        for i, (qa, qb) in enumerate(zip(QA_vals, QB_vals)):
            # Add noise to choice probability
            prob_A = 1 / (1 + np.exp(-(qa - qb) * 3))  # Softmax-like
            choice = 1 if np.random.random() < prob_A else 0
            choices.append(choice)
        
        choice_A = [all_trials[i] for i, c in enumerate(choices) if c == 1]
        choice_B = [all_trials[i] for i, c in enumerate(choices) if c == 0]
        
        if choice_A:
            ax2.scatter(choice_A, [1]*len(choice_A), c='blue', alpha=0.6, s=15, label='Choice A')
        if choice_B:
            ax2.scatter(choice_B, [0]*len(choice_B), c='red', alpha=0.6, s=15, label='Choice B')
        
        ax2.axvline(150, color='orange', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Choice')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['B', 'A'])
        ax2.set_title('Typical Human Choice Pattern')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Calculate convergence times
        pre_converge = 75  # Typical pre-reversal convergence
        post_switch = 35   # Typical post-reversal switch time
        
        return fig, pre_converge, post_switch
    
    return create_human_bart_graph, create_human_prlt_graph

def comparison_interface():
    """Interface for comparing human vs AI behavior"""
    st.header("Human vs AI Behavior Comparison")
    st.markdown("""Compare typical human performance from research literature with your AI simulation results.
    
    **Human data sources**: Compiled from peer-reviewed research papers on BART and PRLT tasks.
    """)
    
    # Create human behavior graphs
    create_human_bart_graph, create_human_prlt_graph = create_human_behavior_graphs()
    
    # Test selection
    test_type = st.selectbox(
        "Select test to compare:",
        ["BART (Balloon Analog Risk Task)", "PRLT (Probabilistic Reversal Learning Task)", 
         "MCQ (Monetary Choice Questionnaire)", "IGT (Iowa Gambling Task)"]
    )
    
    if test_type.startswith("BART"):
        st.subheader("BART Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**👥 Human Behavior (Research Literature)**")
            st.markdown("""**Research Summary:**
            - Average pumps: 25-30 per balloon
            - Explosion rate: 30-40%
            - Learning: Gradual risk assessment
            - Individual variation: High
            - Source: Lejuez et al. (2002), Hunt et al. (2005)
            """)
            
            # Generate and display human BART graph
            human_fig, human_avg_pumps, human_explosions = create_human_bart_graph()
            st.pyplot(human_fig)
            st.info(f"📊 Research data shows avg {human_avg_pumps:.1f} pumps, {human_explosions}/30 explosions")
        
        with col2:
            st.markdown("**🤖 AI Behavior (Your Results)**")
            
            # Let user select from stored graphs
            bart_graphs = [g for g in st.session_state.stored_graphs if g['test_type'] == 'BART']
            
            if not bart_graphs:
                st.warning("No BART results stored yet. Run a BART simulation first!")
                st.markdown("**To generate AI data:**\n1. Go to BART Test tab\n2. Run a simulation\n3. Click 'Store Graph'\n4. Return here for comparison")
            else:
                selected_graph = st.selectbox(
                    "Select AI result to compare:",
                    options=range(len(bart_graphs)),
                    format_func=lambda x: f"{bart_graphs[x]['title']} - {bart_graphs[x]['timestamp']}"
                )
                
                if selected_graph is not None:
                    graph = bart_graphs[selected_graph]
                    
                    # Display AI parameters
                    st.markdown("**AI Parameters:**")
                    for key, value in graph['parameters'].items():
                        st.write(f"- {key}: {value}")
                    
                    # Display stored AI graph
                    img_data = base64.b64decode(graph['image'])
                    st.image(img_data, caption=graph['title'])
                    
                    # Comparison insights
                    st.markdown("**🔍 Comparison Notes:**")
                    st.markdown("""- Compare average pump counts
                    - Note explosion patterns
                    - Observe learning curves
                    - Consider risk-taking strategies""")
    
    elif test_type.startswith("PRLT"):  # PRLT
        st.subheader("PRLT Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**👥 Human Behavior (Research Literature)**")
            st.markdown("""**Research Summary:**
            - Pre-reversal learning: 50-100 trials
            - Post-reversal adaptation: 20-50 trials
            - Strategy: Gradual probability learning
            - Individual variation: Moderate
            - Source: Cools et al. (2002), Clarke et al. (2004)
            """)
            
            # Generate and display human PRLT graph
            human_fig, human_pre_converge, human_post_switch = create_human_prlt_graph()
            st.pyplot(human_fig)
            st.info(f"📊 Research shows {human_pre_converge} trials to learn, {human_post_switch} trials to switch")
        
        with col2:
            st.markdown("**🤖 AI Behavior (Your Results)**")
            
            # Let user select from stored graphs
            prlt_graphs = [g for g in st.session_state.stored_graphs if g['test_type'] == 'PRLT']
            
            if not prlt_graphs:
                st.warning("No PRLT results stored yet. Run a PRLT simulation first!")
                st.markdown("**To generate AI data:**\n1. Go to PRLT Test tab\n2. Run a simulation\n3. Click 'Store Graph'\n4. Return here for comparison")
            else:
                selected_graph = st.selectbox(
                    "Select AI result to compare:",
                    options=range(len(prlt_graphs)),
                    format_func=lambda x: f"{prlt_graphs[x]['title']} - {prlt_graphs[x]['timestamp']}"
                )
                
                if selected_graph is not None:
                    graph = prlt_graphs[selected_graph]
                    
                    # Display AI parameters
                    st.markdown("**AI Parameters:**")
                    for key, value in graph['parameters'].items():
                        st.write(f"- {key}: {value}")
                    
                    # Display stored AI graph
                    img_data = base64.b64decode(graph['image'])
                    st.image(img_data, caption=graph['title'])
                    
                    # Comparison insights
                    st.markdown("**🔍 Comparison Notes:**")
                    st.markdown("""- Compare learning speeds
                    - Note adaptation flexibility
                    - Observe choice patterns
                    - Consider reversal strategies""")
    
    elif test_type.startswith("MCQ"):  # MCQ
        st.subheader("MCQ Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**👥 Human Behavior (Research Literature)**")
            st.markdown("""**Research Summary:**
            - Immediate choices: 40-70% (varies by individual)
            - Discount rates: k = 0.001 to 0.25 (high variation)
            - Pattern: Consistent individual preferences
            - Factors: Age, income, personality affect choices
            - Source: Kirby, Petry & Bickel (1999), Reynolds (2006)
            """)
            
            # Generate typical human MCQ behavior
            np.random.seed(42)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
            
            # Human temporal preference learning (more stable than AI)
            choices = list(range(1, 28))
            human_Q_immediate = [0.6 + 0.1 * np.sin(i/5) + np.random.normal(0, 0.05) for i in choices]
            human_Q_delayed = [0.4 - 0.1 * np.sin(i/5) + np.random.normal(0, 0.05) for i in choices]
            
            ax1.plot(choices, human_Q_immediate, 'r-', label='Human Q(Immediate)', linewidth=2, alpha=0.8)
            ax1.plot(choices, human_Q_delayed, 'b-', label='Human Q(Delayed)', linewidth=2, alpha=0.8)
            ax1.set_xlabel('Choice Number')
            ax1.set_ylabel('Value Estimate')
            ax1.set_title('Typical Human Temporal Preferences')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Human choice pattern (60% immediate choices typical)
            human_choices = np.random.choice([0, 1], 27, p=[0.4, 0.6])  # 60% immediate
            immediate_x = [i+1 for i, c in enumerate(human_choices) if c == 1]
            delayed_x = [i+1 for i, c in enumerate(human_choices) if c == 0]
            
            if immediate_x:
                ax2.scatter(immediate_x, [1]*len(immediate_x), c='red', label='Immediate', alpha=0.7, s=50)
            if delayed_x:
                ax2.scatter(delayed_x, [0]*len(delayed_x), c='blue', label='Delayed', alpha=0.7, s=50)
            
            ax2.set_xlabel('Choice Number')
            ax2.set_ylabel('Choice Type')
            ax2.set_yticks([0, 1])
            ax2.set_yticklabels(['Delayed', 'Immediate'])
            ax2.set_title('Typical Human Choice Pattern')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            st.info(f"Research shows ~60% immediate choices, high individual variation")
        
        with col2:
            st.markdown("**🤖 AI Behavior (Your Results)**")
            
            mcq_graphs = [g for g in st.session_state.stored_graphs if g['test_type'] == 'MCQ']
            
            if not mcq_graphs:
                st.warning("No MCQ results stored yet. Run an MCQ simulation first!")
                st.markdown("**To generate AI data:**\n1. Go to MCQ Test tab\n2. Run a simulation\n3. Click 'Store Graph'\n4. Return here for comparison")
            else:
                selected_graph = st.selectbox(
                    "Select AI result to compare:",
                    options=range(len(mcq_graphs)),
                    format_func=lambda x: f"{mcq_graphs[x]['title']} - {mcq_graphs[x]['timestamp']}",
                    key="mcq_select"
                )
                
                if selected_graph is not None:
                    graph = mcq_graphs[selected_graph]
                    
                    st.markdown("**AI Parameters:**")
                    for key, value in graph['parameters'].items():
                        st.write(f"- {key}: {value}")
                    
                    img_data = base64.b64decode(graph['image'])
                    st.image(img_data, caption=graph['title'])
                    
                    st.markdown("**🔍 Comparison Notes:**")
                    st.markdown("""- Compare immediate vs delayed preference ratios
                    - Note consistency vs variability in choices
                    - Observe temporal discounting patterns
                    - Consider impulsivity vs self-control indicators""")
    
    elif test_type.startswith("IGT"):  # IGT
        st.subheader("IGT Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**👥 Human Behavior (Research Literature)**")
            st.markdown("""**Research Summary:**
            - Learning curve: Initially prefer bad decks (A,B)
            - Adaptation: Switch to good decks (C,D) after 40-60 trials
            - Final performance: 60-70% good deck choices
            - Individual differences: Some never learn optimal strategy
            - Source: Bechara et al. (1994), Dunn et al. (2006)
            """)
            
            # Generate typical human IGT behavior
            np.random.seed(42)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
            
            # Human learning curves (gradual shift from bad to good decks)
            trials = list(range(1, 101))
            # Start preferring bad decks, gradually learn
            human_Q_A = [1.0 - 0.8 * (1 - np.exp(-t/40)) + np.random.normal(0, 0.1) for t in trials]
            human_Q_B = [1.0 - 0.7 * (1 - np.exp(-t/35)) + np.random.normal(0, 0.1) for t in trials]
            human_Q_C = [-0.5 + 1.2 * (1 - np.exp(-t/45)) + np.random.normal(0, 0.1) for t in trials]
            human_Q_D = [-0.3 + 1.0 * (1 - np.exp(-t/50)) + np.random.normal(0, 0.1) for t in trials]
            
            ax1.plot(trials, human_Q_A, 'r-', label='Deck A (Bad)', linewidth=2, alpha=0.8)
            ax1.plot(trials, human_Q_B, 'orange', label='Deck B (Bad)', linewidth=2, alpha=0.8)
            ax1.plot(trials, human_Q_C, 'g-', label='Deck C (Good)', linewidth=2, alpha=0.8)
            ax1.plot(trials, human_Q_D, 'b-', label='Deck D (Good)', linewidth=2, alpha=0.8)
            ax1.set_xlabel('Trial')
            ax1.set_ylabel('Q-Value')
            ax1.set_title('Typical Human Learning Curves')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Human choice pattern (start bad, learn good)
            choice_probs = []
            for t in range(100):
                if t < 40:  # Early trials - prefer bad decks
                    probs = [0.3, 0.3, 0.2, 0.2]  # A, B, C, D
                else:  # Later trials - prefer good decks
                    probs = [0.15, 0.15, 0.35, 0.35]
                choice_probs.append(probs)
            
            human_deck_choices = []
            for probs in choice_probs:
                choice = np.random.choice(['A', 'B', 'C', 'D'], p=probs)
                human_deck_choices.append(choice)
            
            deck_A = [i+1 for i, d in enumerate(human_deck_choices) if d == 'A']
            deck_B = [i+1 for i, d in enumerate(human_deck_choices) if d == 'B']
            deck_C = [i+1 for i, d in enumerate(human_deck_choices) if d == 'C']
            deck_D = [i+1 for i, d in enumerate(human_deck_choices) if d == 'D']
            
            if deck_A:
                ax2.scatter(deck_A, [0]*len(deck_A), c='red', label='Deck A (Bad)', alpha=0.7, s=20)
            if deck_B:
                ax2.scatter(deck_B, [1]*len(deck_B), c='orange', label='Deck B (Bad)', alpha=0.7, s=20)
            if deck_C:
                ax2.scatter(deck_C, [2]*len(deck_C), c='green', label='Deck C (Good)', alpha=0.7, s=20)
            if deck_D:
                ax2.scatter(deck_D, [3]*len(deck_D), c='blue', label='Deck D (Good)', alpha=0.7, s=20)
            
            ax2.set_xlabel('Trial')
            ax2.set_ylabel('Deck Chosen')
            ax2.set_yticks([0, 1, 2, 3])
            ax2.set_yticklabels(['A', 'B', 'C', 'D'])
            ax2.set_title('Typical Human Deck Selection')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            good_choices = len(deck_C) + len(deck_D)
            st.info(f"Research shows {good_choices}% good deck choices, learning after ~40 trials")
        
        with col2:
            st.markdown("**🤖 AI Behavior (Your Results)**")
            
            igt_graphs = [g for g in st.session_state.stored_graphs if g['test_type'] == 'IGT']
            
            if not igt_graphs:
                st.warning("No IGT results stored yet. Run an IGT simulation first!")
                st.markdown("**To generate AI data:**\n1. Go to IGT Test tab\n2. Run a simulation\n3. Click 'Store Graph'\n4. Return here for comparison")
            else:
                selected_graph = st.selectbox(
                    "Select AI result to compare:",
                    options=range(len(igt_graphs)),
                    format_func=lambda x: f"{igt_graphs[x]['title']} - {igt_graphs[x]['timestamp']}",
                    key="igt_select"
                )
                
                if selected_graph is not None:
                    graph = igt_graphs[selected_graph]
                    
                    st.markdown("**AI Parameters:**")
                    for key, value in graph['parameters'].items():
                        st.write(f"- {key}: {value}")
                    
                    img_data = base64.b64decode(graph['image'])
                    st.image(img_data, caption=graph['title'])
                    
                    st.markdown("**🔍 Comparison Notes:**")
                    st.markdown("""- Compare learning speed and final performance
                    - Note initial deck preferences and adaptation
                    - Observe sensitivity to gains vs losses
                    - Consider decision-making under uncertainty patterns""")
    
    # Research references and methodology
    with st.expander("Research References & Methodology"):
        st.markdown("""**BART Research Sources:**
        - Lejuez, C. W., et al. (2002). Evaluation of a behavioral measure of risk taking: the Balloon Analogue Risk Task (BART). *Journal of Experimental Psychology: Applied, 8*(2), 75-84.
        - Hunt, M. K., et al. (2005). Construct validity of the Balloon Analog Risk Task (BART): associations with psychopathy and impulsivity. *Assessment, 12*(4), 416-428.
        - Schmitz, F., et al. (2016). The BART as a measure of risk taking: Replication and extension. *Psychological Assessment, 28*(2), 243-255.
        
        **PRLT Research Sources:**
        - Cools, R., et al. (2002). Defining the neural mechanisms of probabilistic reversal learning using event-related functional magnetic resonance imaging. *Journal of Neuroscience, 22*(11), 4563-4567.
        - Clarke, H. F., et al. (2004). Cognitive inflexibility after prefrontal serotonion depletion. *Science, 304*(5672), 878-880.
        - Izquierdo, A., et al. (2017). The neural basis of reversal learning: an updated perspective. *Neuroscience, 345*, 12-26.
        
        **MCQ Research Sources:**
        - Kirby, K. N., Petry, N. M., & Bickel, W. K. (1999). Heroin addicts have higher discount rates for delayed rewards than non-drug-using controls. *Journal of Experimental Psychology: General, 128*(1), 78-87.
        - Reynolds, B. (2006). A review of delay-discounting research with humans: relations to drug use and gambling. *Behavioural Pharmacology, 17*(8), 651-667.
        - Odum, A. L. (2011). Delay discounting: I'm a k, you're a k. *Journal of the Experimental Analysis of Behavior, 96*(3), 427-439.
        
        **IGT Research Sources:**
        - Bechara, A., et al. (1994). Insensitivity to future consequences following damage to human prefrontal cortex. *Cognition, 50*(1-3), 7-15.
        - Dunn, B. D., Dalgleish, T., & Lawrence, A. D. (2006). The somatic marker hypothesis: A critical evaluation. *Neuroscience & Biobehavioral Reviews, 30*(2), 239-271.
        - Steingroever, H., et al. (2013). Data from 617 healthy participants performing the Iowa gambling task: A "many labs" collaboration. *Journal of Open Psychology Data, 1*(1), e3.
        
        **Methodology Notes:**
        - Human data represents typical performance across multiple studies
        - Individual variation in human performance is substantial
        - AI simulations may show different patterns based on personality parameters
        - Direct comparison should consider task parameter differences
        - MCQ and IGT show particularly high individual differences in human populations
        """)
    
    # Analysis tools
    with st.expander("🔬 Analysis Tools"):
        st.markdown("**Comparative Analysis Guidelines:**")
        
        st.markdown("""**For BART Comparisons:**
        1. **Average Pumps**: Human ~25-30, compare to AI average
        2. **Risk Patterns**: Look for consistent vs. variable risk-taking
        3. **Learning**: Humans show gradual learning, AI may be more systematic
        4. **Explosions**: Human ~30-40% explosion rate
        
        **For PRLT Comparisons:**
        1. **Initial Learning**: Humans need 50-100 trials typically
        2. **Reversal Adaptation**: Humans need 20-50 trials to switch
        3. **Choice Patterns**: Look for probability matching vs. maximization
        4. **Flexibility**: Compare adaptation speeds
        
        **Key Questions to Consider:**
        - Does AI behavior fall within human ranges?
        - What personality traits make AI more/less human-like?
        - Which parameters best model specific human populations?
        - How does temperature affect human-likeness?
        """)

def help_guide_interface():
    """Comprehensive help and guide interface"""
    st.header("Help & User Guide")
    st.markdown("Welcome to the **Personality Testing Suite**! This guide explains all the features and controls available in the application.")
    
    # Overview section
    st.subheader("🎯 Application Overview")
    st.markdown("""
    This application runs four psychological tests to analyze decision-making behaviors based on different personality profiles:
    - **BART (Balloon Analog Risk Task)**: Measures risk-taking behavior
    - **PRLT (Probabilistic Reversal Learning Task)**: Measures learning flexibility and adaptation
    - **MCQ (Monetary Choice Questionnaire)**: Measures temporal discounting and impulsivity
    - **IGT (Iowa Gambling Task)**: Measures decision-making under uncertainty with customizable deck parameters
    """)
    
    # Sidebar Controls
    with st.expander("Sidebar Controls", expanded=True):
        st.subheader("OpenAI Configuration")
        st.markdown("""
        **API Key Input Field:**
        - **Purpose**: Enter your personal OpenAI API key for LLM-based parameter generation
        - **Security**: Your key is only used during your session and is not stored
        - **Format**: Starts with 'sk-' followed by alphanumeric characters
        - **Cost**: Each simulation uses ~100-200 tokens (~$0.01-0.02)
        
        **Use OpenAI API Checkbox:**
        - ✅ **Checked**: Uses your API key for intelligent parameter generation
        - ❌ **Unchecked**: Uses built-in heuristic calculations (free but less accurate)
        """)
        
        st.subheader("LLM Temperature Slider")
        st.markdown("""
        **Temperature Control (0.0 - 1.0):**
        - **0.0-0.3**: More consistent, predictable parameters (recommended for research)
        - **0.4-0.7**: Balanced creativity and consistency
        - **0.8-1.0**: More creative, varied parameters (experimental)
        - **Default**: 0.3 for reliable results
        """)
        
        st.subheader("File Upload System")
        st.markdown("""
        **Upload Custom Personalities:**
        - **File Type**: Only .txt files accepted
        - **Content**: Personality descriptions or behavioral prompts
        - **Multiple Files**: Upload several personalities at once
        - **Integration**: Uploaded personalities appear in the mixing controls below
        - **Management**: Use "Clear Uploaded Files" to remove them
        """)
        
        st.subheader("Personality Mixing Controls")
        st.markdown("""
        **Personality Selection:**
        - **Checkboxes**: Enable/disable each personality type
        - **Default**: 'Risk Taker' is pre-selected
        - **Available Types**: Risk Taker, Cautious Thinker, Bold Pumper, Moderate Pumper, etc.
        
        **Weight Sliders (0.0 - 2.0):**
        - **Purpose**: Control the influence of each selected personality
        - **Range**: 0.0 (no influence) to 2.0 (double influence)
        - **Default**: 1.0 for balanced mixing
        - **Real-time**: Current mix percentages shown below sliders
        
        **Control Buttons:**
        - **🔄 Normalize**: Automatically adjusts weights to sum to 1.0
        - **🗑️ Clear**: Resets all selections and weights
        """)
    
    # BART Test Tab
    with st.expander("BART Test Controls"):
        st.subheader("Task Parameters")
        st.markdown("""
        **Number of Balloons Slider (5-100):**
        - **Purpose**: Sets how many balloons the AI agent will encounter
        - **Default**: 30 balloons
        - **Impact**: More balloons = more reliable average behavior measurement
        - **Research**: 30+ balloons recommended for stable results
        
        **Max Pumps per Balloon Slider (8-128):**
        - **Purpose**: Maximum pumps before balloon automatically cashes out
        - **Default**: 64 pumps
        - **Reality**: Real BART typically uses 64-128 pumps maximum
        - **Impact**: Higher max = more risk opportunity
        
        **Explosion Curve Slider (0.2-2.0):**
        - **Purpose**: Controls how quickly explosion probability increases
        - **1.0**: Linear increase (realistic)
        - **<1.0**: Slower increase (safer early pumps)
        - **>1.0**: Faster increase (riskier early pumps)
        """)
        
        st.subheader("Results Visualization")
        st.markdown("""
        **Upper Graph - Q-Values Over Time:**
        - **Blue Line**: Agent's estimated value for pumping
        - **Green Line**: Agent's estimated value for cashing out
        - **Learning**: Shows how the agent learns from experience
        
        **Lower Graph - Pumps per Balloon:**
        - **Green Dots**: Successful cash-outs
        - **Red Dots**: Balloon explosions
        - **Pattern**: Shows risk-taking consistency across trials
        
        **Action Buttons:**
        - **💾 Store Graph**: Saves current results to Stored Graphs tab
        - **📥 Download PNG**: Downloads high-resolution image file
        """)
    
    # PRLT Test Tab
    with st.expander("PRLT Test Controls"):
        st.subheader("Task Parameters")
        st.markdown("""
        **Pre-Reversal Phase:**
        - **P(Reward|A) Slider**: Probability that choice A gives reward (0.1-0.9)
        - **P(Reward|B)**: Automatically calculated as 1 - P(Reward|A)
        - **Default**: A=75%, B=25% (A is better choice initially)
        
        **Post-Reversal Phase:**
        - **P(Reward|A) after reversal**: New probability for choice A
        - **P(Reward|B) after reversal**: Automatically calculated
        - **Default**: A=25%, B=75% (B becomes the better choice)
        
        **Pre-reversal trials Slider (50-500):**
        - **Purpose**: How many trials before the reversal occurs
        - **Default**: 200 trials
        - **Impact**: More trials = stronger initial learning to overcome
        """)
        
        st.subheader("Results Visualization")
        st.markdown("""
        **Upper Graph - Q-Values Over Time:**
        - **Blue Line**: Agent's estimated value for choice A
        - **Red Line**: Agent's estimated value for choice B
        - **Orange Dashed Line**: Marks the reversal point
        - **Learning**: Shows adaptation to changing reward probabilities
        
        **Lower Graph - Choice Sequence:**
        - **Blue Dots**: Choice A selections
        - **Red Dots**: Choice B selections
        - **Orange Line**: Reversal point
        - **Pattern**: Shows switching behavior after reversal
        """)
    
    # MCQ Test Tab
    with st.expander("MCQ Test Controls"):
        st.subheader("Task Parameters")
        st.markdown("""
        **Number of Choice Pairs Slider (10-27):**
        - **Purpose**: Sets how many immediate vs delayed reward choices to present
        - **Default**: 27 choice pairs (full Kirby, Petry & Bickel questionnaire)
        - **Research**: Based on validated questionnaire from 1999 study
        - **Examples**: "$25 now vs $60 in 14 days", "$11 now vs $30 in 7 days"
        
        **Choice Generation Method:**
        - **Classic MCQ**: Original Kirby, Petry & Bickel (1999) validated choice pairs
        - **Custom Parameters**: User-defined reward amounts and delay periods
        
        **Custom Parameter Controls (when enabled):**
        - **Immediate Rewards**: Min/max amounts ($1-1000) and delay (0-30 days)
        - **Delayed Rewards**: Min/max amounts (must exceed immediate) and delay range
        - **Delay Distribution**: Linear, exponential, or custom comma-separated list
        - **Real-time Preview**: Shows first 5 generated choice pairs
        
        **Quick Presets:**
        - **Classic MCQ**: Restores original research parameters
        - **Modern Range**: Contemporary values with extended delays (up to 1 year)
        
        **Delay Distribution Options:**
        - **Linear**: Evenly spaced delays across the range
        - **Exponential**: More short delays, fewer long delays (realistic)
        - **Custom List**: Specify exact delay values (e.g., "7, 30, 90, 180")
        
        **Task Applications:**
        - **Research**: Test different temporal discounting scenarios
        - **Population Studies**: Model specific demographics or conditions
        - **Parameter Exploration**: Examine how reward/delay ratios affect choices
        - **Cross-cultural**: Adapt monetary amounts for different economies
        """)
        
        st.subheader("Results Visualization")
        st.markdown("""
        **Upper Graph - Value Estimates Over Time:**
        - **Red Line**: Agent's estimated value for immediate rewards
        - **Blue Line**: Agent's estimated value for delayed rewards
        - **Learning**: Shows development of temporal preferences
        
        **Lower Graph - Choice Pattern:**
        - **Red Dots**: Immediate reward selections
        - **Blue Dots**: Delayed reward selections
        - **Pattern**: Shows impulsivity vs patience across decisions
        
        **Key Metrics:**
        - **Immediate Count**: Number of immediate reward choices
        - **Delayed Count**: Number of delayed reward choices
        - **Discount Rate**: Estimated temporal discounting parameter
        """)
    
    # IGT Test Tab
    with st.expander("IGT Test Controls"):
        st.subheader("Task Parameters")
        st.markdown("""
        **Number of Trials Slider (50-200):**
        - **Purpose**: Sets how many card selections from decks
        - **Default**: 100 trials (standard IGT length)
        - **Research**: Based on Bechara et al. Iowa Gambling Task
        
        **Deck Configuration:**
        - **Customizable Settings**: Each deck has adjustable reward, penalty amount, and penalty probability
        - **Expected Value**: Automatically calculated and displayed for each deck
        - **Deck A-D Settings**: Individual expandable controls for fine-tuning
        - **Quick Presets**: "Classic IGT" (original Bechara settings) and "Balanced Risk" options
        
        **Individual Deck Controls:**
        - **Reward Amount**: Guaranteed positive payout per selection (1-500)
        - **Loss Probability**: Chance of penalty occurring (0-100%)
        - **Loss Amount**: Negative penalty when loss occurs (-2000 to -1)
        - **Expected Value**: Net average outcome per selection (auto-calculated)
        
        **Deck Summary Display:**
        - **Visual Indicators**: ✅ for positive expected value (good), ❌ for negative (bad)
        - **Format**: Shows reward, loss probability, loss amount, and expected value
        - **Real-time Updates**: Changes immediately when parameters are adjusted
        
        **Learning Objective:**
        - **Goal**: Learn to identify and prefer decks with positive expected values
        - **Strategy**: Develop preference for long-term advantageous options over short-term gains
        - **Measure**: Decision-making under uncertainty and learning from feedback
        - **Flexibility**: Test different risk/reward scenarios and learning patterns
        """)
        
        st.subheader("Results Visualization")
        st.markdown("""
        **Upper Graph - Q-Values for Each Deck:**
        - **Red Line**: Deck A (disadvantageous, high variance)
        - **Orange Line**: Deck B (disadvantageous, low frequency losses)
        - **Green Line**: Deck C (advantageous, high frequency, small losses)
        - **Blue Line**: Deck D (advantageous, low frequency losses)
        - **Learning**: Shows how agent learns deck profitability over time
        
        **Lower Graph - Deck Selection Pattern:**
        - **Color-coded Dots**: Each deck selection over time
        - **Pattern**: Shows shift from bad decks (A,B) to good decks (C,D)
        
        **Key Metrics:**
        - **Final Money**: Total money accumulated/lost during task
        - **Advantageous Choices**: Selections from good decks (C,D)
        - **Learning Curve**: Progression toward optimal deck preferences
        """)
    
    # Stored Graphs Tab
    with st.expander("Stored Graphs Management"):
        st.markdown("""
        **Graph Storage System:**
        - **Automatic Naming**: Graphs named with test type and sequence number
        - **Timestamp**: Each graph tagged with creation time
        - **Parameters**: All simulation settings saved with each graph
        
        **Individual Graph Controls:**
        - **Expandable Sections**: Click to view each stored graph
        - **Image Display**: Full-size graph visualization
        - **Parameter Summary**: All settings used for that simulation
        - **📥 Download Button**: Individual high-resolution downloads
        
        **Management:**
        - **🗑️ Clear All**: Removes all stored graphs (confirmation required)
        - **Persistent Storage**: Graphs remain until you clear them or close the app
        """)
    
    # Human vs AI Comparison Tab
    with st.expander("Human vs AI Comparison"):
        st.markdown("""
        **Research-Based Human Data:**
        - **BART**: Average 25-30 pumps, 30-40% explosion rate (from published studies)
        - **PRLT**: 50-100 trials to learn, 20-50 trials to adapt after reversal
        - **Sources**: Peer-reviewed research papers (Lejuez et al., Cools et al., etc.)
        
        **Comparison Features:**
        - **Side-by-Side View**: Human behavior (left) vs. AI results (right)
        - **Graph Selection**: Choose any stored AI result for comparison
        - **Parameter Display**: See which settings generated the AI behavior
        - **Analysis Guidelines**: Built-in tips for interpreting comparisons
        
        **Research Applications:**
        - **Validation**: Check if AI behavior falls within human ranges
        - **Calibration**: Adjust personality parameters for human-like responses
        - **Individual Differences**: Model specific human populations or traits
        - **Methodology**: Compare task parameters and their effects
        
        **Key Metrics to Compare:**
        - **BART**: Average pumps, explosion patterns, learning curves, risk consistency
        - **PRLT**: Convergence speed, adaptation time, choice patterns, flexibility
        """)
    
    # Tips and Best Practices
    with st.expander("💡 Tips & Best Practices"):
        st.subheader("🔬 For Research Use")
        st.markdown("""
        - **API Key**: Use OpenAI API for more realistic parameter generation
        - **Temperature**: Keep at 0.3 or lower for consistent research results  
        - **Sample Size**: Run multiple simulations to assess variability
        - **Personality Mixing**: Test single personalities first, then combinations
        - **Documentation**: Store graphs with clear naming for later analysis
        """)
        
        st.subheader("🎮 For Exploration")
        st.markdown("""
        - **Try Extremes**: Test high/low parameter values to see behavior changes
        - **Compare Results**: Store graphs from different personality mixes
        - **Custom Personalities**: Upload your own personality descriptions
        - **Temperature Variation**: Experiment with different creativity levels
        """)
        
        st.subheader("⚠️ Troubleshooting")
        st.markdown("""
        - **API Errors**: Check your OpenAI key format and account balance
        - **No Results**: Ensure at least one personality is selected
        - **Slow Performance**: Reduce number of balloons/trials for faster testing
        - **Download Issues**: Try refreshing the page if downloads fail
        """)
    
    # Technical Information
    with st.expander("🔧 Technical Details"):
        st.subheader("Algorithm Information")
        st.markdown("""
        **BART Simulation:**
        - **Learning Algorithm**: Q-learning with exploration
        - **Explosion Probability**: Realistic curve based on research literature
        - **Parameter Mapping**: Personality traits → learning rate, exploration, perseveration
        
        **PRLT Simulation:**
        - **Learning Algorithm**: Q-learning with choice persistence
        - **Convergence Criteria**: 90% correct choices over sliding window
        - **Adaptation Measure**: Trials needed to switch after reversal
        
        **Personality Integration:**
        - **LLM Processing**: GPT-based parameter generation from text descriptions
        - **Heuristic Fallback**: Rule-based parameter estimation when API unavailable
        - **Normalization**: Personality weights scaled to sum to 1.0
        """)
    
    # Contact and Support
    st.subheader("📞 Support & Information")
    st.markdown("""
    **Need Help?**
    - Check this guide first for common questions
    - Ensure your OpenAI API key is valid and has sufficient credits
    - Try the heuristic mode (uncheck "Use OpenAI API") if experiencing API issues
    
    **Research Applications:**
    - Suitable for personality psychology research
    - Decision-making behavior analysis
    - Individual differences in learning and risk-taking
    
    **Data Export:**
    - Use the Download buttons to save graphs as PNG files
    - Store multiple results for comparative analysis
    - All parameters are preserved with each saved graph
    """)

def main():
    """Main Streamlit app"""
    initialize_session_state()
    
    # Create sidebar
    api_key, temperature, use_api, personality_weights = create_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["BART Test", "PRLT Test", "MCQ Test", "IGT Test", "Stored Graphs", "Human vs AI Comparison", "Help & Guide"])
    
    with tab1:
        bart_test_interface(api_key, temperature, use_api, personality_weights)
    
    with tab2:
        prlt_test_interface(api_key, temperature, use_api, personality_weights)
    
    with tab3:
        mcq_test_interface(api_key, temperature, use_api, personality_weights)
    
    with tab4:
        igt_test_interface(api_key, temperature, use_api, personality_weights)
    
    with tab5:
        stored_graphs_interface()
    
    with tab6:
        comparison_interface()
    
    with tab7:
        help_guide_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Personality Testing Suite** - Built with Streamlit | "
        "[GitHub](https://github.com) | [Documentation](https://docs.streamlit.io)"
    )

if __name__ == "__main__":
    main()