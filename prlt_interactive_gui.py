"""
Interactive PRLT GUI - Personality Mixing and Task Reward Controls
Features:
- Sliders to adjust task reward probabilities (pA_pre, pB_pre, pA_post, pB_post)
- Selection and weighting of up to 5 personality prompts
- Real-time visualization of agent convergence and switching behavior
- Shows QA/QB curves with marked convergence/switch points
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading
import time
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prlt_personality_proportions import ParamGenerator, AgentParams, PRLTSimulator, PERSONA_FILES

class PRLTInteractiveGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive PRLT - Personality Mixing & Task Control")
        self.root.geometry("1200x800")
        
        # Initialize parameter generator
        self.param_gen = ParamGenerator()
        
        # Variables for task parameters
        self.pA_pre = tk.DoubleVar(value=0.75)
        self.pB_pre = tk.DoubleVar(value=0.25)
        self.pA_post = tk.DoubleVar(value=0.25)
        self.pB_post = tk.DoubleVar(value=0.75)
        
        # Personality weights (up to 5)
        self.personality_vars = {}
        self.personality_weights = {}
        available_personas = list(PERSONA_FILES.keys())
        for i, persona in enumerate(available_personas[:5]):
            self.personality_vars[persona] = tk.BooleanVar(value=(i == 0))  # First one selected by default
            self.personality_weights[persona] = tk.DoubleVar(value=1.0 if i == 0 else 0.0)
        
        # Current simulation state
        self.current_result = None
        self.current_history = None
        self.simulation_running = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frames
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Task Reward Controls
        self.setup_task_controls(control_frame)
        
        # Personality Controls
        self.setup_personality_controls(control_frame)
        
        # Run Controls
        self.setup_run_controls(control_frame)
        
        # Plot Area
        self.setup_plot_area(plot_frame)
        
    def setup_task_controls(self, parent):
        task_frame = ttk.LabelFrame(parent, text="Task Reward Probabilities", padding=10)
        task_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Pre-reversal controls
        ttk.Label(task_frame, text="Pre-Reversal Phase:").grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        ttk.Label(task_frame, text="P(Reward|A):").grid(row=1, column=0, sticky=tk.W)
        pA_pre_scale = ttk.Scale(task_frame, from_=0.1, to=0.9, variable=self.pA_pre, 
                                orient=tk.HORIZONTAL, length=200, command=self.update_pB_pre)
        pA_pre_scale.grid(row=1, column=1, padx=5)
        self.pA_pre_label = ttk.Label(task_frame, text=f"{self.pA_pre.get():.2f}")
        self.pA_pre_label.grid(row=1, column=2)
        
        ttk.Label(task_frame, text="P(Reward|B):").grid(row=2, column=0, sticky=tk.W)
        self.pB_pre_label = ttk.Label(task_frame, text=f"{self.pB_pre.get():.2f}")
        self.pB_pre_label.grid(row=2, column=1, sticky=tk.W, padx=5)
        
        # Post-reversal controls
        ttk.Label(task_frame, text="Post-Reversal Phase:").grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        
        ttk.Label(task_frame, text="P(Reward|A):").grid(row=4, column=0, sticky=tk.W)
        pA_post_scale = ttk.Scale(task_frame, from_=0.1, to=0.9, variable=self.pA_post, 
                                 orient=tk.HORIZONTAL, length=200, command=self.update_pB_post)
        pA_post_scale.grid(row=4, column=1, padx=5)
        self.pA_post_label = ttk.Label(task_frame, text=f"{self.pA_post.get():.2f}")
        self.pA_post_label.grid(row=4, column=2)
        
        ttk.Label(task_frame, text="P(Reward|B):").grid(row=5, column=0, sticky=tk.W)
        self.pB_post_label = ttk.Label(task_frame, text=f"{self.pB_post.get():.2f}")
        self.pB_post_label.grid(row=5, column=1, sticky=tk.W, padx=5)
        
    def setup_personality_controls(self, parent):
        personality_frame = ttk.LabelFrame(parent, text="Personality Mixing (Max 5)", padding=10)
        personality_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(personality_frame, text="Select & Weight Personalities:").pack(anchor=tk.W, pady=(0, 5))
        
        for i, (persona, var) in enumerate(self.personality_vars.items()):
            row_frame = ttk.Frame(personality_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            # Checkbox
            check = ttk.Checkbutton(row_frame, text=persona.replace('_', ' ').title(), 
                                   variable=var, command=self.update_personality_weights)
            check.pack(side=tk.LEFT)
            
            # Weight slider  
            weight_scale = ttk.Scale(row_frame, from_=0.0, to=2.0, 
                                   variable=self.personality_weights[persona],
                                   orient=tk.HORIZONTAL, length=150,
                                   command=lambda v, p=persona: self.update_weight_display(p))
            weight_scale.pack(side=tk.LEFT, padx=(10, 5))
            
            # Weight label
            weight_label = ttk.Label(row_frame, text=f"{self.personality_weights[persona].get():.2f}")
            weight_label.pack(side=tk.LEFT)
            setattr(self, f"{persona}_weight_label", weight_label)
            
        # Control buttons
        button_frame = ttk.Frame(personality_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Normalize Weights", 
                  command=self.normalize_weights).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Clear All", 
                  command=self.clear_all_weights).pack(side=tk.LEFT, padx=(5, 0))
        
        # Current mix display
        self.mix_display = ttk.Label(personality_frame, text="Current mix: None selected", 
                                   font=('TkDefaultFont', 8), foreground='blue')
        self.mix_display.pack(pady=(5, 0))
        self.update_mix_display()
                  
    def setup_run_controls(self, parent):
        run_frame = ttk.LabelFrame(parent, text="Simulation Controls", padding=10)
        run_frame.pack(fill=tk.X, pady=(0, 10))
        
        # API mode selection
        self.use_api = tk.BooleanVar(value=True)
        ttk.Checkbutton(run_frame, text="Use OpenAI API", variable=self.use_api).pack(anchor=tk.W)
        
        # Run button
        self.run_button = ttk.Button(run_frame, text="Run Simulation", command=self.run_simulation)
        self.run_button.pack(pady=10)
        
        # Status
        self.status_label = ttk.Label(run_frame, text="Ready")
        self.status_label.pack()
        
    def setup_plot_area(self, parent):
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6), dpi=100)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(211)  # QA/QB over time
        self.ax2 = self.fig.add_subplot(212)  # Choice sequence
        
        self.fig.tight_layout(pad=3.0)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial empty plots
        self.plot_empty()
        
    def update_pB_pre(self, value):
        pA = float(value)
        pB = 1.0 - pA
        self.pB_pre.set(pB)
        self.pA_pre_label.config(text=f"{pA:.2f}")
        self.pB_pre_label.config(text=f"{pB:.2f}")
        
    def update_pB_post(self, value):
        pA = float(value)
        pB = 1.0 - pA
        self.pB_post.set(pB)
        self.pA_post_label.config(text=f"{pA:.2f}")
        self.pB_post_label.config(text=f"{pB:.2f}")
        
    def update_personality_weights(self):
        # When checkbox changes, update weight display
        for persona in self.personality_vars:
            if not self.personality_vars[persona].get():
                self.personality_weights[persona].set(0.0)
            else:
                # If checkbox is being enabled and weight is 0, set to 1.0
                if self.personality_weights[persona].get() == 0.0:
                    self.personality_weights[persona].set(1.0)
            self.update_weight_display(persona)
                
    def update_weight_display(self, persona):
        weight = self.personality_weights[persona].get()
        label = getattr(self, f"{persona}_weight_label")
        label.config(text=f"{weight:.2f}")
        self.update_mix_display()
        
    def update_mix_display(self):
        """Update the current personality mix display"""
        active_mix = self.get_active_personality_mix()
        if not active_mix:
            self.mix_display.config(text="Current mix: None selected")
        else:
            total = sum(active_mix.values())
            if total == 0:
                self.mix_display.config(text="Current mix: All weights are 0")
            else:
                # Show normalized percentages
                norm_mix = {k: v/total for k, v in active_mix.items()}
                mix_str = ", ".join([f"{k}:{v:.1%}" for k, v in norm_mix.items()])
                self.mix_display.config(text=f"Current mix: {mix_str}")
                
    def clear_all_weights(self):
        """Clear all personality selections and weights"""
        for persona in self.personality_vars:
            self.personality_vars[persona].set(False)
            self.personality_weights[persona].set(0.0)
            self.update_weight_display(persona)
        
    def normalize_weights(self):
        # Get selected personalities and their weights
        active_weights = {}
        for persona in self.personality_vars:
            if self.personality_vars[persona].get():
                weight = self.personality_weights[persona].get()
                if weight > 0:  # Only include non-zero weights
                    active_weights[persona] = weight
                
        if not active_weights:
            messagebox.showwarning("Warning", "Please select at least one personality with weight > 0!")
            return
            
        # Normalize to sum to 1.0
        total = sum(active_weights.values())
        if total > 0:
            for persona in self.personality_vars:
                if persona in active_weights:
                    normalized = active_weights[persona] / total
                    self.personality_weights[persona].set(normalized)
                else:
                    # Unselected personas should have weight 0
                    self.personality_weights[persona].set(0.0)
                self.update_weight_display(persona)
                
            # Show normalization result
            total_after = sum(self.personality_weights[p].get() for p in active_weights)
            self.status_label.config(text=f"Normalized: {len(active_weights)} personas, total weight={total_after:.3f}")
                
    def get_active_personality_mix(self):
        """Get the currently selected personality mix"""
        mix = {}
        for persona in self.personality_vars:
            if self.personality_vars[persona].get():
                weight = self.personality_weights[persona].get()
                if weight > 0:
                    mix[persona] = weight
        return mix
        
    def run_simulation(self):
        if self.simulation_running:
            return
            
        # Get current settings
        personality_mix = self.get_active_personality_mix()
        if not personality_mix:
            messagebox.showerror("Error", "Please select at least one personality with weight > 0!")
            return
            
        self.simulation_running = True
        self.run_button.config(state='disabled')
        self.status_label.config(text="Running simulation...")
        
        # Run in separate thread
        thread = threading.Thread(target=self._run_simulation_thread, args=(personality_mix,))
        thread.daemon = True
        thread.start()
        
    def _run_simulation_thread(self, personality_mix):
        try:
            # Normalize mix
            total = sum(personality_mix.values())
            norm_mix = {k: v/total for k, v in personality_mix.items()}
            
            # Get agent parameters
            if self.use_api.get() and self.param_gen.client:
                params = self.param_gen.get_params_from_llm(norm_mix, temperature=0.3)
            else:
                # Heuristic fallback
                lr = 0.1 + 0.4 * norm_mix.get('risk_taker', 0)
                eps = 0.15 + 0.4 * (1 - norm_mix.get('cautious', 0))
                pers = 0.05 + 0.4 * norm_mix.get('cautious', 0)
                patience = int(10 + 30 * norm_mix.get('cautious', 0))
                params = AgentParams(lr, eps, pers, 0.05, patience, rationale='heuristic')
                
            # Create custom simulator with user-defined reward probabilities
            simulator = CustomPRLTSimulator(
                params, 
                pA_pre=self.pA_pre.get(),
                pB_pre=self.pB_pre.get(),
                pA_post=self.pA_post.get(),
                pB_post=self.pB_post.get(),
                rng_seed=int(time.time()) % 2**32
            )
            
            result, history = simulator.run(pre_reversal_trials=200)
            
            # Store results
            self.current_result = result
            self.current_history = history
            
            # Update UI in main thread
            self.root.after(0, self._update_plot)
            self.root.after(0, lambda: self.status_label.config(text="Simulation complete"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Simulation failed: {e}"))
            self.root.after(0, lambda: self.status_label.config(text="Error"))
        finally:
            self.root.after(0, lambda: setattr(self, 'simulation_running', False))
            self.root.after(0, lambda: self.run_button.config(state='normal'))
            
    def _update_plot(self):
        if not self.current_history:
            return
            
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Extract data
        trials = [t['trial'] for t in self.current_history]
        QA_vals = [t['QA'] for t in self.current_history]
        QB_vals = [t['QB'] for t in self.current_history]
        choices = [1 if t['choice'] == 'A' else 0 for t in self.current_history]
        phases = [t['phase'] for t in self.current_history]
        
        # Find reversal point
        reversal_trial = None
        for i, phase in enumerate(phases):
            if phase == 'post':
                reversal_trial = trials[i]
                break
                
        # Plot Q-values
        self.ax1.plot(trials, QA_vals, 'b-', label='Q(A)', linewidth=2)
        self.ax1.plot(trials, QB_vals, 'r-', label='Q(B)', linewidth=2)
        
        # Mark reversal point
        if reversal_trial:
            self.ax1.axvline(reversal_trial, color='orange', linestyle='--', alpha=0.7, label='Reversal')
            
        # Mark convergence points
        if self.current_result.pre_rev_trials_to_converge:
            self.ax1.axvline(self.current_result.pre_rev_trials_to_converge, 
                           color='green', linestyle=':', alpha=0.7, label='Pre-Converge')
            
        if self.current_result.post_rev_trials_to_switch and reversal_trial:
            switch_trial = reversal_trial + self.current_result.post_rev_trials_to_switch
            if switch_trial <= max(trials):
                self.ax1.axvline(switch_trial, color='purple', linestyle=':', alpha=0.7, label='Post-Switch')
                
        self.ax1.set_xlabel('Trial')
        self.ax1.set_ylabel('Q-Value')
        self.ax1.set_title('Agent Q-Values Over Time')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)
        
        # Plot choice sequence
        choice_trials = []
        choice_A = []
        choice_B = []
        
        for i, choice in enumerate(choices):
            if choice == 1:  # A
                choice_A.append(trials[i])
            else:  # B
                choice_B.append(trials[i])
                
        if choice_A:
            self.ax2.scatter(choice_A, [1]*len(choice_A), c='blue', alpha=0.6, s=20, label='Choice A')
        if choice_B:
            self.ax2.scatter(choice_B, [0]*len(choice_B), c='red', alpha=0.6, s=20, label='Choice B')
            
        # Mark phases and key points
        if reversal_trial:
            self.ax2.axvline(reversal_trial, color='orange', linestyle='--', alpha=0.7)
            
        if self.current_result.pre_rev_trials_to_converge:
            self.ax2.axvline(self.current_result.pre_rev_trials_to_converge, 
                           color='green', linestyle=':', alpha=0.7)
            
        if self.current_result.post_rev_trials_to_switch and reversal_trial:
            switch_trial = reversal_trial + self.current_result.post_rev_trials_to_switch
            if switch_trial <= max(trials):
                self.ax2.axvline(switch_trial, color='purple', linestyle=':', alpha=0.7)
                
        self.ax2.set_xlabel('Trial')
        self.ax2.set_ylabel('Choice')
        self.ax2.set_yticks([0, 1])
        self.ax2.set_yticklabels(['B', 'A'])
        self.ax2.set_title('Choice Sequence')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)
        
        # Add result text
        result_text = f"Pre-converge: {self.current_result.pre_rev_trials_to_converge} trials\n"
        result_text += f"Post-switch: {self.current_result.post_rev_trials_to_switch} trials"
        self.ax2.text(0.02, 0.98, result_text, transform=self.ax2.transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def plot_empty(self):
        self.ax1.text(0.5, 0.5, 'Run simulation to see Q-value curves', 
                     transform=self.ax1.transAxes, ha='center', va='center', fontsize=12)
        self.ax1.set_title('Agent Q-Values Over Time')
        
        self.ax2.text(0.5, 0.5, 'Run simulation to see choice sequence', 
                     transform=self.ax2.transAxes, ha='center', va='center', fontsize=12)
        self.ax2.set_title('Choice Sequence')
        
        self.fig.tight_layout()
        self.canvas.draw()


class CustomPRLTSimulator(PRLTSimulator):
    """Custom simulator with user-configurable reward probabilities"""
    
    def __init__(self, params, pA_pre=0.75, pB_pre=0.25, pA_post=0.25, pB_post=0.75, rng_seed=None):
        super().__init__(params, rng_seed)
        self.pA_pre = pA_pre
        self.pB_pre = pB_pre  
        self.pA_post = pA_post
        self.pB_post = pB_post
        
    def run(self, pre_reversal_trials=200, reversal_after_stable=True):
        # Use custom reward probabilities instead of fixed ones
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
        max_pre = pre_reversal_trials
        correct_pre = 'A' if pA > pB else 'B'
        
        for t in range(1, max_pre + 1):
            choice = self.choose(QA, QB, prev_choice)
            reward = 1 if self.rng.random() < (pA if choice == 'A' else pB) else 0
            QA, QB = self.update(QA, QB, choice, reward)
            prev_choice = choice
            trial_history.append({'phase':'pre', 'trial': t, 'choice': choice, 'reward': reward, 'QA': QA, 'QB': QB})
            
            if check_convergence(trial_history, correct_pre, window=self.patience, threshold=0.9):
                pre_converge_trial = t
                break

        if pre_converge_trial is None:
            pre_converge_trial = max_pre

        # Post-reversal phase
        pA = self.pA_post
        pB = self.pB_post
        correct_post = 'A' if pA > pB else 'B'

        post_converge_trial = None
        max_post = 1000
        for t2 in range(1, max_post + 1):
            t_index = pre_converge_trial + t2
            choice = self.choose(QA, QB, prev_choice)
            reward = 1 if self.rng.random() < (pA if choice == 'A' else pB) else 0
            QA, QB = self.update(QA, QB, choice, reward)
            prev_choice = choice
            trial_history.append({'phase':'post', 'trial': t_index, 'choice': choice, 'reward': reward, 'QA': QA, 'QB': QB})
            
            if check_convergence(trial_history, correct_post, window=self.patience, threshold=0.9):
                post_converge_trial = t2
                break

        if post_converge_trial is None:
            post_converge_trial = max_post

        from prlt_personality_proportions import PRLTResult
        from dataclasses import asdict
        
        result = PRLTResult(
            persona_mix_name='interactive',
            params=asdict(self.params),
            pre_rev_trials_to_converge=pre_converge_trial,
            post_rev_trials_to_switch=post_converge_trial,
            total_trials_run=len(trial_history),
            trial_history=trial_history
        )

        return result, trial_history


def main():
    root = tk.Tk()
    app = PRLTInteractiveGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()