"""
Interactive BART GUI - Personality Mixing and Task Controls
Features:
- Sliders to adjust task parameters (number of balloons, max pumps, explosion curve)
- Selection and weighting of up to 5 personality prompts (same UI behavior as PRLT GUI)
- Real-time visualization of pumps per balloon and agent Q-values over time
- Shows pumps, explosions and cash-outs with marked learning / adaptation points
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

from prlt_personality_proportions import ParamGenerator, AgentParams, PERSONA_FILES


class BARTInteractiveGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive BART - Personality Mixing & Task Control")
        self.root.geometry("1200x800")

        # Initialize parameter generator
        self.param_gen = ParamGenerator()

        # Task parameters
        self.num_balloons = tk.IntVar(value=30)
        self.max_pumps = tk.IntVar(value=64)
        # explosion_curve controls how quickly explosion probability rises (0.1..2.0)
        self.explosion_curve = tk.DoubleVar(value=1.0)

        # Personality weights (up to 5)
        self.personality_vars = {}
        self.personality_weights = {}
        available_personas = list(PERSONA_FILES.keys())
        for i, persona in enumerate(available_personas[:5]):
            self.personality_vars[persona] = tk.BooleanVar(value=(i == 0))
            self.personality_weights[persona] = tk.DoubleVar(value=1.0 if i == 0 else 0.0)

        self.current_result = None
        self.current_history = None
        self.simulation_running = False

        self.setup_ui()

    def setup_ui(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.setup_task_controls(control_frame)
        self.setup_personality_controls(control_frame)
        self.setup_run_controls(control_frame)
        self.setup_plot_area(plot_frame)

    def setup_task_controls(self, parent):
        task_frame = ttk.LabelFrame(parent, text="Task Parameters", padding=10)
        task_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(task_frame, text="Number of Balloons:").grid(row=0, column=0, sticky=tk.W)
        num_scale = ttk.Scale(task_frame, from_=5, to=100, variable=self.num_balloons,
                              orient=tk.HORIZONTAL, length=200, command=lambda v: self._int_var_update(self.num_balloons, self.num_balloons_label))
        num_scale.grid(row=0, column=1, padx=5)
        self.num_balloons_label = ttk.Label(task_frame, text=f"{self.num_balloons.get()}")
        self.num_balloons_label.grid(row=0, column=2)

        ttk.Label(task_frame, text="Max Pumps per Balloon:").grid(row=1, column=0, sticky=tk.W)
        max_scale = ttk.Scale(task_frame, from_=8, to=128, variable=self.max_pumps,
                              orient=tk.HORIZONTAL, length=200, command=lambda v: self._int_var_update(self.max_pumps, self.max_pumps_label))
        max_scale.grid(row=1, column=1, padx=5)
        self.max_pumps_label = ttk.Label(task_frame, text=f"{self.max_pumps.get()}")
        self.max_pumps_label.grid(row=1, column=2)

        ttk.Label(task_frame, text="Explosion Curve:").grid(row=2, column=0, sticky=tk.W)
        curve_scale = ttk.Scale(task_frame, from_=0.2, to=2.0, variable=self.explosion_curve,
                                orient=tk.HORIZONTAL, length=200, command=lambda v: self._float_var_update(self.explosion_curve, self.explosion_curve_label))
        curve_scale.grid(row=2, column=1, padx=5)
        self.explosion_curve_label = ttk.Label(task_frame, text=f"{self.explosion_curve.get():.2f}")
        self.explosion_curve_label.grid(row=2, column=2)

    def _int_var_update(self, var, label):
        label.config(text=f"{int(var.get())}")

    def _float_var_update(self, var, label):
        label.config(text=f"{var.get():.2f}")

    def setup_personality_controls(self, parent):
        personality_frame = ttk.LabelFrame(parent, text="Personality Mixing (Max 5)", padding=10)
        personality_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(personality_frame, text="Select & Weight Personalities:").pack(anchor=tk.W, pady=(0, 5))

        for i, (persona, var) in enumerate(self.personality_vars.items()):
            row_frame = ttk.Frame(personality_frame)
            row_frame.pack(fill=tk.X, pady=2)

            check = ttk.Checkbutton(row_frame, text=persona.replace('_', ' ').title(),
                                    variable=var, command=self.update_personality_weights)
            check.pack(side=tk.LEFT)

            weight_scale = ttk.Scale(row_frame, from_=0.0, to=2.0,
                                     variable=self.personality_weights[persona],
                                     orient=tk.HORIZONTAL, length=150,
                                     command=lambda v, p=persona: self.update_weight_display(p))
            weight_scale.pack(side=tk.LEFT, padx=(10, 5))

            weight_label = ttk.Label(row_frame, text=f"{self.personality_weights[persona].get():.2f}")
            weight_label.pack(side=tk.LEFT)
            setattr(self, f"{persona}_weight_label", weight_label)

        button_frame = ttk.Frame(personality_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(button_frame, text="Normalize Weights", command=self.normalize_weights).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Clear All", command=self.clear_all_weights).pack(side=tk.LEFT, padx=(5, 0))

        self.mix_display = ttk.Label(personality_frame, text="Current mix: None selected",
                                     font=('TkDefaultFont', 8), foreground='blue')
        self.mix_display.pack(pady=(5, 0))
        self.update_mix_display()

    def setup_run_controls(self, parent):
        run_frame = ttk.LabelFrame(parent, text="Simulation Controls", padding=10)
        run_frame.pack(fill=tk.X, pady=(0, 10))

        self.use_api = tk.BooleanVar(value=True)
        ttk.Checkbutton(run_frame, text="Use OpenAI API", variable=self.use_api).pack(anchor=tk.W)

        self.run_button = ttk.Button(run_frame, text="Run Simulation", command=self.run_simulation)
        self.run_button.pack(pady=10)

        self.status_label = ttk.Label(run_frame, text="Ready")
        self.status_label.pack()

    def setup_plot_area(self, parent):
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(211)  # Q-values / estimates over balloons
        self.ax2 = self.fig.add_subplot(212)  # Pumps per balloon sequence

        self.fig.tight_layout(pad=3.0)

        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.plot_empty()

    # Personality weight helpers (same logic as PRLT GUI)
    def update_personality_weights(self):
        for persona in self.personality_vars:
            if not self.personality_vars[persona].get():
                self.personality_weights[persona].set(0.0)
            else:
                if self.personality_weights[persona].get() == 0.0:
                    self.personality_weights[persona].set(1.0)
            self.update_weight_display(persona)

    def update_weight_display(self, persona):
        weight = self.personality_weights[persona].get()
        label = getattr(self, f"{persona}_weight_label")
        label.config(text=f"{weight:.2f}")
        self.update_mix_display()

    def update_mix_display(self):
        active_mix = self.get_active_personality_mix()
        if not active_mix:
            self.mix_display.config(text="Current mix: None selected")
        else:
            total = sum(active_mix.values())
            if total == 0:
                self.mix_display.config(text="Current mix: All weights are 0")
            else:
                norm_mix = {k: v/total for k, v in active_mix.items()}
                mix_str = ", ".join([f"{k}:{v:.1%}" for k, v in norm_mix.items()])
                self.mix_display.config(text=f"Current mix: {mix_str}")

    def clear_all_weights(self):
        for persona in self.personality_vars:
            self.personality_vars[persona].set(False)
            self.personality_weights[persona].set(0.0)
            self.update_weight_display(persona)

    def normalize_weights(self):
        active_weights = {}
        for persona in self.personality_vars:
            if self.personality_vars[persona].get():
                weight = self.personality_weights[persona].get()
                if weight > 0:
                    active_weights[persona] = weight

        if not active_weights:
            messagebox.showwarning("Warning", "Please select at least one personality with weight > 0!")
            return

        total = sum(active_weights.values())
        if total > 0:
            for persona in self.personality_vars:
                if persona in active_weights:
                    normalized = active_weights[persona] / total
                    self.personality_weights[persona].set(normalized)
                else:
                    self.personality_weights[persona].set(0.0)
                self.update_weight_display(persona)

            total_after = sum(self.personality_weights[p].get() for p in active_weights)
            self.status_label.config(text=f"Normalized: {len(active_weights)} personas, total weight={total_after:.3f}")

    def get_active_personality_mix(self):
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

        personality_mix = self.get_active_personality_mix()
        if not personality_mix:
            messagebox.showerror("Error", "Please select at least one personality with weight > 0!")
            return

        self.simulation_running = True
        self.run_button.config(state='disabled')
        self.status_label.config(text="Running simulation...")

        thread = threading.Thread(target=self._run_simulation_thread, args=(personality_mix,))
        thread.daemon = True
        thread.start()

    def _run_simulation_thread(self, personality_mix):
        try:
            total = sum(personality_mix.values())
            norm_mix = {k: v/total for k, v in personality_mix.items()}

            if self.use_api.get() and self.param_gen.client:
                params = self.param_gen.get_params_from_llm(norm_mix, temperature=0.3)
            else:
                # Heuristic mapping: risk_taker -> higher pump tendency (lower epsilon), cautious -> lower
                lr = 0.2 + 0.3 * norm_mix.get('risk_taker', 0)
                eps = 0.2 * (1 - norm_mix.get('risk_taker', 0))
                pers = 0.05 + 0.3 * norm_mix.get('cautious', 0)
                patience = int(8 + 20 * norm_mix.get('cautious', 0))
                params = AgentParams(lr, eps, pers, 0.05, patience, rationale='heuristic')

            simulator = CustomBARTSimulator(
                params,
                num_balloons=int(self.num_balloons.get()),
                max_pumps=int(self.max_pumps.get()),
                curve=self.explosion_curve.get(),
                rng_seed=int(time.time()) % 2**32
            )

            result, history = simulator.run()

            self.current_result = result
            self.current_history = history

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

        self.ax1.clear()
        self.ax2.clear()

        balloons = [b['balloon'] for b in self.current_history]
        pumps = [b['pumps'] for b in self.current_history]
        exploded = [b['exploded'] for b in self.current_history]
        cashed = [not e for e in exploded]

        # Q-values recorded per balloon
        Q_pump = [b['Q_pump'] for b in self.current_history]
        Q_cash = [b['Q_cash'] for b in self.current_history]

        self.ax1.plot(balloons, Q_pump, 'b-', label='Q(Pump)', linewidth=2)
        self.ax1.plot(balloons, Q_cash, 'g-', label='Q(Cash)', linewidth=2)
        self.ax1.set_xlabel('Balloon')
        self.ax1.set_ylabel('Estimated Value')
        self.ax1.set_title('Agent Value Estimates Across Balloons')
        self.ax1.legend()
        self.ax1.grid(True, alpha=0.3)

        # Pumps per balloon scatter
        exploded_x = [balloons[i] for i, e in enumerate(exploded) if e]
        exploded_y = [pumps[i] for i, e in enumerate(exploded) if e]
        cashed_x = [balloons[i] for i, e in enumerate(exploded) if not e]
        cashed_y = [pumps[i] for i, e in enumerate(exploded) if not e]

        if cashed_x:
            self.ax2.scatter(cashed_x, cashed_y, c='green', label='Cashed', alpha=0.7)
        if exploded_x:
            self.ax2.scatter(exploded_x, exploded_y, c='red', label='Exploded', alpha=0.7)

        self.ax2.set_xlabel('Balloon')
        self.ax2.set_ylabel('Pumps')
        self.ax2.set_title('Pumps per Balloon (Green=cash, Red=exploded)')
        self.ax2.legend()
        self.ax2.grid(True, alpha=0.3)

        # Add summary text
        avg_pumps = np.mean(pumps) if pumps else 0
        exploded_count = sum(1 for e in exploded if e)
        summary = f"Avg pumps: {avg_pumps:.2f}\nExploded: {exploded_count}/{len(balloons)}"
        self.ax2.text(0.02, 0.98, summary, transform=self.ax2.transAxes, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        self.fig.tight_layout()
        self.canvas.draw()

    def plot_empty(self):
        self.ax1.text(0.5, 0.5, 'Run simulation to see value estimates', transform=self.ax1.transAxes, ha='center', va='center', fontsize=12)
        self.ax1.set_title('Agent Value Estimates Across Balloons')

        self.ax2.text(0.5, 0.5, 'Run simulation to see pumps per balloon', transform=self.ax2.transAxes, ha='center', va='center', fontsize=12)
        self.ax2.set_title('Pumps per Balloon')

        self.fig.tight_layout()
        self.canvas.draw()


class CustomBARTSimulator:
    """Simple BART simulator using AgentParams for behavioral mapping.

    Model:
    - Agent maintains Q_pump (expected gain from pumping once more) and Q_cash (expected gain from cashing now).
    - On each balloon, agent repeatedly chooses to pump or cash using epsilon-greedy on Q_pump vs Q_cash with perseveration bias.
    - Pumping increases explosion hazard; if explosion -> 0 reward for that balloon, else if cash -> reward = pumps.
    """

    def __init__(self, params: AgentParams, num_balloons=30, max_pumps=64, curve=1.0, rng_seed=None):
        self.params = params
        self.num_balloons = int(num_balloons)
        self.max_pumps = int(max_pumps)
        self.curve = float(curve)
        self.rng = np.random.RandomState(rng_seed)

    def explosion_probability(self, pump_count):
        # Linearized hazard controlled by curve: p = (pump_count / max_pumps) ** curve
        return min(0.999, (pump_count / float(self.max_pumps)) ** self.curve)

    def choose(self, Q_pump, Q_cash, prev_action):
        # Incorporate perseveration bias: small bonus for repeating previous action
        pump_val = Q_pump + (self.params.perseveration if prev_action == 'pump' else 0.0)
        cash_val = Q_cash + (self.params.perseveration if prev_action == 'cash' else 0.0)

        if self.rng.random() < self.params.epsilon:
            return 'pump' if self.rng.random() < 0.5 else 'cash'
        return 'pump' if pump_val >= cash_val else 'cash'

    def update(self, Q, reward):
        return Q + self.params.learning_rate * (reward - Q)

    def run(self):
        history = []
        # Initialize estimates
        Q_pump = 1.0  # expected immediate gain for an extra pump
        Q_cash = 0.0  # expected gain for cashing now
        prev_action = None

        for b in range(1, self.num_balloons + 1):
            pumps = 0
            exploded = False

            while True:
                action = self.choose(Q_pump, Q_cash, prev_action)
                prev_action = 'pump' if action == 'pump' else 'cash'

                if action == 'pump':
                    pumps += 1
                    # check explosion
                    if self.rng.random() < self.explosion_probability(pumps):
                        exploded = True
                        reward = 0
                        # update Q_pump based on 0 reward
                        Q_pump = self.update(Q_pump, reward)
                        break
                    else:
                        # no explosion yet; agent may update Q_pump with small positive outcome expectation
                        # treat a successful pump as incremental potential value (not actual cash)
                        # keep Q_pump unchanged or slightly updated toward expected pump value
                        Q_pump = self.update(Q_pump, 1.0 * (1 - self.explosion_probability(pumps)))
                        # continue deciding
                        if pumps >= self.max_pumps:
                            # force cash if reached max
                            action = 'cash'

                if action == 'cash':
                    reward = pumps
                    Q_cash = self.update(Q_cash, reward)
                    break

            history.append({'balloon': b, 'pumps': pumps, 'exploded': exploded, 'Q_pump': Q_pump, 'Q_cash': Q_cash})

        # Build result structure similar to PRLTResult minimal fields
        from dataclasses import asdict, dataclass

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
            persona_mix_name='interactive',
            params=asdict(self.params) if hasattr(self.params, '__dict__') else {},
            total_balloons=self.num_balloons,
            avg_pumps=avg_pumps,
            exploded_count=exploded_count,
            trial_history=history
        )

        return result, history


def main():
    root = tk.Tk()
    app = BARTInteractiveGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
