"""
offlineTirApp.py – Weighted multi-agent aggregator (offline llama.cpp)

Run fully offline with local GGUF models via llama-cpp-python (LangChain ChatLlamaCpp).
Mirrors onlineModel/onlineTirApp.py features: up to 5 agents, weights, aggregation
template, fatigue awareness, and a modern CustomTkinter GUI.

Run from PSN root:
    python -m offlineModel.offlineTirApp

Models: place .gguf files in PSN/models (or click Browse…). The included install.py
downloads a small LLaMA 2 7B GGUF as a starter. Very large models (e.g., ~120B params)
require enormous RAM/VRAM even when quantized; start with 7B–13B for typical PCs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

from tkinter import filedialog, messagebox
import customtkinter as ctk

# Ensure project root is importable when running this file directly
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from middleware import FatigueController

try:
    # LangChain community chat wrapper for llama.cpp
    from langchain_community.chat_models import ChatLlamaCpp
    from langchain.prompts import ChatPromptTemplate
except Exception as e:  # pragma: no cover
    ChatLlamaCpp = None  # type: ignore
    ChatPromptTemplate = None  # type: ignore

TXT_DIR = _ROOT  # persona .txt files live in repo root
MODELS_DIR = _ROOT / "models"
MAX_TOKENS = 512
MAX_AGENTS = 5


def list_files(folder: Path, ext: str) -> List[Path]:
    try:
        return [p for p in folder.glob(f"*{ext}")]
    except Exception:
        return []


def load_persona(path: Path) -> str:
    return path.read_text(encoding="utf-8")


DEFAULT_AGG_PROMPT = """You are an Aggregation Agent. Read the question and the advice given by up to three advisors,
each with a weight (importance). Treat higher weights as much more influential. Base your decision primarily on
the highest-weighted advice, considering lower-weight advice only if it adds unique or critical info.

Return exactly this format:
chosen_action: <single choice, one line>
rationale: <2-4 sentences explaining how advisor weights shaped your decision>
confidence: <0-1 float, higher if the top-weighted advisor was decisive>

### Question
{question}

### Advisors
{advisor_block}
"""


def _load_agg_prompt_from_file() -> str:
    p = _ROOT / "aggregation_prompt.txt"
    try:
        if p.exists():
            return p.read_text(encoding="utf-8")
    except Exception:
        pass
    return DEFAULT_AGG_PROMPT


def _fill_agg_template(tmpl: str, **kwargs) -> str:
    """Safely fill aggregation template without interpreting other braces.
    Only replaces the placeholders we expect: {question}, {draft}, {advisor_block}.
    Leaves all other braces (like JSON) untouched.
    """
    out = tmpl
    for key in ("question", "draft", "advisor_block"):
        if key in kwargs and kwargs[key] is not None:
            out = out.replace("{" + key + "}", str(kwargs[key]))
    return out


class AgentSelector(ctk.CTkToplevel):
    def __init__(self, master, files, callback):
        super().__init__(master)
        self.title(f"Select agents (max {MAX_AGENTS})")
        self.grab_set()
        self.callback = callback
        self.vars: List[Tuple[ctk.BooleanVar, ctk.CTkEntry, Path]] = []

        ctk.CTkLabel(self, text=f"Tick up to {MAX_AGENTS} agents and set weight (1-100)", font=("", 12, "bold")).pack(pady=(8, 4))
        frame = ctk.CTkFrame(self); frame.pack(padx=12, pady=8, fill="both")
        for p in files:
            row = ctk.CTkFrame(frame); row.pack(fill="x", pady=2)
            var = ctk.BooleanVar()
            chk = ctk.CTkCheckBox(row, text=p.name, variable=var)
            chk.pack(side="left")
            weight = ctk.CTkEntry(row, width=50, justify="center")
            weight.insert(0, "33")
            weight.pack(side="right", padx=4)
            self.vars.append((var, weight, p))
        ctk.CTkButton(self, text="Apply", command=self.apply).pack(pady=(0, 8))

    def apply(self):
        selected = []
        total_w = 0
        for var, entry, path in self.vars:
            if var.get():
                if len(selected) == MAX_AGENTS:
                    messagebox.showinfo("Limit", f"Select max {MAX_AGENTS}")
                    return
                if not entry.get().isdigit():
                    messagebox.showinfo("Weight", "Weights must be integers 1-100")
                    return
                w = int(entry.get())
                if w <= 0 or w > 100:
                    messagebox.showinfo("Weight", "Weights must be 1-100")
                    return
                selected.append((path, w))
                total_w += w
        if not selected:
            messagebox.showinfo("None", "Select at least one agent"); return
        selected = [(p, round(w / total_w, 3)) for p, w in selected]
        self.callback(selected)
        self.destroy()


class OfflineAggregatorGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        self.title("Persona LLaMA – Weighted Multi-Agent (Offline)")
        self.geometry("1100x780")
        self.resizable(False, False)

        self.llm: ChatLlamaCpp | None = None
        self.current_model_path: Path | None = None
        self.temperature = 0.8
        self.energy_cap = 5
        self.fatigue = 0

        self.selected_agents: List[Tuple[Path, float]] = []
        self.agent_boxes: List[ctk.CTkTextbox] = []
        self.fatigue_middleware = FatigueController()

        self.persona_files = list_files(TXT_DIR, ".txt")
        if not self.persona_files:
            self.persona_files = [Path("(Put persona .txt files in PSN root)")]
        self.model_files = list_files(MODELS_DIR, ".gguf")
        if not self.model_files:
            self.model_files = [Path("(Browse…)")]

        self.agg_template = _load_agg_prompt_from_file()

        self._build_ui()

    # -------------- UI -------------- #
    def _build_ui(self):
        top = ctk.CTkFrame(self, corner_radius=10); top.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(top, text="Model").pack(side="left", padx=(10, 0))
        self.model_var = ctk.StringVar(value=str(self.model_files[0]))
        ctk.CTkOptionMenu(top, variable=self.model_var,
                          values=[str(p) for p in self.model_files],
                          command=self._on_model_change).pack(side="left", padx=6)
        ctk.CTkButton(top, text="Browse…", width=90, command=self._browse_model).pack(side="left", padx=(0, 18))

        ctk.CTkLabel(top, text="Temp").pack(side="left")
        self.temp_slider = ctk.CTkSlider(top, from_=0.0, to=1.2, number_of_steps=24, command=self._on_temp_change, width=160)
        self.temp_slider.set(self.temperature); self.temp_slider.pack(side="left", padx=6)
        self.temp_entry = ctk.CTkEntry(top, width=60, justify="center"); self.temp_entry.insert(0, f"{self.temperature:.2f}")
        self.temp_entry.pack(side="left", padx=(0, 8))
        ctk.CTkButton(top, text="Reload model", width=110, command=self._reload_llm).pack(side="left", padx=(0, 8))

        ctk.CTkButton(top, text="Select agents…", command=self._open_agent_selector).pack(side="left")

        ctk.CTkLabel(top, text="Energy").pack(side="left", padx=(20, 0))
        self.energy_slider = ctk.CTkSlider(top, from_=1, to=5, number_of_steps=4, command=self._on_energy_change, width=120)
        self.energy_slider.set(self.energy_cap); self.energy_slider.pack(side="left", padx=6)
        self.energy_entry = ctk.CTkEntry(top, width=50, justify="center"); self.energy_entry.insert(0, str(self.energy_cap))
        self.energy_entry.pack(side="left", padx=(0, 4))
        ctk.CTkButton(top, text="Apply", width=60, command=self._apply_energy).pack(side="left")

        self.agent_frame = ctk.CTkFrame(self, corner_radius=10)
        self.agent_frame.pack(fill="both", expand=True, padx=8, pady=(2, 2))
        self._build_agent_boxes()

        ctk.CTkLabel(self, text="Final Aggregated Answer", font=("", 13, "bold")).pack(pady=(2, 0))
        self.final_box = ctk.CTkTextbox(self, height=240, wrap="word", state="disabled", corner_radius=10, font=("Consolas", 12))
        self.final_box.pack(fill="x", padx=8, pady=(2, 4))

        self.pb = ctk.CTkProgressBar(self, mode="indeterminate"); self.pb.pack(fill="x", padx=12, pady=(0, 6)); self.pb.stop()

        bottom = ctk.CTkFrame(self, corner_radius=10); bottom.pack(fill="x", padx=12, pady=6)
        self.entry = ctk.CTkEntry(bottom); self.entry.pack(side="left", fill="x", expand=True, padx=(0, 6), pady=6)
        ctk.CTkButton(bottom, text="Send", width=90, command=self._on_send).pack(side="left", padx=(0, 6))
        ctk.CTkButton(bottom, text="Clear", width=80, command=self._reset_chat).pack(side="left")

    def _build_agent_boxes(self):
        for w in self.agent_frame.winfo_children(): w.destroy()
        self.agent_boxes = []
        for idx, (path, weight) in enumerate(self.selected_agents or [(None, 0)]):
            col_frame = ctk.CTkFrame(self.agent_frame, corner_radius=10)
            col_frame.grid(row=0, column=idx, sticky="nsew", padx=4, pady=2)
            self.agent_frame.grid_columnconfigure(idx, weight=1)
            name = path.name if path else "Agent"
            ctk.CTkLabel(col_frame, text=f"{name}  ({weight*100:.0f}%)", font=("", 12, "bold")).pack(anchor="w", padx=4, pady=(2, 0))
            box = ctk.CTkTextbox(col_frame, height=240, wrap="word", state="disabled", font=("Consolas", 11))
            box.pack(fill="both", expand=True, padx=2, pady=2)
            self.agent_boxes.append(box)

    # -------------- controls -------------- #
    def _on_model_change(self, *_):
        p = Path(self.model_var.get())
        if p.name == "(Browse…)":
            self._browse_model()
        else:
            self._load_llm(p)

    def _browse_model(self):
        path = filedialog.askopenfilename(title="Select GGUF model", filetypes=[("GGUF", "*.gguf")])
        if path:
            self.model_var.set(path)
            self._load_llm(Path(path))

    def _on_temp_change(self, val):
        try:
            self.temperature = round(float(val), 2)
            self.temp_entry.delete(0, "end"); self.temp_entry.insert(0, f"{self.temperature:.2f}")
        except Exception:
            pass

    def _on_energy_change(self, val):
        self.energy_cap = int(float(val))
        self.energy_entry.delete(0, "end"); self.energy_entry.insert(0, str(self.energy_cap))

    def _apply_energy(self):
        txt = self.energy_entry.get().strip()
        if txt.isdigit() and 1 <= int(txt) <= 5:
            self.energy_cap = int(txt); self.energy_slider.set(self.energy_cap)
        else:
            messagebox.showinfo("Energy", "Enter 1-5")

    def _open_agent_selector(self):
        AgentSelector(self, self.persona_files, self._set_agents)

    def _set_agents(self, sel: List[Tuple[Path, float]]):
        self.selected_agents = sel
        self._build_agent_boxes()

    # -------------- model -------------- #
    def _load_llm(self, path: Path):
        if ChatLlamaCpp is None:
            messagebox.showerror("Missing dependency", "LangChain/llama-cpp not installed. pip install -r requirements.txt")
            return
        if not path.exists():
            messagebox.showerror("Missing model", f"{path}")
            return
        self.current_model_path = path
        self._append_final("System", f"Loading {path.name} … (temp={self.temperature})")
        self.pb.start()
        try:
            # n_ctx: try a reasonable context size for common 7B–13B models
            self.llm = ChatLlamaCpp(
                model_path=str(path),
                n_ctx=4096,
                temperature=self.temperature,
                top_p=0.9,
                max_tokens=MAX_TOKENS,
                stop=[],
            )
            self._append_final("System", "Model loaded. If you changed Temp, 'Reload model' applies it.")
            self._reset_chat()
        except Exception as e:
            self._append_final("Error", str(e)); self.llm = None
        finally:
            self.pb.stop()

    def _reload_llm(self):
        if self.current_model_path is None:
            messagebox.showinfo("No model", "Select a GGUF model first.")
            return
        self._load_llm(self.current_model_path)

    # -------------- chat workflow -------------- #
    def _on_send(self):
        if not self.llm:
            messagebox.showinfo("No model", "Load a GGUF model first.")
            return
        if not self.selected_agents:
            messagebox.showinfo("No agents", f"Select 1-{MAX_AGENTS} agents")
            return
        question = self.entry.get().strip()
        if not question:
            return
        self.entry.delete(0, "end")
        self._clear_boxes()
        self.pb.start()

        import threading
        threading.Thread(target=self._worker, args=(question,), daemon=True).start()

    def _worker(self, question: str):
        if self.fatigue_middleware.is_tired(question):
            self._append_final("Aggregator", "❌ Agent is too tired to answer this prompt again.\nTry rephrasing or switching tasks.")
            self.pb.stop(); return

        advisor_lines = []
        for idx, (path, w) in enumerate(self.selected_agents):
            persona = load_persona(path)
            # If over energy cap, degrade quality by noting fatigue in the prompt
            if self.fatigue >= self.energy_cap:
                persona += "\n\n# Note: Agent is exhausted; keep answer extremely brief."
            try:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", persona),
                    ("user", question),
                ]).format()
                ans = self._to_text(self.llm.invoke(prompt)).replace("\\n", "\n")
            except Exception as e:
                ans = f"Error: {e}"
            self._set_box(idx, ans)
            advisor_lines.append(f"- Advisor {idx+1} ({path.stem}, weight {w*100:.0f}%): \"{ans}\"")

        advisors_text = "\n".join(advisor_lines)
        agg_prompt = _fill_agg_template(
            self.agg_template,
            question=question,
            advisor_block=advisors_text,
            draft=advisors_text,
        )

        # Fatigue-aware aggregator behavior
        if self.fatigue >= self.energy_cap:
            agg_prefix = "You are extremely tired; politely refuse to perform the task and explain you are too tired. If you must respond, keep it to one short line.\n\n"
        elif self.fatigue >= max(1, self.energy_cap - 1):
            agg_prefix = "You are getting tired; keep your answer extremely brief (<= 3 lines) and mention that you are tired.\n\n"
        else:
            agg_prefix = ""
        agg_full = agg_prefix + agg_prompt

        try:
            final = self._to_text(self.llm.invoke(agg_full)).replace("\\n", "\n")
        except Exception as e:
            final = f"Error during aggregation: {e}"

        self._append_final("Aggregator", final)
        self.fatigue += 1
        self.fatigue_middleware.increment(question)
        self.pb.stop()

    # -------------- helpers -------------- #
    @staticmethod
    def _to_text(msg):
        return msg.content if hasattr(msg, "content") else str(msg)

    def _clear_boxes(self):
        for box in self.agent_boxes:
            box.configure(state="normal"); box.delete("1.0", "end"); box.configure(state="disabled")
        self.final_box.configure(state="normal"); self.final_box.delete("1.0", "end"); self.final_box.configure(state="disabled")

    def _set_box(self, idx: int, text: str):
        box = self.agent_boxes[idx]
        box.configure(state="normal"); box.insert("end", text); box.configure(state="disabled")

    def _append_final(self, label: str, text: str):
        self.final_box.configure(state="normal")
        self.final_box.insert("end", f"{label}: {text}\n\n")
        self.final_box.configure(state="disabled"); self.final_box.see("end")

    def _reset_chat(self, *_):
        self._clear_boxes()
        self.fatigue = 0
        self.fatigue_middleware.reset()


def main():
    OfflineAggregatorGUI().mainloop()


if __name__ == "__main__":
    main()
