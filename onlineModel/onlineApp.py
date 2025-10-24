"""
onlineApp.py – Persona GUI using OpenAI's GPT models (online)

What this is
- A drop-in GUI similar to app.py but it talks to OpenAI instead of a local GGUF model.
- Picks a persona from your existing .txt files (in the PSN root) and chats with it.
- Tracks simple "energy/fatigue" like the local version.

API key setup (Windows PowerShell)
- Option A (temporary for this session):
	$env:OPENAI_API_KEY = "sk-..."
- Option B (recommended): create a .env file at the PSN project root with a line:
	OPENAI_API_KEY=sk-...
  We'll auto-load it if python-dotenv is installed (added to requirements.txt).

Dependencies
- Added to PSN/requirements.txt: openai, python-dotenv

Run
- From the PSN root (so persona .txt files are found):
	python -m onlineModel.onlineApp
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import List, Tuple

from tkinter import messagebox
import customtkinter as ctk

try:
	# Optional: load .env from the PSN root
	from dotenv import load_dotenv
	_ROOT = Path(__file__).resolve().parents[1]
	load_dotenv(_ROOT / ".env")
except Exception:
	pass

try:
	# OpenAI SDK v1.x
	from openai import OpenAI
except Exception as e:
	OpenAI = None  # we'll validate at runtime


# Directories: personas live in the PSN root next to the .txt files
TXT_DIR = Path(__file__).resolve().parents[1]
MAX_TOKENS = 512

# Aggregation support (final decision block)
DEFAULT_AGG_PROMPT = (
	"You are an Aggregation Agent. Given the question and the assistant's draft answer, "
	"return a final, decisive response in this exact format:\n"
	"chosen_action: <one-line decisive action>\n"
	"rationale: <2-4 sentences justifying the action>\n"
	"confidence: <0-1 float summarizing confidence>\n\n"
	"### Question\n{question}\n\n### Draft Answer\n{draft}\n"
)

def _load_agg_prompt_from_file() -> str:
	p = TXT_DIR / "aggregation_prompt.txt"
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


def list_files(folder: Path, ext: str) -> List[Path]:
	return sorted([p for p in folder.glob(f"*{ext}") if p.is_file()])


MAX_AGENTS = 5


class AgentSelector(ctk.CTkToplevel):
	def __init__(self, master, files: List[Path], callback):
		super().__init__(master)
		self.title(f"Select agents (max {MAX_AGENTS})")
		self.grab_set()
		self.callback = callback
		self.vars = []
		ctk.CTkLabel(self, text=f"Tick up to {MAX_AGENTS} agents and set weight (1-100)", font=("",12,"bold")).pack(pady=(8,4))
		frame = ctk.CTkFrame(self); frame.pack(padx=12, pady=8, fill="both")
		for p in files:
			row = ctk.CTkFrame(frame); row.pack(fill="x", pady=2)
			var = ctk.BooleanVar()
			chk = ctk.CTkCheckBox(row, text=p.name, variable=var); chk.pack(side="left")
			weight = ctk.CTkEntry(row, width=50, justify="center"); weight.insert(0, "20")
			weight.pack(side="right", padx=4)
			self.vars.append((var, weight, p))
		ctk.CTkButton(self, text="Apply", command=self.apply).pack(pady=(0,8))

	def apply(self):
		selected = []; total_w = 0
		for var, entry, path in self.vars:
			if var.get():
				if len(selected) == MAX_AGENTS:
					messagebox.showinfo("Limit", f"Select max {MAX_AGENTS}"); return
				if not entry.get().isdigit():
					messagebox.showinfo("Weight", "Weights must be integers 1-100"); return
				w = int(entry.get()); if_invalid = (w<=0 or w>100)
				if if_invalid:
					messagebox.showinfo("Weight", "Weights must be 1-100"); return
				selected.append((path, w)); total_w += w
		if not selected:
			messagebox.showinfo("None", "Select at least one agent"); return
		selected = [(p, round(w/total_w,3)) for p,w in selected]
		self.callback(selected); self.destroy()


class OnlineChatGUI(ctk.CTk):
	def __init__(self):
		super().__init__()

		ctk.set_appearance_mode("light")
		ctk.set_default_color_theme("blue")
		self.title("Persona GPT – Online")
		self.geometry("980x740")
		self.resizable(False, False)

		# runtime state
		self.client = None
		self.model_name = "gpt-4o-mini"  # default fast+smart
		self.temperature = 0.8
		self.fatigue = 0
		self.energy_cap = 5
		self.messages = []  # (role, content)
		# multi-agent state
		self.selected_agents = []  # list[(Path, weight_float)]
		self.agent_boxes = []

		# files
		self.persona_files = list_files(TXT_DIR, ".txt")
		if not self.persona_files:
			# still show a placeholder to avoid crashing the UI
			self.persona_files = [TXT_DIR / "(Put persona .txt files in PSN root)"]

		# models list (curated) – limited to Chat Completions-compatible
		self.available_models = [
			"gpt-4o",          # high quality, multimodal
			"gpt-4o-mini",     # cheaper/faster, good quality
		]

		self._build_ui()
		self._init_openai()
		self.agg_template = _load_agg_prompt_from_file()

	# -------------- setup -------------- #
	def _init_openai(self):
		if OpenAI is None:
			messagebox.showerror(
				"Missing dependency",
				"The OpenAI SDK is not installed. Please pip install -r requirements.txt",
			)
			return
		api_key = os.getenv("OPENAI_API_KEY", "").strip()
		if not api_key:
			messagebox.showwarning(
				"OPENAI_API_KEY not set",
				"Set it in your environment or in a .env file at the PSN root.\n"
				"Example (PowerShell): $env:OPENAI_API_KEY = 'sk-...'")
		try:
			self.client = OpenAI()  # will read API key from env
		except Exception as e:
			messagebox.showerror("OpenAI init failed", str(e))
			self.client = None

	# -------------- UI ----------------- #
	def _build_ui(self):
		self.configure(fg_color="#f0f0f0")

		# top bar
		top = ctk.CTkFrame(self, corner_radius=10)
		top.pack(fill="x", padx=14, pady=8)

		# model picker
		ctk.CTkLabel(top, text="Model").pack(side="left", padx=(10, 0))
		default_model_value = self.model_name if self.model_name in self.available_models else self.available_models[0]
		self.model_var = ctk.StringVar(value=default_model_value)
		self.model_var.trace_add("write", lambda *_: self._on_model_change())
		ctk.CTkOptionMenu(
			top, variable=self.model_var, values=self.available_models
		).pack(side="left", padx=6)

		# temperature
		ctk.CTkLabel(top, text="Temp").pack(side="left", padx=(24, 0))
		self.temp_slider = ctk.CTkSlider(
			top, from_=0.0, to=1.2, number_of_steps=24,
			command=self._on_temp_change, width=160
		)
		self.temp_slider.set(self.temperature)
		self.temp_slider.pack(side="left", padx=6)
		self.temp_entry = ctk.CTkEntry(top, width=60, justify="center")
		self.temp_entry.insert(0, f"{self.temperature:.2f}")
		self.temp_entry.pack(side="left", padx=(0, 8))

		# persona picker (default persona used when no multi-agent selection)
		ctk.CTkLabel(top, text="Persona").pack(side="left")
		self.persona_var = ctk.StringVar(value=str(self.persona_files[0]))
		ctk.CTkOptionMenu(top, variable=self.persona_var, values=[str(p) for p in self.persona_files], command=lambda *_: self._reset_chat()).pack(side="left", padx=6)

		# agent selection button (optional multi-agent mode)
		ctk.CTkButton(top, text="Select agents…", command=self._open_agent_selector).pack(side="left", padx=(8,0))

		# energy controls
		ctk.CTkLabel(top, text="Energy").pack(side="left", padx=(24, 0))
		self.energy_slider = ctk.CTkSlider(
			top, from_=1, to=20, number_of_steps=19,
			command=self._on_energy_change, width=140
		)
		self.energy_slider.set(self.energy_cap)
		self.energy_slider.pack(side="left", padx=6)
		self.energy_entry = ctk.CTkEntry(top, width=60, justify="center")
		self.energy_entry.insert(0, str(self.energy_cap))
		self.energy_entry.pack(side="left", padx=(0, 4))
		ctk.CTkButton(top, text="Apply", width=60, command=self._apply_energy).pack(side="left")

		# chat area + (optional) per-agent boxes shown only if agents selected
		self.chat_box = ctk.CTkTextbox(self, wrap="word", state="disabled", corner_radius=10, font=("Consolas", 12))
		self.chat_box.pack(fill="both", expand=True, padx=14, pady=(8, 4))

		# progress
		self.pb = ctk.CTkProgressBar(self, mode="indeterminate")
		self.pb.pack(fill="x", padx=14, pady=(0, 8))
		self.pb.stop()

		# bottom input
		bottom = ctk.CTkFrame(self, corner_radius=10)
		bottom.pack(fill="x", padx=14, pady=6)
		self.entry = ctk.CTkEntry(bottom)
		self.entry.pack(side="left", fill="x", expand=True, padx=(0, 6), pady=6)
		ctk.CTkButton(bottom, text="Send", width=90, command=self._on_send).pack(side="left", padx=(0, 6))
		ctk.CTkButton(bottom, text="Clear", width=80, command=self._reset_chat).pack(side="left")

	def _open_agent_selector(self):
		AgentSelector(self, self.persona_files, self._set_agents)

	def _set_agents(self, sel: List[Tuple[Path, float]]):
		self.selected_agents = sel
		# Clear chat to indicate new selection context
		self._reset_chat()

	# -------------- UI events ---------- #
	def _on_model_change(self):
		self.model_name = self.model_var.get()

	def _on_temp_change(self, val):
		try:
			self.temperature = round(float(val), 2)
			self.temp_entry.delete(0, "end")
			self.temp_entry.insert(0, f"{self.temperature:.2f}")
		except Exception:
			pass

	def _on_energy_change(self, val):
		self.energy_cap = int(float(val))
		self.energy_entry.delete(0, "end")
		self.energy_entry.insert(0, str(self.energy_cap))

	def _apply_energy(self):
		txt = self.energy_entry.get().strip()
		if txt.isdigit() and 1 <= int(txt) <= 20:
			self.energy_cap = int(txt)
			self.energy_slider.set(self.energy_cap)
		else:
			messagebox.showinfo("Energy must be 1-20", "Please enter an integer between 1 and 20.")
			self.energy_entry.delete(0, "end")
			self.energy_entry.insert(0, str(self.energy_cap))

	# -------------- chat workflow ------- #
	def _on_send(self):
		if self.client is None:
			self._append("System", "OpenAI client not initialized. Check your dependencies/API key.")
			return

		msg = self.entry.get().strip()
		if not msg:
			return
		self.entry.delete(0, "end")
		self._append("You", msg)
		self.messages.append(("user", msg))

		# Multi-agent path if agents are selected; else single agent with selected persona
		if self.selected_agents:
			threading.Thread(target=self._multi_agent_thread, args=(msg,), daemon=True).start()
		else:
			persona_text = self._read_persona()
			if self.fatigue >= self.energy_cap:
				persona_text += "\n\nFATIGUE_COEF = 2.0  # agent is exhausted"
			messages = [{"role": "system", "content": persona_text}] + [{"role": r, "content": c} for r,c in self.messages]
			threading.Thread(target=self._openai_thread, args=(messages,), daemon=True).start()

	def _openai_thread(self, messages: List[dict]):
		self.pb.start()
		try:
			# Prefer Chat Completions for compatibility
			resp = self.client.chat.completions.create(
				model=self.model_name,
				messages=messages,
				temperature=self.temperature,
				max_tokens=MAX_TOKENS,
			)
			text = (resp.choices[0].message.content or "").replace("\\n", "\n")
			if not text:
				text = "<empty response>"
			self.messages.append(("assistant", text))
			self._append("Agent", text)

			# Final aggregation step: convert draft into decisive final block
			try:
				question = next((c for r,c in reversed(self.messages) if r=="user"), "")
				# Build a synthetic advisor block for single-agent case
				try:
					persona_name = Path(self.persona_var.get()).stem
				except Exception:
					persona_name = "agent"
				advisor_block = f"- Advisor 1 ({persona_name}, weight 100%): \"{text}\""
				agg_prompt = _fill_agg_template(
					self.agg_template,
					question=question,
					draft=text,
					advisor_block=advisor_block,
				)
				# fatigue-aware aggregator: shorter or refusal when tired
				sys_prompt = "You are a careful, concise aggregator."
				if self.fatigue >= self.energy_cap:
					sys_prompt += " You are extremely tired; politely refuse and explain you are too tired. If you must respond, keep it to one short line."
					agg_tokens = 64
				elif self.fatigue >= max(1, self.energy_cap-1):
					sys_prompt += " You are getting tired; keep your answer extremely brief (<= 3 lines) and mention that you are tired."
					agg_tokens = 128
				else:
					agg_tokens = MAX_TOKENS
				final = self._openai_chat([
					{"role": "system", "content": sys_prompt},
					{"role": "user", "content": agg_prompt},
				], max_tokens=agg_tokens)
				self._append("Final", final)
			except Exception as e:
				self._append("Final-Error", str(e))
			self.fatigue += 1
		except Exception as e:
			self._append("Error", str(e))
		finally:
			self.pb.stop()

	def _multi_agent_thread(self, question: str):
		self.pb.start()
		try:
			advisor_lines = []
			for idx, (path, w) in enumerate(self.selected_agents):
				persona = path.read_text(encoding="utf-8") if path.exists() else "You are a helpful, concise assistant."
				if self.fatigue >= self.energy_cap:
					persona += "\n\nFATIGUE_COEF = 2.0  # agent is exhausted"
				messages = [
					{"role": "system", "content": persona},
					{"role": "user", "content": question},
				]
				try:
					ans = self._openai_chat(messages)
				except Exception as e:
					ans = f"Error: {e}"
				# show individual agent answers in chat for transparency
				self._append(f"Agent {idx+1} ({path.stem}, {w*100:.0f}%)", ans)
				advisor_lines.append(f"- Advisor {idx+1} ({path.stem}, weight {w*100:.0f}%): \"{ans}\"")
			# aggregate
			agg_prompt = _fill_agg_template(
				self.agg_template,
				question=question,
				draft="\n".join(advisor_lines),
				advisor_block="\n".join(advisor_lines),
			)
			sys_prompt = "You are a careful, concise aggregator. Adopt a natural human tone consistent with the highest-weighted advisor. Prioritize higher weights strongly when making the final choice."
			if self.fatigue >= self.energy_cap:
				sys_prompt += " You are extremely tired; politely refuse to perform the task and explain you are too tired. If you must respond, keep it to one short line."
				agg_tokens = 64
			elif self.fatigue >= max(1, self.energy_cap-1):
				sys_prompt += " You are getting tired; keep your answer extremely brief (<= 3 lines) and mention that you are tired."
				agg_tokens = 128
			else:
				agg_tokens = MAX_TOKENS
			final = self._openai_chat([
				{"role": "system", "content": sys_prompt},
				{"role": "user", "content": agg_prompt},
			], max_tokens=agg_tokens)
			self._append("Final", final)
			self.fatigue += 1
		except Exception as e:
			self._append("Error", str(e))
		finally:
			self.pb.stop()

	def _openai_chat(self, messages: List[dict], max_tokens: int | None = None) -> str:
		resp = self.client.chat.completions.create(
			model=self.model_name,
			messages=messages,
			temperature=self.temperature,
			max_tokens=(max_tokens or MAX_TOKENS),
		)
		text = (resp.choices[0].message.content or "").replace("\\n", "\n").strip()
		return text or "<empty response>"

	# -------------- helpers ------------- #
	def _read_persona(self) -> str:
		path = Path(self.persona_var.get())
		try:
			if path.exists():
				return path.read_text(encoding="utf-8")
			return "You are a helpful, concise assistant."
		except Exception:
			return "You are a helpful, concise assistant."

	def _append(self, who: str, txt: str):
		self.chat_box.configure(state="normal")
		self.chat_box.insert("end", f"{who}: {txt}\n\n")
		self.chat_box.configure(state="disabled")
		self.chat_box.see("end")

	def _reset_chat(self, *_):
		self.chat_box.configure(state="normal"); self.chat_box.delete("1.0", "end")
		self.chat_box.configure(state="disabled")
		self.messages, self.fatigue = [], 0


def main():
	OnlineChatGUI().mainloop()


if __name__ == "__main__":
	main()

