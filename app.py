"""
app.py  –  Persona LLaMA GUI (light theme, working energy entry, visible progress bar)

pip install -U customtkinter "langchain-community>=0.2.0" llama-cpp-python
"""

import threading
from pathlib import Path
from tkinter import filedialog, messagebox
import customtkinter as ctk
from langchain_community.chat_models import ChatLlamaCpp
from langchain.prompts import ChatPromptTemplate

TXT_DIR = Path(".")
MODELS_DIR = Path("models")
MAX_TOKENS = 256


def list_files(folder: Path, ext: str):
    return [p for p in folder.glob(f"*{ext}")]


class ChatGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        self.title("Persona LLaMA — Light")
        self.geometry("960x720")
        self.resizable(False, False)

        # runtime state
        self.llm = None
        self.fatigue = 0
        self.energy_cap = 5
        self.messages = []

        # files
        self.persona_files = list_files(TXT_DIR, ".txt")
        self.model_files = list_files(MODELS_DIR, ".gguf") or [Path("(Browse…)")]

        self._build_ui()
        if self.model_files and self.model_files[0].suffix == ".gguf":
            self._load_llm(self.model_files[0])

    # ---------------- UI ---------------- #
    def _build_ui(self):
        self.configure(fg_color="#f0f0f0")

        # ── top bar
        top = ctk.CTkFrame(self, corner_radius=10)
        top.pack(fill="x", padx=14, pady=8)

        # model
        ctk.CTkLabel(top, text="Model").pack(side="left", padx=(10, 0))
        self.model_var = ctk.StringVar(value=str(self.model_files[0]))
        ctk.CTkOptionMenu(
            top, variable=self.model_var,
            values=[str(p) for p in self.model_files],
            command=self._on_model_change
        ).pack(side="left", padx=6)
        ctk.CTkButton(top, text="Browse…", command=self._browse_model,
                      width=90).pack(side="left", padx=(0, 16))

        # persona
        ctk.CTkLabel(top, text="Persona").pack(side="left")
        self.persona_var = ctk.StringVar(value=str(self.persona_files[0]))
        ctk.CTkOptionMenu(
            top, variable=self.persona_var,
            values=[str(p) for p in self.persona_files],
            command=lambda *_: self._reset_chat()
        ).pack(side="left", padx=6)

        # energy controls
        ctk.CTkLabel(top, text="Energy").pack(side="left", padx=(24, 0))
        self.energy_slider = ctk.CTkSlider(
            top, from_=1, to=20, number_of_steps=19,
            command=self._slider_changed, width=140
        )
        self.energy_slider.set(self.energy_cap)
        self.energy_slider.pack(side="left", padx=6)

        self.energy_entry = ctk.CTkEntry(top, width=60, justify="center")
        self.energy_entry.insert(0, str(self.energy_cap))
        self.energy_entry.pack(side="left", padx=(0, 4))
        ctk.CTkButton(top, text="√", width=50,
                      command=self._entry_apply).pack(side="left")

        # ── chat window
        self.chat_box = ctk.CTkTextbox(self, wrap="word",
                                       state="disabled", corner_radius=10,
                                       font=("Consolas", 12))
        self.chat_box.pack(fill="both", expand=True, padx=14, pady=(8, 4))

        # ── progress bar
        self.pb = ctk.CTkProgressBar(self, mode="indeterminate")
        self.pb.pack(fill="x", padx=14, pady=(0, 8))
        self.pb.stop()  # hide animation until needed

        # ── bottom input
        bottom = ctk.CTkFrame(self, corner_radius=10)
        bottom.pack(fill="x", padx=14, pady=6)
        self.entry = ctk.CTkEntry(bottom)
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 6), pady=6)
        ctk.CTkButton(bottom, text="Send", width=90,
                      command=self._on_send).pack(side="left", padx=(0, 6))
        ctk.CTkButton(bottom, text="Clear", width=80,
                      command=self._reset_chat).pack(side="left")

    # ---------- energy control ---------- #
    def _slider_changed(self, val):
        self.energy_cap = int(float(val))
        self.energy_entry.delete(0, "end")
        self.energy_entry.insert(0, str(self.energy_cap))

    def _entry_apply(self):
        txt = self.energy_entry.get().strip()
        if txt.isdigit() and 1 <= int(txt) <= 20:
            self.energy_cap = int(txt)
            self.energy_slider.set(self.energy_cap)
        else:
            messagebox.showinfo("Energy must be 1-20",
                                "Please enter an integer between 1 and 20.")
            self.energy_entry.delete(0, "end")
            self.energy_entry.insert(0, str(self.energy_cap))

    # ---------- model picker ------------ #
    def _on_model_change(self, *_):
        p = Path(self.model_var.get())
        if p.name == "(Browse…)":
            self._browse_model()
        else:
            self._load_llm(p)

    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select GGUF model", filetypes=[("GGUF files", "*.gguf")]
        )
        if path:
            self.model_var.set(path)
            self._load_llm(Path(path))

    # ---------- load LLM ---------------- #
    def _load_llm(self, path: Path):
        if not path.exists():
            messagebox.showerror("Error", f"Model not found:\n{path}"); return
        self._append("System", f"Loading {path.name} …")
        self.pb.start()
        try:
            self.llm = ChatLlamaCpp(
                model_path=str(path), n_ctx=4096,
                temperature=0.8, top_p=0.9,
                max_tokens=MAX_TOKENS, stop=[]
            )
            self._append("System", "Model loaded.\n")
            self._reset_chat()
        except Exception as e:
            self._append("Error", str(e))
            messagebox.showerror("LLM load failed", str(e))
            self.llm = None
        finally:
            self.pb.stop()

    # ---------- chat workflow ----------- #
    def _on_send(self):
        if not self.llm:
            messagebox.showinfo("No model", "Please load a GGUF model first."); return
        msg = self.entry.get().strip()
        if not msg: return
        self.entry.delete(0, "end")
        self._append("You", msg)
        self.messages.append(("user", msg))
        prompt = self._build_prompt(msg)
        threading.Thread(target=self._llm_thread, args=(prompt,), daemon=True).start()

    def _build_prompt(self, user_msg):
        persona = Path(self.persona_var.get()).read_text(encoding="utf-8")
        if self.fatigue >= self.energy_cap:
            persona += "\n\nFATIGUE_COEF = 2.0  # agent is exhausted"
        tmpl = ChatPromptTemplate.from_messages(
            [("system", persona)] + self.messages + [("user", "{q}")]
        )
        return tmpl.format(q=user_msg)

    def _llm_thread(self, prompt):
        # ---- inside _llm_thread ----
        self.pb.start()
        try:
            msg_obj = self.llm.invoke(prompt)         # AIMessage
            resp = msg_obj.content.replace("\\n", "\n")
            self.messages.append(("assistant", resp))
            self._append("Agent", resp)
            self.fatigue += 1
        except Exception as e:
            self._append("Error", str(e))
        finally:
            self.pb.stop()


    # ---------- helpers ----------------- #
    def _append(self, who, txt):
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", f"{who}: {txt}\n\n")
        self.chat_box.configure(state="disabled")
        self.chat_box.see("end")

    def _reset_chat(self, *_):
        self.chat_box.configure(state="normal"); self.chat_box.delete("1.0", "end")
        self.chat_box.configure(state="disabled")
        self.messages, self.fatigue = [], 0


if __name__ == "__main__":
    ChatGUI().mainloop()
