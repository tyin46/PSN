"""
app.py – Multi-Agent LLaMA GUI with Weighted Aggregation

© 2025 – use freely, no warranty.
"""

import threading
from pathlib import Path
from tkinter import filedialog, messagebox
import customtkinter as ctk
from langchain_community.chat_models import ChatLlamaCpp
from langchain.prompts import ChatPromptTemplate

# --------- configuration -------------------------------------------------
TXT_DIR    = Path(".")          # where persona .txt live
MODELS_DIR = Path("models")     # default model folder
MAX_TOKENS = 256                # generation cap
MAX_AGENTS = 3                  # UI limit

# -------------------------------------------------------------------------

def list_files(folder: Path, ext: str):
    return [p for p in folder.glob(f"*{ext}")]

def load_persona(path: Path) -> str:
    """read a persona txt file (utf-8)"""
    return path.read_text(encoding="utf-8")

def _to_text(msg):
    """
    Convert ChatLlamaCpp.invoke() output to plain text.
    It returns AIMessage on some versions; fallback to str().
    """
    return msg.content if hasattr(msg, "content") else str(msg)

# --------------- aggregation prompt --------------------------------------
AGG_PROMPT = """You are an **Aggregation Agent**. Your task is to read the
question and the advice given by up to three advisors, each with an assigned
weight (importance). Treat the weights as highly significant: prioritize and trust
the advice of higher-weight advisors much more, using their input as the main basis
for your decision. Only consider lower-weight advisors if they provide unique or
critical information not addressed by higher-weight advisors.

Deliver **one clear final choice** based primarily on the highest-weighted advice,
and provide a short, human-like rationale that explicitly references the influence
of advisor weights.

### Question
{question}

### Advisors
{advisor_block}

### Output format (return exactly this):
chosen_action: <single choice, one line>
rationale: <2-4 sentences, clearly showing how the weights influenced your reasoning>
confidence: <0-1 float representing your overall confidence, higher if the top-weighted advisor was decisive>
"""

# -------------------------------------------------------------------------
class AgentSelector(ctk.CTkToplevel):
    """popup for choosing up to MAX_AGENTS persona files + weights"""
    def __init__(self, master, files, callback):
        super().__init__(master)
        self.title("Select agents (max 3)")
        self.grab_set()                        # modal
        self.callback = callback
        self.vars = []                         # (checkVar, weightEntry, path)
        ctk.CTkLabel(self, text="Tick up to 3 agents and set weight (1-100)",
                     font=("",12,"bold")).pack(pady=(8,4))
        frame = ctk.CTkFrame(self); frame.pack(padx=12, pady=8, fill="both")
        for p in files:
            row = ctk.CTkFrame(frame); row.pack(fill="x", pady=2)
            var = ctk.BooleanVar()
            chk = ctk.CTkCheckBox(row, text=p.name, variable=var,
                                  onvalue=True, offvalue=False)
            chk.pack(side="left")
            weight = ctk.CTkEntry(row, width=50, justify="center")
            weight.insert(0,"33")
            weight.pack(side="right", padx=4)
            self.vars.append((var, weight, p))
        ctk.CTkButton(self, text="Apply", command=self.apply)\
              .pack(pady=(0,8))

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
                if w<=0 or w>100:
                    messagebox.showinfo("Weight", "Weights must be 1-100")
                    return
                selected.append((path, w))
                total_w += w
        if not selected:
            messagebox.showinfo("None","Select at least one agent"); return
        # normalize weights -> percentages
        selected = [(p, round(w/total_w,3)) for p,w in selected]
        self.callback(selected)
        self.destroy()

# -------------------------------------------------------------------------
class ChatGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        self.title("Persona LLaMA – Weighted Multi-Agent")
        self.geometry("1080x760")
        self.resizable(False, False)

        # state -------------------
        self.llm = None
        self.energy_cap = 5
        self.fatigue = 0
        self.messages = []
        self.selected_agents = []   # list[(Path, weight_float)]
        self.agent_boxes = []       # CTkTextbox widgets

        # files -------------------
        self.persona_files = list_files(TXT_DIR, ".txt")
        self.model_files   = list_files(MODELS_DIR, ".gguf") or [Path("(Browse…)")]

        # UI ----------------------
        self._build_ui()
        if self.model_files and self.model_files[0].suffix==".gguf":
            self._load_llm(self.model_files[0])

    # ----- UI construction ---------------------------------------------
    def _build_ui(self):
        top = ctk.CTkFrame(self, corner_radius=10); top.pack(fill="x", padx=12, pady=6)

        # model picker
        ctk.CTkLabel(top, text="Model").pack(side="left", padx=(10,0))
        self.model_var = ctk.StringVar(value=str(self.model_files[0]))
        ctk.CTkOptionMenu(top, variable=self.model_var,
                          values=[str(p) for p in self.model_files],
                          command=self._on_model_change)\
            .pack(side="left", padx=6)
        ctk.CTkButton(top, text="Browse…", width=90,
                      command=self._browse_model).pack(side="left", padx=(0,18))

        # agent selector
        ctk.CTkButton(top, text="Select agents…",
                      command=self._open_agent_selector).pack(side="left")

        # energy controls
        ctk.CTkLabel(top, text="Energy").pack(side="left", padx=(26,0))
        self.energy_slider = ctk.CTkSlider(top, from_=1, to=5,
                                           number_of_steps=19,
                                           command=self._energy_slide, width=140)
        self.energy_slider.set(self.energy_cap)
        self.energy_slider.pack(side="left", padx=6)
        self.energy_entry = ctk.CTkEntry(top, width=60, justify="center")
        self.energy_entry.insert(0,str(self.energy_cap)); self.energy_entry.pack(side="left")
        ctk.CTkButton(top, text="Apply", width=50,
                      command=self._energy_apply).pack(side="left", padx=(4,0))

        # agent output frame
        self.agent_frame = ctk.CTkFrame(self, corner_radius=10)
        self.agent_frame.pack(fill="both", expand=True, padx=8, pady=(2,2))
        self._build_agent_boxes()

        # final answer box
        ctk.CTkLabel(self, text="Final Aggregated Answer",
                     font=("",13,"bold")).pack(pady=(2,0))
        self.final_box = ctk.CTkTextbox(self, height=240, wrap="word",
                                        state="disabled", corner_radius=10,
                                        font=("Consolas", 12))
        self.final_box.pack(fill="x", padx=8, pady=(2,4))

        # progress bar
        self.pb = ctk.CTkProgressBar(self, mode="indeterminate")
        self.pb.pack(fill="x", padx=12, pady=(0,6)); self.pb.stop()

        # input line
        bottom = ctk.CTkFrame(self, corner_radius=10); bottom.pack(fill="x", padx=12, pady=6)
        self.entry = ctk.CTkEntry(bottom)
        self.entry.pack(side="left", fill="x", expand=True, padx=(0,6), pady=6)
        ctk.CTkButton(bottom, text="Send", width=90, command=self._on_send)\
            .pack(side="left", padx=(0,6))
        ctk.CTkButton(bottom, text="Clear", width=80, command=self._reset_chat)\
            .pack(side="left")

    # build empty agent text boxes
    def _build_agent_boxes(self):
        # clear previous
        for w in self.agent_frame.winfo_children(): w.destroy()
        self.agent_boxes = []
        cols = max(1,len(self.selected_agents))
        for idx,(path,weight) in enumerate(self.selected_agents or [(None,0)]):
            col_frame = ctk.CTkFrame(self.agent_frame, corner_radius=10)
            col_frame.grid(row=0,column=idx, sticky="nsew", padx=4, pady=2)
            self.agent_frame.grid_columnconfigure(idx,weight=1)
            name = path.name if path else "Agent"
            ctk.CTkLabel(col_frame, text=f"{name}  ({weight*100:.0f}%)",
                         font=("",12,"bold")).pack(anchor="w", padx=4, pady=(2,0))
            box = ctk.CTkTextbox(col_frame, height=240, wrap="word",
                                 state="disabled", font=("Consolas",11))
            box.pack(fill="both", expand=True, padx=2, pady=2)
            self.agent_boxes.append(box)

    # ---------------- energy handlers ----------------
    def _energy_slide(self,val):
        self.energy_cap = int(float(val))
        self.energy_entry.delete(0,"end"); self.energy_entry.insert(0,str(self.energy_cap))

    def _energy_apply(self):
        txt=self.energy_entry.get().strip()
        if txt.isdigit() and 1<=int(txt)<=5:
            self.energy_cap=int(txt); self.energy_slider.set(self.energy_cap)
        else:
            messagebox.showinfo("Energy","Enter 1-5")

    # ---------------- model handlers -----------------
    def _on_model_change(self,*_):
        p=Path(self.model_var.get())
        if p.name=="(Browse…)": self._browse_model()
        else: self._load_llm(p)

    def _browse_model(self):
        path=filedialog.askopenfilename(title="Select GGUF model",
                                        filetypes=[("GGUF","*.gguf")])
        if path:
            self.model_var.set(path); self._load_llm(Path(path))

    def _load_llm(self,path:Path):
        if not path.exists(): messagebox.showerror("Missing",f"{path}");return
        self._append_final("System",f"Loading {path.name} …"); self.pb.start()
        try:
            self.llm=ChatLlamaCpp(model_path=str(path),n_ctx=4096,
                                  temperature=0.8,top_p=0.9,
                                  max_tokens=MAX_TOKENS,stop=[])
            self._append_final("System","Model loaded.\n")
            self._reset_chat()
        except Exception as e:
            self._append_final("Error",str(e)); self.llm=None
        finally: self.pb.stop()

    # -------------- agent selector popup -------------
    def _open_agent_selector(self):
        AgentSelector(self,self.persona_files,self._set_agents)

    def _set_agents(self,sel):
        self.selected_agents=sel   # list[(Path, weight_float)]
        self._build_agent_boxes()

    # -------------- chat flow ------------------------
    def _on_send(self):
        if not self.llm:
            messagebox.showinfo("No model","Load a GGUF model first."); return
        if not self.selected_agents:
            messagebox.showinfo("No agents","Select 1-3 agents"); return
        query=self.entry.get().strip()
        if not query:return
        self.entry.delete(0,"end")
        self._clear_boxes()
        self.pb.start()
        threading.Thread(target=self._worker,args=(query,),daemon=True).start()

    def _worker(self,question):
        # step 1: generate each agent answer
        advisor_lines=[]
        for idx,(path,w) in enumerate(self.selected_agents):
            persona=load_persona(path)
            if self.fatigue>=self.energy_cap:
                persona+="\n\nFATIGUE_COEF = 2.0"
            prompt=ChatPromptTemplate.from_messages(
                [("system",persona),("user",question)]
            ).format()
            try:
                ans=_to_text(self.llm.invoke(prompt)).replace("\\n","\n")
            except Exception as e:
                ans=f"Error: {e}"
            self._set_box(idx,ans)
            advisor_lines.append(f"- Advisor {idx+1} ({path.stem}, weight {w*100:.0f}%): \"{ans}\"")
        # step 2: aggregation
        agg_prompt=AGG_PROMPT.format(
            question=question,
            advisor_block="\n".join(advisor_lines)
        )
        try:
            final=_to_text(self.llm.invoke(agg_prompt)).replace("\\n","\n")
        except Exception as e:
            final=f"Error during aggregation: {e}"
        self._append_final("Aggregator",final)
        self.fatigue+=1
        self.pb.stop()

    # -------------- helpers --------------------------
    def _clear_boxes(self):
        for box in self.agent_boxes:
            box.configure(state="normal"); box.delete("1.0","end"); box.configure(state="disabled")
        self.final_box.configure(state="normal"); self.final_box.delete("1.0","end"); self.final_box.configure(state="disabled")

    def _set_box(self,idx,text):
        box=self.agent_boxes[idx]
        box.configure(state="normal"); box.insert("end",text); box.configure(state="disabled")

    def _append_final(self,label,text):
        self.final_box.configure(state="normal")
        self.final_box.insert("end",f"{label}: {text}\n\n")
        self.final_box.configure(state="disabled"); self.final_box.see("end")

    def _reset_chat(self,*_):
        self._clear_boxes()
        self.messages=[]; self.fatigue=0

# -------------------------------------------------------------------------
if __name__ == "__main__":
    ChatGUI().mainloop()
