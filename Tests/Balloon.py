import threading, random, time
from pathlib import Path
from tkinter import filedialog, messagebox
import customtkinter as ctk
from langchain_community.chat_models import ChatLlamaCpp
from langchain.prompts import ChatPromptTemplate

"""
Balloon.py – GUI BART tester (v2)
- 可视化显示已选人格及其每一步决策
- 进度条显示整体完成百分比
© 2025 – free for any use
"""

# ------------------- 常量 -------------------
TXT_DIR = Path('.')            # 人格 txt 存放目录
MODELS_DIR = Path('models')    # GGUF 模型目录
MAX_TOKENS = 64                # 单次生成 token 上限
MAX_AGENTS = 3                 # 最多并行人格
STEP_REWARD = 0.05             # 每打气一次收益

# ------------ 任务环境 -----------------------
class BART:
    def __init__(self, max_pumps: int = 32):
        self.max_pumps = max_pumps
        self.step_reward = STEP_REWARD
        self.reset()

    def reset(self):
        self.threshold = random.randint(1, self.max_pumps)
        self.pumps = 0
        self.done = False
        self.exploded = False
        return self._obs()

    # 当前状态
    def _obs(self):
        return dict(pumps=self.pumps, exploded=int(self.exploded))

    def step(self, act: int):
        """act: 1 = PUMP, 0 = COLLECT"""
        if self.done:
            raise RuntimeError('Call reset() first')
        if act == 1:  # PUMP
            self.pumps += 1
            if self.pumps >= self.threshold:
                self.exploded = True
                self.done = True
                rew = 0.0
            else:
                rew = self.step_reward
        else:        # COLLECT
            self.done = True
            rew = self.pumps * self.step_reward
        return self._obs(), rew, self.done, {}

# -------------- GUI 组件：选择人格 -----------------
class AgentSelector(ctk.CTkToplevel):
    def __init__(self, master, files, callback):
        super().__init__(master)
        self.title('Select agents (max 3)')
        self.grab_set()
        self.callback = callback
        self.vars = []  # (boolVar, entry, path)

        ctk.CTkLabel(self, text='Tick up to 3 agents and set weight 1–100').pack(pady=(8, 4))
        frame = ctk.CTkFrame(self)
        frame.pack(padx=12, pady=8, fill='both')
        for p in files:
            row = ctk.CTkFrame(frame)
            row.pack(fill='x', pady=2)
            var = ctk.BooleanVar()
            chk = ctk.CTkCheckBox(row, text=p.name, variable=var)
            chk.pack(side='left')
            ent = ctk.CTkEntry(row, width=50, justify='center')
            ent.insert(0, '33')
            ent.pack(side='right', padx=4)
            self.vars.append((var, ent, p))

        ctk.CTkButton(self, text='Apply', command=self.apply).pack(pady=8)

    def apply(self):
        sel, total = [], 0
        for var, ent, p in self.vars:
            if var.get():
                if len(sel) == MAX_AGENTS:
                    messagebox.showinfo('Limit', f'Max {MAX_AGENTS} agents')
                    return
                if not ent.get().isdigit():
                    messagebox.showinfo('Weight', 'Weight must be int 1–100')
                    return
                w = int(ent.get())
                if w < 1 or w > 100:
                    messagebox.showinfo('Weight', 'Weight 1–100')
                    return
                sel.append((p, w))
                total += w
        if not sel:
            messagebox.showinfo('None', 'Select at least one agent')
            return
        sel = [(p, round(w / total, 3)) for p, w in sel]
        self.callback(sel)
        self.destroy()

# ----------------- 聚合提示 ----------------------
AGG_PROMPT = """You are an Aggregation Agent for a balloon risk task.\nState:\n{state}\nAdvisor suggestions (each already weighted):\n{advisor_block}\n\nChoose the single action that best reflects the highest-weighted advice unless lower-weighted advice adds unique safety info.\nOutput ONLY one word (PUMP or COLLECT)."""

# ----------------- 主 GUI ------------------------
class GUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode('light')
        self.title('BART Persona Tester')
        self.geometry('1080x760')
        self.resizable(False, False)

        # ---- 状态 ----
        self.llm = None
        self.personas = []             # [(Path, weight)]
        self.trials = 30
        self.model_files = [p for p in MODELS_DIR.glob('*.gguf')] or [Path('(Browse…)')]
        self.persona_files = [p for p in TXT_DIR.glob('*.txt')]

        self._build_ui()
        if self.model_files and self.model_files[0].suffix == '.gguf':
            self._load_llm(self.model_files[0])

    # ---------- UI ----------
    def _build_ui(self):
        top = ctk.CTkFrame(self, corner_radius=10)
        top.pack(fill='x', padx=12, pady=6)
        # 模型
        ctk.CTkLabel(top, text='Model').pack(side='left', padx=(6, 0))
        self.model_var = ctk.StringVar(value=str(self.model_files[0]))
        ctk.CTkOptionMenu(top, variable=self.model_var, values=[str(p) for p in self.model_files],
                          command=self._model_change).pack(side='left', padx=6)
        ctk.CTkButton(top, text='Browse…', command=self._browse_model, width=90).pack(side='left', padx=(0, 12))
        # 人格选择
        ctk.CTkButton(top, text='Select agents…', command=self._open_agents).pack(side='left')
        # 试次
        ctk.CTkLabel(top, text='Trials').pack(side='left', padx=(24, 0))
        self.trial_entry = ctk.CTkEntry(top, width=60, justify='center')
        self.trial_entry.insert(0, '30')
        self.trial_entry.pack(side='left', padx=4)
        ctk.CTkButton(top, text='Apply', command=self._apply_trials, width=60).pack(side='left')
        # Start
        self.start_btn = ctk.CTkButton(top, text='Start', command=self._start, width=100)
        self.start_btn.pack(side='right', padx=8)

        # --------- Agent boxes ---------
        self.agent_frame = ctk.CTkFrame(self, corner_radius=10)
        self.agent_frame.pack(fill='both', expand=False, padx=8, pady=(4, 2))
        self.agent_boxes = []  # will fill later

        # --------- Output log ---------
        ctk.CTkLabel(self, text='Run Log', font=('', 13, 'bold')).pack()
        self.log_box = ctk.CTkTextbox(self, height=300, wrap='word', font=('Consolas', 11), state='disabled')
        self.log_box.pack(fill='both', expand=True, padx=8, pady=4)

        # --------- Progress bar ---------
        self.pb = ctk.CTkProgressBar(self, mode='determinate')
        self.pb.pack(fill='x', padx=12, pady=(0, 6))
        self.pb.set(0)

    # --------- Agent panels ---------
    def _refresh_agent_boxes(self):
        for w in self.agent_frame.winfo_children():
            w.destroy()
        self.agent_boxes = []
        if not self.personas:
            return
        for idx, (p, w) in enumerate(self.personas):
            f = ctk.CTkFrame(self.agent_frame, corner_radius=8)
            f.grid(row=0, column=idx, sticky='nsew', padx=4, pady=2)
            self.agent_frame.grid_columnconfigure(idx, weight=1)
            ctk.CTkLabel(f, text=f'{p.stem} ({w * 100:.0f}%)', font=('', 11, 'bold')).pack(anchor='w', padx=4, pady=(2, 0))
            box = ctk.CTkTextbox(f, height=120, wrap='word', font=('Consolas', 10), state='disabled')
            box.pack(fill='both', expand=True, padx=2, pady=2)
            self.agent_boxes.append(box)

    # --------- 日志写入 ---------
    def _log(self, text: str):
        self.log_box.configure(state='normal')
        self.log_box.insert('end', text + '\n')
        self.log_box.configure(state='disabled')
        self.log_box.see('end')

    def _set_agent_box(self, idx, text):
        box = self.agent_boxes[idx]
        box.configure(state='normal')
        box.delete('1.0', 'end')
        box.insert('end', text)
        box.configure(state='disabled')

    # --------- Model loading ---------
    def _browse_model(self):
        path = filedialog.askopenfilename(title='Select GGUF model', filetypes=[('GGUF', '*.gguf')])
        if path:
            self.model_var.set(path)
            self._load_llm(Path(path))

    def _model_change(self, *_):
        p = Path(self.model_var.get())
        if p.name == '(Browse…)':
            self._browse_model()
        else:
            self._load_llm(p)

    def _load_llm(self, path: Path):
        if not path.exists():
            messagebox.showerror('Error', f'Model {path} not found')
            return
        self._log(f'Loading model {path.name} …')
        self.start_btn.configure(state='disabled')
        self.update()
        try:
            self.llm = ChatLlamaCpp(model_path=str(path), n_ctx=2048, temperature=0.7, top_p=0.9,
                                    max_tokens=MAX_TOKENS, stop=[])
            self._log('Model loaded.')
        except Exception as e:
            messagebox.showerror('Error', str(e))
            self.llm = None
        self.start_btn.configure(state='normal')

    # --------- Persona selection ---------
    def _open_agents(self):
        AgentSelector(self, self.persona_files, self._set_agents)

    def _set_agents(self, sel):
        self.personas = sel
        self._refresh_agent_boxes()

    # --------- Trial count ---------
    def _apply_trials(self):
        txt = self.trial_entry.get().strip()
        if txt.isdigit() and int(txt) > 0:
            self.trials = int(txt)
        else:
            messagebox.showinfo('Input', 'Enter a positive integer')

    # --------- Run ---------
    def _start(self):
        if not self.llm:
            messagebox.showinfo('No model', 'Load a GGUF model first')
            return
        if not self.personas:
            messagebox.showinfo('No agents', 'Select 1-3 agents')
            return
        self.log_box.configure(state='normal'); self.log_box.delete('1.0', 'end'); self.log_box.configure(state='disabled')
        for box in self.agent_boxes:
            box.configure(state='normal'); box.delete('1.0', 'end'); box.configure(state='disabled')
        self.pb.set(0)
        self.start_btn.configure(state='disabled')
        threading.Thread(target=self._worker, daemon=True).start()

    # --------- Worker thread ---------
    def _worker(self):
        env = BART()
        total_money, explosions, adj_pumps = 0.0, 0, []
        for t in range(self.trials):
            env.reset()
            self._log(f'Balloon {t + 1}/{self.trials} start')
            # 回合循环
            while True:
                state_txt = f'Balloon {t + 1}, pumps={env.pumps}, earned={env.pumps * STEP_REWARD:.2f}'
                advisor_lines, actions = [], []
                # 每个 advisor 决策
                for idx, (p, w) in enumerate(self.personas):
                    persona = p.read_text(encoding='utf-8')
                    prompt = ChatPromptTemplate.from_messages([
                        ('system', persona),
                        ('user', state_txt + '. Reply with PUMP or COLLECT only.')
                    ]).format()
                    try:
                        ans = self.llm.invoke(prompt).content.strip().upper()
                    except Exception as e:
                        ans = 'COLLECT'
                    act = 'PUMP' if 'PUMP' in ans else 'COLLECT'
                    self._set_agent_box(idx, act)
                    advisor_lines.append(f'- ({w * 100:.0f}%) {act}')
                    actions.append((act, w))
                # 聚合
                agg_prompt = AGG_PROMPT.format(state=state_txt, advisor_block='\n'.join(advisor_lines))
                try:
                    agg = self.llm.invoke(agg_prompt).content.strip().upper()
                except Exception as e:
                    agg = 'COLLECT'
                action = 1 if agg == 'PUMP' else 0
                self._log(f'Aggregator: {agg}')
                obs, rew, done, _ = env.step(action)
                if done:
                    total_money += rew
                    if env.exploded:
                        explosions += 1
                    else:
                        adj_pumps.append(env.pumps)
                    self._log(f'Balloon end – pumps={env.pumps}, exploded={env.exploded}, reward={rew:.2f}')
                    break
            # 更新进度
            self.pb.set((t + 1) / self.trials)

        avg_adj = sum(adj_pumps) / len(adj_pumps) if adj_pumps else 0.0
        summary = (f'--- SUMMARY ---\n'
                   f'Trials: {self.trials}\n'
                   f'Total money: ${total_money:.2f}\n'
                   f'Explosions: {explosions}\n'
                   f'Average adjusted pumps: {avg_adj:.2f}')
        self._log(summary)
        self.start_btn.configure(state='normal')

# ----------------- main -------------------------
if __name__ == '__main__':
    GUI().mainloop()
