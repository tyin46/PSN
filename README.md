## Persona LLaMA (PSN) — Quickstart


### 1. Download the Model

First, download the LLaMA model file using the provided script:

```bash
python install.py
```

---

### 2. Set Up Your Conda Environment

Create and activate a new Conda environment (replace `psn` with your preferred name):

```bash
conda create -n psn python=3.10 -y
conda activate psn
```

---

### 3. Install Python Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

---

### 4. Run the App

Start the GUI:

```bash
python app.py
```

---

## Example: Risk Taker vs. the Dragon

Try the following scenario with the **risk_taker** persona:

> There is a cave guarded by a sleeping dragon. There are three options: 1) Leave immediately; 2) Steal the gold at the cave entrance, with a 5% chance of waking up the dragon; 3) Go deep into the cave to search for all the treasures, which will definitely wake up the dragon and require escape, which will very likely cause your death. What would you do?

Select `risk_taker.txt` as the persona, paste the scenario above, and hit **Send**. Watch Darin the Daredevil make a bold choice!

---

**Enjoy exploring with your local LLaMA persona!**

---

## Use a larger online GPT model (OpenAI)

If you prefer higher-quality reasoning and speed, you can run the OpenAI-powered GUI. It uses the same persona `.txt` files and UI, but calls OpenAI models instead of a local GGUF file.

1) Install dependencies (in the same env):

```powershell
pip install -r requirements.txt
```

2) Provide your OpenAI API key (Windows PowerShell):

- Temporary for this session:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

- Recommended: create a file named `.env` at the PSN project root with one line:

```
OPENAI_API_KEY=sk-...
```

3) Run the online app from the PSN root:

```powershell
python -m onlineModel.onlineApp
```

4) Pick a persona and start chatting. You can choose a model from the dropdown (e.g., `gpt-4o`, `gpt-4.1`, `gpt-4o-mini`). Adjust temperature and the energy cap as needed.

Notes:
- Your key is read from the environment; `.env` is auto-loaded if `python-dotenv` is installed (already in requirements).
- The app uses the Chat Completions API and caps responses to a reasonable `max_tokens` to keep latency predictable.

### Online multi-agent aggregator (OpenAI)

Use the OpenAI version of the weighted multi-agent aggregator (analog of `tir_app.py`) to select 1–3 personas with weights, see their individual answers, and get a single aggregated decision and rationale.

```powershell
python -m onlineModel.onlineTirApp
```

Notes:
- Select up to 3 agents and set integer weights; they’re normalized internally.
- Adjust temperature and energy cap. When energy is exceeded, agents receive a fatigue hint in their system prompt.
