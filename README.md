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
