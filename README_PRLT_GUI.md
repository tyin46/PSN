# Interactive PRLT GUI

Interactive GUI for exploring Probabilistic Reversal Learning Task (PRLT) with personality mixing and customizable task parameters.

## Features

### 🎛️ Task Reward Controls
- **Pre-Reversal Phase Sliders**: Adjust P(Reward|A) from 0.1 to 0.9
  - P(Reward|B) automatically adjusts to maintain probabilities sum to 1.0
- **Post-Reversal Phase Sliders**: Independent control of reward probabilities after reversal
- Real-time probability display

### 👥 Personality Mixing (Max 5)
- **Available Personas**: Risk Taker, Cautious, Easily Tired, Unmotivated (up to 5 total)
- **Selection**: Check/uncheck personas to include in the mix
- **Weighting**: Individual sliders (0.0-2.0) to control personality influence
- **Normalize Button**: Automatically normalizes all selected weights to sum to 1.0

### 📊 Real-Time Visualization
- **Upper Plot**: Q-values (QA and QB) over trials
  - Blue line: Q(A) - value estimate for option A
  - Red line: Q(B) - value estimate for option B
  - Vertical markers: Reversal point, convergence points
- **Lower Plot**: Choice sequence scatter plot
  - Blue dots: Choices of option A
  - Red dots: Choices of option B
  - Same vertical markers as upper plot

### ⚙️ Simulation Controls
- **API Mode**: Toggle between OpenAI API calls vs heuristic parameter generation
- **Run Button**: Execute simulation with current settings
- **Status Display**: Shows simulation progress

## Usage

### Quick Start
```bash
# Ensure dependencies are installed
pip install tkinter matplotlib numpy

# Run the GUI
python prlt_interactive_gui.py
```

### Basic Workflow
1. **Adjust Task Rewards**: Use sliders to set reward probabilities for pre/post reversal phases
2. **Select Personalities**: Check desired personas and adjust their weights
3. **Normalize** (optional): Click "Normalize Weights" for proper probability distribution
4. **Run Simulation**: Click "Run Simulation" to see results
5. **Analyze**: Examine Q-value curves and choice patterns in the plots

### Understanding the Plots

#### Q-Value Plot (Upper)
- Shows how the agent's value estimates evolve over trials
- **Convergence**: When one Q-value clearly dominates (agent "learns")
- **Switching**: After reversal, when Q-values cross over (agent "adapts")

#### Choice Plot (Lower)  
- Visualizes actual decisions made by the agent
- **Pre-reversal**: Should converge to choosing the higher-reward option
- **Post-reversal**: Should eventually switch to the new higher-reward option

#### Key Markers
- **Orange dashed line**: Reversal point (when probabilities flip)
- **Green dotted line**: Pre-reversal convergence (agent learned initial task)  
- **Purple dotted line**: Post-reversal switch (agent adapted to new task)

### Personality Effects
- **Risk Taker**: Higher exploration, faster learning, quicker switching
- **Cautious**: Lower exploration, slower but more stable learning
- **Easily Tired**: Reduced effort, may give up easier
- **Unmotivated**: Less persistent, more random choices

### Advanced Usage

#### Custom Scenarios
- **Asymmetric Rewards**: Set unequal pre/post probabilities to test different difficulty levels
- **Personality Blends**: Mix multiple personas with different weights to create complex behavioral profiles
- **API vs Heuristic**: Compare LLM-generated parameters vs rule-based ones

#### Troubleshooting
- **No API Key**: Uncheck "Use OpenAI API" for offline mode
- **Slow Performance**: Reduce number of trials or use heuristic mode
- **No Convergence**: Agent may need different parameters or more trials

## Technical Details

### Simulation Parameters
- **Pre-reversal trials**: Up to 200 (stops early if converged)
- **Post-reversal trials**: Up to 1000 (stops early if switched)
- **Convergence criteria**: 90% correct choices in sliding window
- **Window size**: Determined by agent's patience parameter

### API Integration
- Uses OpenAI GPT-4o-mini for parameter extraction when API mode enabled
- Fallback to heuristic parameters when API unavailable
- One API call per simulation (parameter extraction only)

### File Dependencies
- `prlt_personality_proportions.py`: Core simulation classes
- Persona text files: `risk_taker.txt`, `cautious_thinker.txt`, etc.
- `.env`: OpenAI API key (optional, for API mode)

## Example Experiments

1. **Classic PRLT**: Pre=0.75/0.25, Post=0.25/0.75, Pure Risk Taker
2. **Difficult Reversal**: Pre=0.6/0.4, Post=0.4/0.6, Mixed personalities  
3. **Extreme Switch**: Pre=0.9/0.1, Post=0.1/0.9, Pure Cautious
4. **Balanced Task**: Pre=0.5/0.5, Post=0.5/0.5, Any personality (random task)