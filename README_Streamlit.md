# Personality Testing Suite - Streamlit Deployment

🧠 **Interactive web app for BART and PRLT personality testing with LLM parameter generation**

## 🌟 Features

- **Dual Tests**: Balloon Analog Risk Task (BART) + Probabilistic Reversal Learning Task (PRLT)
- **Personality Mixing**: Select and weight up to 5 personality types
- **Temperature Control**: Adjust LLM creativity for parameter generation
- **Graph Storage**: Save and download test results
- **API Integration**: Use your own OpenAI API key
- **Online Ready**: Deployable to Streamlit Cloud, Heroku, or any cloud platform

## 🚀 Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` to access the app.

## 🌐 Online Deployment

### Streamlit Cloud (Recommended)

1. **Fork/Upload** this repository to GitHub
2. **Visit** [share.streamlit.io](https://share.streamlit.io)
3. **Connect** your GitHub repository
4. **Deploy** using these settings:
   - **Main file path**: `streamlit_app.py`
   - **Python version**: 3.9+
   - **Requirements**: `requirements.txt`

### Heroku Deployment

1. **Create** `Procfile`:
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Deploy**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Railway/Render Deployment

1. **Connect** your GitHub repo
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`

## 📊 Usage Guide

### 1. API Key Setup
- Enter your OpenAI API key in the sidebar
- Toggle "Use OpenAI API" for LLM-based parameter generation
- Without API: Falls back to heuristic parameter generation

### 2. Personality Configuration
- **Select personalities** from the sidebar checkboxes
- **Adjust weights** using sliders (0.0 - 2.0)
- **Normalize weights** to ensure proper mixing
- **View current mix** percentages in real-time

### 3. Temperature Control
- **0.0**: Deterministic, consistent parameters
- **0.5**: Balanced creativity and consistency  
- **1.0**: High creativity, more variable parameters

### 4. Running Tests

#### BART (Balloon Analog Risk Task)
- **Number of Balloons**: 5-100 (default: 30)
- **Max Pumps**: 8-128 (default: 64)
- **Explosion Curve**: 0.2-2.0 (default: 1.0)
  - Higher = balloon explodes faster with more pumps

#### PRLT (Probabilistic Reversal Learning Task)
- **Pre-Reversal**: Set reward probabilities (A: 0.1-0.9, B: complement)
- **Post-Reversal**: Set reversed probabilities
- **Pre-trials**: Number of trials before reversal (50-500)

### 5. Results & Storage
- **View**: Real-time plots with Q-values and performance metrics
- **Store**: Save graphs to session storage
- **Download**: Export individual plots as PNG files
- **Manage**: View all stored graphs in dedicated tab

## 📁 File Structure

```
PSN/
├── streamlit_app.py              # Main Streamlit application
├── prlt_personality_proportions.py  # Core PRLT/parameter logic
├── requirements.txt              # Python dependencies
├── README_Streamlit.md          # This file
├── bart_interactive_gui.py      # Original BART GUI (reference)
├── prlt_interactive_gui.py      # Original PRLT GUI (reference)
└── *.txt                        # Personality prompt files
```

## 🔧 Technical Details

### Core Components
- **ParamGenerator**: Extracts learning parameters from personality descriptions
- **CustomBARTSimulator**: Q-learning agent for balloon pumping decisions  
- **CustomPRLTSimulator**: Q-learning agent for probabilistic choice tasks
- **Graph Storage**: Base64-encoded image storage with metadata

### Agent Parameters
- **Learning Rate** (α): How quickly agent updates beliefs (0.0-1.0)
- **Epsilon** (ε): Exploration vs exploitation balance (0.0-1.0)  
- **Perseveration**: Bias toward repeating previous actions (0.0-1.0)
- **Patience**: Window size for convergence detection (5-50 trials)

### Personality Mapping
- **Risk Taker**: ↑ Learning rate, ↓ Epsilon (less exploration, quick decisions)
- **Cautious**: ↑ Perseveration, ↑ Patience (stick to safe choices longer)
- **Easily Tired**: ↓ Patience (gives up sooner)
- **Motivated**: ↑ Learning rate, ↓ Perseveration (adaptive, flexible)
- **Unmotivated**: ↓ Learning rate (slower to learn from feedback)

## 🎯 Research Applications

### Clinical Psychology
- **Risk Assessment**: BART measures risk-taking propensity
- **Cognitive Flexibility**: PRLT measures adaptation to changing environments
- **Personality Profiling**: Multiple personality combinations

### Educational Research  
- **Learning Styles**: Compare different personality-based learning approaches
- **Behavioral Modeling**: Understand decision-making patterns
- **Intervention Design**: Test personality-targeted interventions

### AI/ML Research
- **Agent Behavior**: Study personality effects on reinforcement learning
- **Parameter Sensitivity**: Analyze how personality traits map to learning parameters  
- **Human-AI Alignment**: Model human-like decision patterns

## 🛠️ Customization

### Adding New Personalities
1. Create new `.txt` files with personality descriptions
2. Add to `PERSONA_FILES` in `prlt_personality_proportions.py`
3. Update sidebar personality selection in `streamlit_app.py`

### Modifying Tasks
- **BART**: Adjust explosion probability functions, reward structures
- **PRLT**: Change convergence criteria, add more options, modify reward schedules

### Styling
- Modify Streamlit theme in `.streamlit/config.toml`
- Customize matplotlib plot styles
- Add custom CSS with `st.markdown()`

## 📈 Performance & Scaling

### Optimization Tips
- **Caching**: Use `@st.cache_data` for expensive computations
- **Session State**: Minimize large data in session state
- **API Calls**: Implement rate limiting for OpenAI requests

### Resource Usage
- **Memory**: ~50-200MB per user session
- **CPU**: Moderate for simulations (1-5 seconds typical)
- **Storage**: Minimal (graphs stored as base64 in memory)

## 🔒 Security & Privacy

### API Key Handling  
- **Client-side only**: Keys never sent to server
- **Session-based**: Keys not persisted beyond session
- **No logging**: API keys not logged or stored

### Data Privacy
- **No personal data**: Only simulation parameters and results
- **Local storage**: Graphs stored in browser session only
- **No tracking**: No user analytics or tracking

## 🐛 Troubleshooting

### Common Issues

**API Error**: "Invalid API key"
- ✅ Verify API key is correct and active
- ✅ Check API quota/billing status
- ✅ Toggle "Use OpenAI API" off for heuristic mode

**Simulation Hangs**: Long processing time
- ✅ Reduce number of balloons/trials
- ✅ Check personality weights are normalized
- ✅ Refresh page if stuck

**Graph Display Issues**: Plots not showing
- ✅ Ensure matplotlib backend is compatible
- ✅ Check browser console for JavaScript errors
- ✅ Try refreshing the page

**Deployment Fails**: App won't start on cloud
- ✅ Verify all files are uploaded correctly
- ✅ Check `requirements.txt` has correct versions
- ✅ Ensure Python version compatibility (3.9+)

### Debug Mode
```bash
# Run with debug logging
streamlit run streamlit_app.py --logger.level=debug
```

## 📚 References

- **BART**: Lejuez et al. (2002). Evaluation of a behavioral measure of risk taking
- **PRLT**: Cools et al. (2002). Defining the neural mechanisms of probabilistic reversal learning
- **Q-Learning**: Sutton & Barto (2018). Reinforcement Learning: An Introduction

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)  
5. **Open** Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ using Streamlit** | [Documentation](https://docs.streamlit.io) | [Community](https://discuss.streamlit.io)