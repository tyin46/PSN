# BART Empirical Evaluation System

This package provides a comprehensive empirical evaluation system for the Balloon Analogue Risk Task (BART) using personality-driven AI agents. The system implements grid search through personality quality proportions, comprehensive metrics collection, and publication-ready visualizations.

## 🚀 Quick Start

1. **Setup dependencies:**
   ```bash
   python setup_evaluation.py
   ```

2. **Run quick evaluation (no external APIs needed):**
   ```bash
   python quick_bart_evaluation.py
   ```

3. **Analyze and visualize results:**
   ```bash
   python bart_analysis.py
   ```

## 📊 Evaluation Components

### 1. Quick Evaluation (`quick_bart_evaluation.py`)
- **Fast simulation-based evaluation**
- No external dependencies beyond standard packages
- Simulates personality behavior based on risk/cautious weights
- Grid search through 45 configurations (9 risk weights × 5 temperatures)
- Generates comprehensive visualizations

**Key Features:**
- 30 trials per configuration × 3 seeds = 90 trials per config
- Personality simulation based on psychological models
- Human baseline comparison
- Pareto optimal configuration identification

### 2. LLM-Based Evaluation (`bart_persona_evaluation.py`)
- **Real LLM evaluation using OpenAI GPT models**
- Uses actual persona files (`risk_taker.txt`, `cautious_thinker.txt`)
- Aggregates decisions from multiple personas
- More realistic but requires API access

**Requirements:**
```bash
pip install openai python-dotenv
export OPENAI_API_KEY="your-key-here"
```

### 3. Comprehensive Analysis (`bart_analysis.py`)
- **Publication-ready visualizations**
- Loads and compares results from different evaluation methods
- Human behavior alignment analysis
- Method comparison capabilities

## 📈 Metrics Collected

### Performance Metrics
- **Total Reward**: Cumulative reward across all trials
- **Average Pumps**: Mean number of pumps per balloon
- **Explosion Rate**: Proportion of balloons that exploded
- **Adjusted Pumps**: Average pumps for successful (non-exploded) balloons

### Decision Quality Metrics
- **Consistency Score**: Inverse of decision variance (higher = more consistent)
- **Optimal Stopping Score**: How close decisions are to optimal (threshold-1)
- **Human Similarity**: Distance from human baseline performance

## 🎯 Grid Search Parameters

### Risk Personality Weights
- Range: 0.1 to 0.9 (9 levels)
- Cautious weight = 1 - Risk weight (complementary)
- Tests full spectrum from highly cautious to highly risk-seeking

### Temperature (Decision Randomness)
- Range: 0.3 to 1.2 (5 levels)
- Lower = more deterministic decisions
- Higher = more exploration/randomness

### Multiple Seeds
- 2-3 seeds per configuration for robustness
- Reduces random variation in results
- Enables statistical significance testing

## 📋 Generated Outputs

### Data Files
- `quick_bart_results_[timestamp].json`: Raw results data
- `bart_persona_evaluation_[timestamp].json`: LLM-based results

### Visualizations
- `comprehensive_results.png`: 6-panel overview of all results
- `personality_analysis.png`: Risk vs cautious personality effects
- `human_alignment.png`: Comparison with human baselines
- `publication_summary.png`: Publication-ready summary figure

### Analysis Reports
- Console output with detailed statistics
- Best configuration identification
- Human behavior matching analysis
- Pareto optimal configuration highlighting

## 🏆 Key Findings Framework

The evaluation system is designed to identify:

1. **Optimal Personality Configurations**
   - Balance between risk-taking and caution
   - Temperature settings for decision consistency
   - Trade-offs between reward and safety

2. **Human Behavior Alignment**
   - Configurations that best match human BART performance
   - Psychological validity of AI decision-making
   - Baseline comparison metrics

3. **Robustness Analysis**
   - Consistency across different seeds
   - Parameter sensitivity analysis
   - Performance stability measures

## 🔧 Customization Options

### Modify Evaluation Parameters
```python
# In quick_bart_evaluation.py
risk_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
temperatures = [0.3, 0.5, 0.8, 1.0, 1.2]
n_trials = 30  # trials per configuration
n_seeds = 3    # seeds for robustness
```

### Human Baseline Adjustment
```python
# Update human baseline values
human_baseline = {
    'avg_pumps': 16.5,        # From literature
    'explosion_rate': 0.37,   # From literature  
    'avg_adjusted_pumps': 20.2 # From literature
}
```

### Add Custom Metrics
```python
def custom_metric(trials):
    # Your custom metric calculation
    return metric_value
```

## 📚 Background: BART Task

The Balloon Analogue Risk Task (BART) is a behavioral measure of risk-taking propensity:

- **Task**: Inflate virtual balloons to earn money
- **Risk**: Each pump risks explosion (lose all money for that balloon)
- **Reward**: More pumps = more money (if balloon doesn't explode)
- **Decision**: When to stop pumping and collect money

### Human Performance Baselines
- Average pumps: ~16.5
- Explosion rate: ~37%
- Individual differences relate to real-world risk behaviors

## 🧠 Personality Simulation

### Risk-Taker Persona
- High probability of continuing to pump
- Focuses on maximum reward potential
- Less sensitive to explosion risk
- Decreases pumping slowly with balloon size

### Cautious Persona  
- Low probability of continuing to pump
- Focuses on preserving accumulated rewards
- Highly sensitive to explosion risk
- Decreases pumping quickly with balloon size

### Temperature Effects
- **Low Temperature (0.3)**: Deterministic, consistent decisions
- **Medium Temperature (0.8)**: Balanced exploration/exploitation
- **High Temperature (1.2)**: High randomness, inconsistent decisions

## 🔬 Research Applications

This evaluation system supports research in:

- **AI Decision-Making**: How personality affects AI risk assessment
- **Human-AI Alignment**: Matching AI behavior to human patterns
- **Behavioral Modeling**: Computational models of risk-taking
- **Parameter Optimization**: Finding optimal personality configurations
- **Comparative Analysis**: Evaluating different decision-making approaches

## 📖 Usage Examples

### Basic Evaluation
```bash
# Run complete evaluation pipeline
python quick_bart_evaluation.py  # ~2-3 minutes
python bart_analysis.py          # Generates all visualizations
```

### LLM Evaluation (with API)
```bash
# Set up API key
export OPENAI_API_KEY="your-key"

# Run LLM-based evaluation  
python bart_persona_evaluation.py  # ~10-15 minutes

# Analyze results
python bart_analysis.py
```

### Custom Analysis
```python
from bart_analysis import BARTResultsAnalyzer

analyzer = BARTResultsAnalyzer()
results = analyzer.load_results("your_results.json")
analyzer.create_publication_summary(results, "Your Title")
```

## 🤝 Contributing

To extend the evaluation system:

1. **Add New Personas**: Create new .txt files with personality descriptions
2. **Custom Metrics**: Implement additional performance measures
3. **New Visualizations**: Add specialized plotting functions
4. **Alternative Models**: Integrate different LLM providers

## 📄 Citation

If you use this evaluation system in research, please cite:

```bibtex
@software{bart_evaluation_2025,
  title = {BART Empirical Evaluation System},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-repo/PSN}
}
```

## 🏁 Expected Results

After running the complete evaluation, you should have:

✅ **Quantitative Results**
- Grid search through 45+ configurations
- Statistical analysis of 2000+ trials
- Performance metrics for each configuration

✅ **Visual Analysis**  
- Risk vs reward trade-off curves
- Temperature effect visualizations
- Human behavior comparison charts
- Best configuration highlights

✅ **Research Insights**
- Optimal personality weight combinations
- Human-aligned AI configurations  
- Decision consistency patterns
- Pareto optimal solutions

---

*This evaluation system provides a complete framework for empirical analysis of personality-driven decision-making in the BART task, with comprehensive metrics, visualizations, and human behavior alignment analysis.*