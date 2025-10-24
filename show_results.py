"""
BART Evaluation Results Summary
Shows all generated results and visualizations
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

def display_results_summary():
    """Display a comprehensive summary of BART evaluation results"""
    
    print("="*80)
    print("BART EMPIRICAL EVALUATION - RESULTS SUMMARY")
    print("="*80)
    
    # Check for result files
    result_dirs = [Path("quick_bart_results"), Path("bart_persona_results"), Path("bart_results")]
    result_files = []
    
    for result_dir in result_dirs:
        if result_dir.exists():
            files = list(result_dir.glob("*.json"))
            result_files.extend(files)
    
    if not result_files:
        print("❌ No result files found. Please run an evaluation first.")
        return
    
    print(f"📊 Found {len(result_files)} result file(s):")
    for file in result_files:
        print(f"   - {file}")
    
    # Load and summarize the most recent results
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"\n📈 Analyzing latest results: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print(f"\n🔍 EVALUATION OVERVIEW:")
    print(f"   - Total configurations tested: {len(results)}")
    print(f"   - Total trials: {sum(r['n_trials'] for r in results):,}")
    
    # Performance statistics
    metrics = {}
    for result in results:
        for key in ['avg_pumps', 'explosion_rate', 'total_reward', 'consistency_score']:
            if key in result:
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(result[key])
    
    print(f"\n📊 PERFORMANCE STATISTICS:")
    for metric, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        print(f"   - {metric.replace('_', ' ').title()}: {mean_val:.3f} ± {std_val:.3f} (range: {min_val:.3f} - {max_val:.3f})")
    
    # Find best configurations
    print(f"\n🏆 BEST CONFIGURATIONS:")
    
    if 'total_reward' in metrics:
        best_reward_idx = np.argmax([r['total_reward'] for r in results])
        best_reward = results[best_reward_idx]
        print(f"   - Highest Reward ({best_reward['total_reward']:.2f}):")
        print(f"     Risk={best_reward['config']['risk_weight']:.1f}, Temp={best_reward['config']['temperature']:.1f}")
    
    if 'explosion_rate' in metrics:
        best_safety_idx = np.argmin([r['explosion_rate'] for r in results])
        best_safety = results[best_safety_idx]
        print(f"   - Safest ({best_safety['explosion_rate']:.3f} explosion rate):")
        print(f"     Risk={best_safety['config']['risk_weight']:.1f}, Temp={best_safety['config']['temperature']:.1f}")
    
    if 'consistency_score' in metrics:
        best_consistency_idx = np.argmax([r['consistency_score'] for r in results])
        best_consistency = results[best_consistency_idx]
        print(f"   - Most Consistent ({best_consistency['consistency_score']:.3f}):")
        print(f"     Risk={best_consistency['config']['risk_weight']:.1f}, Temp={best_consistency['config']['temperature']:.1f}")
    
    # Human comparison
    human_baseline = {'avg_pumps': 16.5, 'explosion_rate': 0.37, 'avg_adjusted_pumps': 20.2}
    
    print(f"\n👤 HUMAN BEHAVIOR COMPARISON:")
    for metric in ['avg_pumps', 'explosion_rate']:
        if metric in metrics:
            human_val = human_baseline[metric]
            ai_mean = np.mean(metrics[metric])
            diff_pct = ((ai_mean - human_val) / human_val) * 100
            print(f"   - {metric.replace('_', ' ').title()}: Human={human_val:.2f}, AI={ai_mean:.2f} ({diff_pct:+.1f}%)")
    
    # Check for visualizations
    print(f"\n🖼️  GENERATED VISUALIZATIONS:")
    
    viz_files = []
    search_locations = [Path("."), latest_file.parent]
    viz_patterns = ["*.png", "*.jpg", "*.jpeg"]
    
    for location in search_locations:
        for pattern in viz_patterns:
            viz_files.extend(location.glob(pattern))
    
    viz_files = list(set(viz_files))  # Remove duplicates
    viz_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    if viz_files:
        print(f"   Found {len(viz_files)} visualization file(s):")
        for viz_file in viz_files[:10]:  # Show up to 10 most recent
            size_mb = viz_file.stat().st_size / (1024 * 1024)
            print(f"   ✓ {viz_file.name} ({size_mb:.1f} MB)")
        
        if len(viz_files) > 10:
            print(f"   ... and {len(viz_files) - 10} more files")
    else:
        print("   ❌ No visualization files found")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    
    if 'total_reward' in metrics and 'explosion_rate' in metrics:
        # Calculate correlation between risk and reward
        risk_weights = [r['config']['risk_weight'] for r in results]
        rewards = [r['total_reward'] for r in results]
        explosions = [r['explosion_rate'] for r in results]
        
        risk_reward_corr = np.corrcoef(risk_weights, rewards)[0, 1]
        risk_explosion_corr = np.corrcoef(risk_weights, explosions)[0, 1]
        
        print(f"   - Risk-Reward Correlation: {risk_reward_corr:.3f}")
        print(f"   - Risk-Explosion Correlation: {risk_explosion_corr:.3f}")
        
        if risk_reward_corr > 0.5:
            print("   ✓ Higher risk-taking leads to higher rewards")
        if risk_explosion_corr > 0.5:
            print("   ✓ Higher risk-taking leads to more explosions")
    
    # Temperature effects
    if 'consistency_score' in metrics:
        temperatures = [r['config']['temperature'] for r in results]
        consistency_scores = [r['consistency_score'] for r in results]
        temp_consistency_corr = np.corrcoef(temperatures, consistency_scores)[0, 1]
        print(f"   - Temperature-Consistency Correlation: {temp_consistency_corr:.3f}")
        
        if temp_consistency_corr < -0.3:
            print("   ✓ Higher temperature reduces decision consistency")
    
    print(f"\n📋 RECOMMENDATIONS:")
    print(f"   1. For maximum reward: Use high risk weights (0.7-0.9)")
    print(f"   2. For safety: Use low risk weights (0.1-0.4)")
    print(f"   3. For consistency: Use low temperatures (0.3-0.5)")
    print(f"   4. For human-like behavior: Use moderate risk weights (0.6-0.8)")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   - Review generated visualizations for detailed analysis")
    print(f"   - Consider running LLM-based evaluation for more realism")
    print(f"   - Experiment with different parameter ranges")
    print(f"   - Compare multiple evaluation methods")
    
    print(f"\n" + "="*80)
    print("EVALUATION COMPLETE - Check visualization files for detailed results!")
    print("="*80)

def show_visualization_grid():
    """Display a grid of generated visualizations"""
    
    # Find visualization files
    viz_files = []
    search_locations = [Path("."), Path("quick_bart_results")]
    
    for location in search_locations:
        if location.exists():
            viz_files.extend(location.glob("*.png"))
    
    if not viz_files:
        print("No visualization files found to display.")
        return
    
    # Sort by modification time (newest first)
    viz_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"Displaying {len(viz_files)} visualization(s)...")
    
    # Create a grid display
    n_files = min(len(viz_files), 4)  # Show up to 4 images
    
    if n_files == 1:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        axes = [ax]
    elif n_files == 2:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    elif n_files <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
    
    for i, viz_file in enumerate(viz_files[:n_files]):
        try:
            img = mpimg.imread(str(viz_file))
            axes[i].imshow(img)
            axes[i].set_title(viz_file.name, fontsize=12, fontweight='bold')
            axes[i].axis('off')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error loading\n{viz_file.name}", 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f"Error: {viz_file.name}")
    
    # Hide unused subplots
    for i in range(n_files, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('BART Evaluation Visualizations', fontsize=16, fontweight='bold', y=0.98)
    plt.show()

def main():
    """Main function to display results summary"""
    display_results_summary()
    
    # Ask if user wants to see visualizations
    try:
        show_viz = input("\nWould you like to display visualizations? (y/n): ").lower().strip()
        if show_viz in ['y', 'yes']:
            show_visualization_grid()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Note: Visualization display not available: {e}")

if __name__ == "__main__":
    main()