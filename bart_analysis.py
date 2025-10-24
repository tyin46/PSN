"""
BART Results Analysis and Comparison
Loads and compares results from different evaluation approaches
Creates publication-ready visualizations
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.patches as mpatches

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16
})

class BARTResultsAnalyzer:
    """Analyzes and compares BART evaluation results"""
    
    def __init__(self, results_dir: str = "."):
        self.results_dir = Path(results_dir)
        self.human_baseline = {
            'avg_pumps': 16.5,
            'explosion_rate': 0.37,
            'avg_adjusted_pumps': 20.2
        }
        
    def load_results(self, filename: str) -> Optional[List[Dict]]:
        """Load results from JSON file"""
        filepath = self.results_dir / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            return None
            
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
            
    def results_to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        data = []
        for result in results:
            row = result.get('config', {}).copy()
            # Add all metrics
            for key in ['total_reward', 'explosion_rate', 'avg_pumps', 
                       'avg_adjusted_pumps', 'consistency_score', 'optimal_stopping_score']:
                if key in result:
                    row[key] = result[key]
            data.append(row)
        return pd.DataFrame(data)
        
    def create_method_comparison(self, method_results: Dict[str, List[Dict]]):
        """Compare results from different evaluation methods"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('BART Evaluation Method Comparison', fontsize=18, fontweight='bold')
        
        # Convert all results to DataFrames
        dfs = {}
        for method, results in method_results.items():
            if results:
                dfs[method] = self.results_to_dataframe(results)
        
        if not dfs:
            print("No results to compare")
            return
            
        # Define metrics to compare
        metrics = [
            ('avg_pumps', 'Average Pumps'),
            ('explosion_rate', 'Explosion Rate'),
            ('total_reward', 'Total Reward'),
            ('consistency_score', 'Consistency Score'),
            ('optimal_stopping_score', 'Optimal Stopping Score'),
            ('avg_adjusted_pumps', 'Avg Adjusted Pumps')
        ]
        
        for i, (metric, title) in enumerate(metrics):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            # Box plot comparison
            data_for_plot = []
            labels = []
            
            for method, df in dfs.items():
                if metric in df.columns:
                    data_for_plot.append(df[metric].values)
                    labels.append(method)
            
            if data_for_plot:
                bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
                
                # Color the boxes
                colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                
                # Add human baseline if applicable
                if metric in self.human_baseline:
                    ax.axhline(self.human_baseline[metric], color='red', 
                              linestyle='--', linewidth=2, label='Human Baseline')
                    ax.legend()
                
                ax.set_title(title)
                ax.set_ylabel(title)
                ax.grid(True, alpha=0.3)
                
                # Rotate x-labels if needed
                if len(labels) > 2:
                    plt.setp(ax.get_xticklabels(), rotation=45)
            
        plt.tight_layout()
        plt.savefig(self.results_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_risk_personality_analysis(self, results: List[Dict]):
        """Deep dive into risk vs cautious personality effects"""
        
        df = self.results_to_dataframe(results)
        if df.empty:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Risk vs Cautious Personality Analysis', fontsize=16, fontweight='bold')
        
        # 1. Risk weight vs performance metrics
        ax = axes[0, 0]
        risk_groups = df.groupby('risk_weight').agg({
            'avg_pumps': ['mean', 'std'],
            'explosion_rate': ['mean', 'std'],
            'total_reward': ['mean', 'std']
        })
        
        x = risk_groups.index
        
        # Plot with error bars
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        
        line1 = ax.errorbar(x, risk_groups[('avg_pumps', 'mean')], 
                           yerr=risk_groups[('avg_pumps', 'std')],
                           marker='o', linewidth=2, capsize=5, label='Avg Pumps', color='blue')
        
        line2 = ax2.errorbar(x, risk_groups[('explosion_rate', 'mean')], 
                            yerr=risk_groups[('explosion_rate', 'std')],
                            marker='s', linewidth=2, capsize=5, label='Explosion Rate', color='red')
        
        line3 = ax3.errorbar(x, risk_groups[('total_reward', 'mean')], 
                            yerr=risk_groups[('total_reward', 'std')],
                            marker='^', linewidth=2, capsize=5, label='Total Reward', color='green')
        
        ax.set_xlabel('Risk Taker Weight')
        ax.set_ylabel('Average Pumps', color='blue')
        ax2.set_ylabel('Explosion Rate', color='red')
        ax3.set_ylabel('Total Reward', color='green')
        ax.set_title('Risk Weight Effects on Performance')
        
        # Combined legend
        lines = [line1, line2, line3]
        labels = ['Avg Pumps', 'Explosion Rate', 'Total Reward']
        ax.legend(lines, labels, loc='upper left')
        
        # 2. Decision consistency heatmap
        ax = axes[0, 1]
        if 'temperature' in df.columns and 'consistency_score' in df.columns:
            pivot = df.pivot_table(values='consistency_score', 
                                 index='temperature', 
                                 columns='risk_weight', 
                                 aggfunc='mean')
            
            im = ax.imshow(pivot, cmap='viridis', aspect='auto', origin='lower')
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f'{x:.1f}' for x in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f'{y:.1f}' for y in pivot.index])
            ax.set_xlabel('Risk Weight')
            ax.set_ylabel('Temperature')
            ax.set_title('Decision Consistency Heatmap')
            plt.colorbar(im, ax=ax)
        
        # 3. Risk-reward scatter with human comparison
        ax = axes[1, 0]
        scatter = ax.scatter(df['explosion_rate'], df['total_reward'], 
                           c=df['risk_weight'], s=80, alpha=0.7, 
                           cmap='RdYlBu', edgecolors='black', linewidth=0.5)
        
        # Add human baseline point
        human_reward_est = self.human_baseline['avg_pumps'] * 0.05 * (1 - self.human_baseline['explosion_rate'])
        ax.scatter(self.human_baseline['explosion_rate'], human_reward_est, 
                  marker='*', s=300, color='red', edgecolors='black', 
                  linewidth=2, label='Human Baseline', zorder=5)
        
        ax.set_xlabel('Explosion Rate')
        ax.set_ylabel('Total Reward')
        ax.set_title('Risk-Reward Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Risk Weight')
        
        # 4. Performance distribution by personality type
        ax = axes[1, 1]
        
        # Categorize by personality dominance
        df['personality_type'] = df['risk_weight'].apply(
            lambda x: 'Risk-Dominant' if x > 0.6 
                     else 'Cautious-Dominant' if x < 0.4 
                     else 'Balanced'
        )
        
        # Box plot by personality type
        personality_types = ['Risk-Dominant', 'Balanced', 'Cautious-Dominant']
        reward_data = [df[df['personality_type'] == ptype]['total_reward'].values 
                      for ptype in personality_types if ptype in df['personality_type'].values]
        valid_types = [ptype for ptype in personality_types if ptype in df['personality_type'].values]
        
        if reward_data:
            bp = ax.boxplot(reward_data, labels=valid_types, patch_artist=True)
            colors = ['lightcoral', 'lightyellow', 'lightblue']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                
        ax.set_ylabel('Total Reward')
        ax.set_title('Reward Distribution by Personality Type')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'personality_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_human_alignment_analysis(self, results: List[Dict]):
        """Analyze alignment with human behavior"""
        
        df = self.results_to_dataframe(results)
        if df.empty:
            return
            
        # Calculate human similarity scores
        similarity_scores = []
        for _, row in df.iterrows():
            score = 0
            n_metrics = 0
            for metric in ['avg_pumps', 'explosion_rate', 'avg_adjusted_pumps']:
                if metric in row and metric in self.human_baseline:
                    normalized_diff = abs(row[metric] - self.human_baseline[metric]) / self.human_baseline[metric]
                    score += normalized_diff
                    n_metrics += 1
            similarity_scores.append(score / n_metrics if n_metrics > 0 else float('inf'))
            
        df['human_similarity'] = similarity_scores
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Human Behavior Alignment Analysis', fontsize=16, fontweight='bold')
        
        # 1. Human similarity by configuration
        ax = axes[0, 0]
        best_matches = df.nsmallest(10, 'human_similarity')
        
        scatter = ax.scatter(df['risk_weight'], df['human_similarity'], 
                           c=df.get('temperature', 0.8), s=60, alpha=0.7, 
                           cmap='coolwarm', edgecolors='black', linewidth=0.3)
        
        # Highlight best matches
        ax.scatter(best_matches['risk_weight'], best_matches['human_similarity'],
                  color='red', s=120, marker='*', edgecolors='black', 
                  linewidth=1, label='Top 10 Matches', zorder=5)
        
        ax.set_xlabel('Risk Weight')
        ax.set_ylabel('Human Similarity Score (lower is better)')
        ax.set_title('Configuration Similarity to Human Baseline')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if 'temperature' in df.columns:
            plt.colorbar(scatter, ax=ax, label='Temperature')
        
        # 2. Metrics comparison with human baseline
        ax = axes[0, 1]
        
        metrics = ['avg_pumps', 'explosion_rate', 'avg_adjusted_pumps']
        human_vals = [self.human_baseline[m] for m in metrics if m in self.human_baseline]
        ai_means = [df[m].mean() for m in metrics if m in df.columns]
        ai_stds = [df[m].std() for m in metrics if m in df.columns]
        
        valid_metrics = [m for m in metrics if m in df.columns and m in self.human_baseline]
        human_vals = [self.human_baseline[m] for m in valid_metrics]
        ai_means = [df[m].mean() for m in valid_metrics]
        ai_stds = [df[m].std() for m in valid_metrics]
        
        x_pos = np.arange(len(valid_metrics))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, human_vals, width, 
                      label='Human Baseline', color='red', alpha=0.7)
        bars2 = ax.bar(x_pos + width/2, ai_means, width, yerr=ai_stds,
                      label='AI Mean ± SD', color='blue', alpha=0.7, capsize=5)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('AI vs Human Performance Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in valid_metrics], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Best matching configurations details
        ax = axes[1, 0]
        
        top_5_matches = df.nsmallest(5, 'human_similarity')
        
        # Create a comparison table visualization
        config_labels = []
        for i, (_, row) in enumerate(top_5_matches.iterrows()):
            risk_w = row.get('risk_weight', 0)
            temp = row.get('temperature', 0.8)
            config_labels.append(f"Config {i+1}\nR:{risk_w:.1f} T:{temp:.1f}")
        
        metrics_to_show = ['avg_pumps', 'explosion_rate']
        
        x_pos = np.arange(len(config_labels))
        width = 0.3
        
        for i, metric in enumerate(metrics_to_show):
            if metric in top_5_matches.columns:
                values = top_5_matches[metric].values
                human_val = self.human_baseline.get(metric, 0)
                
                bars = ax.bar(x_pos + i*width, values, width, 
                             label=f'AI {metric.replace("_", " ").title()}', alpha=0.7)
                ax.axhline(human_val, color=f'C{i}', linestyle='--', 
                          label=f'Human {metric.replace("_", " ").title()}')
        
        ax.set_xlabel('Top Matching Configurations')
        ax.set_ylabel('Values')
        ax.set_title('Best Human-Matching Configurations')
        ax.set_xticks(x_pos + width/2)
        ax.set_xticklabels(config_labels, fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Similarity score distribution
        ax = axes[1, 1]
        
        ax.hist(df['human_similarity'], bins=20, alpha=0.7, color='skyblue', 
               edgecolor='black', density=True)
        
        # Mark percentiles
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            val = np.percentile(df['human_similarity'], p)
            ax.axvline(val, color='red', linestyle=':', alpha=0.7)
            ax.text(val, ax.get_ylim()[1]*0.9, f'{p}th', rotation=90, ha='right')
        
        ax.set_xlabel('Human Similarity Score')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Human Similarity Scores')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'human_alignment.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed analysis
        self._print_human_alignment_summary(df)
        
    def _print_human_alignment_summary(self, df: pd.DataFrame):
        """Print detailed human alignment analysis"""
        
        print("\n" + "="*80)
        print("HUMAN BEHAVIOR ALIGNMENT ANALYSIS")
        print("="*80)
        
        best_match = df.loc[df['human_similarity'].idxmin()]
        
        print(f"\nBest Human-Matching Configuration:")
        print(f"- Risk Weight: {best_match.get('risk_weight', 'N/A'):.2f}")
        print(f"- Cautious Weight: {best_match.get('cautious_weight', 'N/A'):.2f}")
        print(f"- Temperature: {best_match.get('temperature', 'N/A'):.2f}")
        print(f"- Similarity Score: {best_match['human_similarity']:.3f}")
        
        print(f"\nPerformance Comparison (Best Match vs Human):")
        for metric in ['avg_pumps', 'explosion_rate', 'avg_adjusted_pumps']:
            if metric in best_match and metric in self.human_baseline:
                ai_val = best_match[metric]
                human_val = self.human_baseline[metric]
                diff_pct = ((ai_val - human_val) / human_val) * 100
                print(f"- {metric.replace('_', ' ').title()}: AI={ai_val:.3f}, Human={human_val:.3f} ({diff_pct:+.1f}%)")
        
        print(f"\nTop 5 Human-Matching Configurations:")
        top_5 = df.nsmallest(5, 'human_similarity')
        for i, (_, row) in enumerate(top_5.iterrows()):
            print(f"  {i+1}. Risk={row.get('risk_weight', 'N/A'):.1f}, "
                  f"Temp={row.get('temperature', 'N/A'):.1f}, "
                  f"Similarity={row['human_similarity']:.3f}")
        
        print(f"\nSimilarity Score Statistics:")
        print(f"- Mean: {df['human_similarity'].mean():.3f}")
        print(f"- Median: {df['human_similarity'].median():.3f}")
        print(f"- Best (lowest): {df['human_similarity'].min():.3f}")
        print(f"- Worst (highest): {df['human_similarity'].max():.3f}")
        print(f"- Std: {df['human_similarity'].std():.3f}")
        
    def create_publication_summary(self, results: List[Dict], title: str = "BART Evaluation Results"):
        """Create publication-ready summary visualization"""
        
        df = self.results_to_dataframe(results)
        if df.empty:
            return
            
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Risk weight effects (large plot)
        ax1 = fig.add_subplot(gs[0, :2])
        risk_grouped = df.groupby('risk_weight').agg({
            'avg_pumps': ['mean', 'std'],
            'explosion_rate': ['mean', 'std'],
            'total_reward': ['mean', 'std']
        })
        
        x = risk_grouped.index
        ax1_twin = ax1.twinx()
        
        line1 = ax1.errorbar(x, risk_grouped[('avg_pumps', 'mean')], 
                            yerr=risk_grouped[('avg_pumps', 'std')],
                            marker='o', linewidth=3, capsize=5, 
                            label='Average Pumps', color='blue', markersize=8)
        
        line2 = ax1_twin.errorbar(x, risk_grouped[('explosion_rate', 'mean')], 
                                 yerr=risk_grouped[('explosion_rate', 'std')],
                                 marker='s', linewidth=3, capsize=5,
                                 label='Explosion Rate', color='red', markersize=8)
        
        ax1.set_xlabel('Risk Taker Weight', fontsize=14)
        ax1.set_ylabel('Average Pumps', color='blue', fontsize=14)
        ax1_twin.set_ylabel('Explosion Rate', color='red', fontsize=14)
        ax1.set_title('A. Risk Personality Effects', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Combined legend
        lines = [line1, line2]
        labels = ['Average Pumps', 'Explosion Rate']
        ax1.legend(lines, labels, loc='upper left', fontsize=12)
        
        # 2. Performance heatmap
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'temperature' in df.columns:
            pivot = df.pivot_table(values='total_reward', 
                                 index='temperature', 
                                 columns='risk_weight', 
                                 aggfunc='mean')
            
            im = ax2.imshow(pivot, cmap='viridis', aspect='auto', origin='lower')
            ax2.set_xticks(range(len(pivot.columns)))
            ax2.set_xticklabels([f'{x:.1f}' for x in pivot.columns])
            ax2.set_yticks(range(len(pivot.index)))
            ax2.set_yticklabels([f'{y:.1f}' for y in pivot.index])
            ax2.set_xlabel('Risk Weight', fontsize=14)
            ax2.set_ylabel('Temperature', fontsize=14)
            ax2.set_title('B. Total Reward Heatmap', fontsize=16, fontweight='bold')
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label('Total Reward', fontsize=12)
        
        # 3. Human comparison
        ax3 = fig.add_subplot(gs[1, :2])
        
        metrics = ['avg_pumps', 'explosion_rate', 'avg_adjusted_pumps']
        valid_metrics = [m for m in metrics if m in df.columns and m in self.human_baseline]
        
        if valid_metrics:
            human_vals = [self.human_baseline[m] for m in valid_metrics]
            ai_means = [df[m].mean() for m in valid_metrics]
            ai_stds = [df[m].std() for m in valid_metrics]
            
            x_pos = np.arange(len(valid_metrics))
            width = 0.35
            
            bars1 = ax3.bar(x_pos - width/2, human_vals, width, 
                           label='Human Baseline', color='red', alpha=0.8)
            bars2 = ax3.bar(x_pos + width/2, ai_means, width, yerr=ai_stds,
                           label='AI Performance', color='blue', alpha=0.8, capsize=5)
            
            ax3.set_xlabel('Metrics', fontsize=14)
            ax3.set_ylabel('Values', fontsize=14)
            ax3.set_title('C. Human vs AI Comparison', fontsize=16, fontweight='bold')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([m.replace('_', ' ').title() for m in valid_metrics])
            ax3.legend(fontsize=12)
            ax3.grid(True, alpha=0.3)
        
        # 4. Risk-reward scatter
        ax4 = fig.add_subplot(gs[1, 2:])
        scatter = ax4.scatter(df['explosion_rate'], df['total_reward'], 
                             c=df.get('risk_weight', 0.5), s=100, alpha=0.7, 
                             cmap='RdYlBu', edgecolors='black', linewidth=0.5)
        
        # Add human baseline estimate
        human_reward_est = self.human_baseline['avg_pumps'] * 0.05 * (1 - self.human_baseline['explosion_rate'])
        ax4.scatter(self.human_baseline['explosion_rate'], human_reward_est, 
                   marker='*', s=400, color='red', edgecolors='black', 
                   linewidth=2, label='Human Baseline', zorder=5)
        
        ax4.set_xlabel('Explosion Rate', fontsize=14)
        ax4.set_ylabel('Total Reward', fontsize=14)
        ax4.set_title('D. Risk-Reward Trade-off', fontsize=16, fontweight='bold')
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Risk Weight', fontsize=12)
        
        # 5. Performance statistics table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Create summary statistics
        stats_data = []
        for metric in ['avg_pumps', 'explosion_rate', 'total_reward', 'consistency_score']:
            if metric in df.columns:
                stats_data.append([
                    metric.replace('_', ' ').title(),
                    f"{df[metric].mean():.3f}",
                    f"{df[metric].std():.3f}",
                    f"{df[metric].min():.3f}",
                    f"{df[metric].max():.3f}",
                    f"{self.human_baseline.get(metric, 'N/A')}"
                ])
        
        if stats_data:
            table = ax5.table(cellText=stats_data,
                             colLabels=['Metric', 'Mean', 'Std', 'Min', 'Max', 'Human'],
                             cellLoc='center',
                             loc='center',
                             bbox=[0.1, 0.3, 0.8, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(stats_data) + 1):
                for j in range(6):
                    cell = table[(i, j)]
                    if i == 0:  # Header
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
            
            ax5.set_title('E. Performance Statistics Summary', fontsize=16, fontweight='bold', y=0.8)
        
        # 6. Best configurations
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('off')
        
        # Find best configurations for different objectives
        objectives = [
            ('Highest Reward', 'total_reward', True),
            ('Lowest Explosions', 'explosion_rate', False),
            ('Most Consistent', 'consistency_score', True),
            ('Best Human Match', 'human_similarity', False)
        ]
        
        # Calculate human similarity if not already done
        if 'human_similarity' not in df.columns:
            similarity_scores = []
            for _, row in df.iterrows():
                score = 0
                n_metrics = 0
                for metric in ['avg_pumps', 'explosion_rate', 'avg_adjusted_pumps']:
                    if metric in row and metric in self.human_baseline:
                        normalized_diff = abs(row[metric] - self.human_baseline[metric]) / self.human_baseline[metric]
                        score += normalized_diff
                        n_metrics += 1
                similarity_scores.append(score / n_metrics if n_metrics > 0 else float('inf'))
            df['human_similarity'] = similarity_scores
        
        best_configs_text = "F. Best Configurations:\n\n"
        
        for obj_name, metric, maximize in objectives:
            if metric in df.columns:
                if maximize:
                    best_row = df.loc[df[metric].idxmax()]
                else:
                    best_row = df.loc[df[metric].idxmin()]
                
                risk_w = best_row.get('risk_weight', 'N/A')
                temp = best_row.get('temperature', 'N/A')
                value = best_row[metric]
                
                best_configs_text += f"• {obj_name}: Risk={risk_w:.1f}, Temp={temp:.1f} (Score: {value:.3f})\n"
        
        ax6.text(0.05, 0.95, best_configs_text, transform=ax6.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.savefig(self.results_dir / 'publication_summary.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main analysis function"""
    
    analyzer = BARTResultsAnalyzer()
    
    print("BART Results Analysis and Comparison")
    print("="*50)
    
    # Try to load different types of results from various directories
    search_dirs = [Path("."), Path("quick_bart_results"), Path("bart_persona_results"), Path("bart_results")]
    result_patterns = [
        "quick_bart_results_*.json",
        "bart_persona_evaluation_*.json", 
        "bart_evaluation_*.json"
    ]
    
    all_results = {}
    
    # Look for result files in multiple directories
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for pattern in result_patterns:
            files = list(search_dir.glob(pattern))
            if files:
                # Load the most recent file
                latest_file = max(files, key=lambda p: p.stat().st_mtime)
                # Set the analyzer to look in the correct directory
                analyzer.results_dir = latest_file.parent
                results = analyzer.load_results(latest_file.name)
                if results:
                    method_name = pattern.split('_')[0:2]
                    method_name = '_'.join(method_name).replace('_results', '').title()
                    all_results[method_name] = results
                    print(f"✓ Loaded {method_name}: {len(results)} configurations from {latest_file}")
    
    if not all_results:
        print("No result files found. Please run an evaluation first.")
        print("Available evaluation scripts:")
        print("- quick_bart_evaluation.py (fastest, no dependencies)")
        print("- bart_persona_evaluation.py (uses OpenAI API)")
        print("- bart_evaluation.py (comprehensive simulation)")
        return
    
    # If we have results, create comprehensive analysis
    if len(all_results) > 1:
        print(f"\nCreating method comparison with {len(all_results)} methods...")
        analyzer.create_method_comparison(all_results)
    
    # Use the first available results for detailed analysis
    primary_results = list(all_results.values())[0]
    primary_method = list(all_results.keys())[0]
    
    print(f"\nCreating detailed analysis for {primary_method}...")
    
    # Create comprehensive visualizations
    analyzer.create_risk_personality_analysis(primary_results)
    analyzer.create_human_alignment_analysis(primary_results)
    analyzer.create_publication_summary(primary_results, f"BART {primary_method} Results")
    
    print(f"\n🎉 Analysis complete!")
    print(f"📊 Visualizations saved to current directory")
    print(f"📁 Check these files:")
    print(f"   - method_comparison.png (if multiple methods)")
    print(f"   - personality_analysis.png")
    print(f"   - human_alignment.png")
    print(f"   - publication_summary.png")

if __name__ == "__main__":
    main()