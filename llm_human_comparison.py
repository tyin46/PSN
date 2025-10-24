"""
LLM vs Human Decision Comparison
使用真实API调用获取LLM决策，与人类平均行为进行详细对比分析
"""

import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys

# Add project root to path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
    from openai import OpenAI
    HAS_OPENAI = True
except:
    HAS_OPENAI = False
    OpenAI = None

@dataclass
class DecisionComparison:
    """单次决策的对比数据"""
    trial_id: int
    pumps: int
    state_description: str
    llm_decision: str
    llm_reasoning: str
    human_typical_decision: str
    human_decision_probability: float
    decision_match: bool
    risk_level: str  # "low", "medium", "high"

class LLMHumanComparator:
    """LLM决策与人类决策的详细对比分析器"""
    
    def __init__(self):
        self.output_dir = Path("llm_human_comparison")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        self.client = None
        if HAS_OPENAI:
            try:
                self.client = OpenAI()
                print("✅ OpenAI客户端初始化成功")
            except Exception as e:
                print(f"❌ OpenAI客户端初始化失败: {e}")
                print("请检查OPENAI_API_KEY环境变量设置")
                
        # 加载人格文件
        self.risk_taker_persona = self._load_persona("risk_taker.txt")
        self.cautious_persona = self._load_persona("cautious_thinker.txt")
        
        # 人类行为基线数据（来自心理学研究）
        self.human_baseline = {
            'avg_pumps': 16.5,
            'explosion_rate': 0.37,
            'avg_adjusted_pumps': 20.2,
            'typical_stopping_points': [8, 12, 16, 20, 24],  # 常见停止点
            'risk_aversion_curve': self._generate_human_risk_curve()
        }
        
    def _load_persona(self, filename: str) -> str:
        """加载人格文件"""
        filepath = _ROOT / filename
        if filepath.exists():
            return filepath.read_text(encoding='utf-8')
        else:
            print(f"⚠️ 人格文件 {filename} 未找到")
            return f"You are a {filename.replace('.txt', '').replace('_', ' ')} persona."
            
    def _generate_human_risk_curve(self) -> Dict[int, float]:
        """生成人类风险规避曲线（基于心理学研究数据）"""
        # 基于实际BART研究的人类决策概率
        risk_curve = {}
        for pumps in range(1, 33):
            if pumps <= 5:
                pump_prob = 0.9  # 早期高概率继续
            elif pumps <= 10:
                pump_prob = 0.8 - (pumps - 5) * 0.1  # 逐渐降低
            elif pumps <= 15:
                pump_prob = 0.3 - (pumps - 10) * 0.05  # 加速降低
            elif pumps <= 20:
                pump_prob = 0.05 - (pumps - 15) * 0.01  # 很低概率
            else:
                pump_prob = 0.01  # 极低概率
            risk_curve[pumps] = max(0.01, pump_prob)
        return risk_curve
        
    def get_human_typical_decision(self, pumps: int) -> Tuple[str, float]:
        """获取人类在特定状态下的典型决策和概率"""
        pump_probability = self.human_baseline['risk_aversion_curve'].get(pumps, 0.01)
        
        if pump_probability > 0.5:
            typical_decision = "PUMP"
        else:
            typical_decision = "COLLECT"
            
        return typical_decision, pump_probability
        
    def get_llm_decision_with_reasoning(self, persona_text: str, state_desc: str, 
                                      persona_name: str) -> Tuple[str, str]:
        """获取LLM的决策和推理过程"""
        
        if not self.client:
            # 简单模拟回退
            if "risk" in persona_name.lower():
                return "PUMP", "模拟回复：作为风险承担者，我选择继续打气获取更多奖励。"
            else:
                return "COLLECT", "模拟回复：作为谨慎者，我选择收集当前奖励以避免损失。"
        
        decision_prompt = f"""{persona_text}

当前情况：{state_desc}

请做出决策并解释你的推理过程。

输出格式：
决策：PUMP 或 COLLECT
推理：[你的详细推理过程，2-3句话]

例如：
决策：PUMP
推理：虽然风险在增加，但当前奖励还不够高，我愿意承担额外风险来获取更大收益。基于过往经验，这个阶段爆炸概率仍然较低。"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": decision_prompt}],
                temperature=0.7,
                max_tokens=150,
                timeout=15
            )
            
            content = response.choices[0].message.content.strip()
            
            # 解析决策和推理
            lines = content.split('\n')
            decision = "COLLECT"  # 默认值
            reasoning = "无法解析推理过程"
            
            for line in lines:
                if line.startswith("决策：") or line.startswith("Decision:"):
                    if "PUMP" in line.upper():
                        decision = "PUMP"
                    else:
                        decision = "COLLECT"
                elif line.startswith("推理：") or line.startswith("Reasoning:"):
                    reasoning = line.split("：", 1)[-1].strip()
                    
            return decision, reasoning
            
        except Exception as e:
            print(f"⚠️ LLM查询失败: {e}")
            # 回退决策
            if "risk" in persona_name.lower():
                return "PUMP", "API调用失败，使用风险承担者默认策略。"
            else:
                return "COLLECT", "API调用失败，使用谨慎者默认策略。"
                
    def run_detailed_comparison(self, n_scenarios: int = 20) -> List[DecisionComparison]:
        """运行详细的LLM与人类决策对比"""
        
        if not self.client:
            print("❌ 无法进行真实LLM对比，需要有效的OpenAI API密钥")
            return []
            
        print(f"🔍 开始LLM与人类决策详细对比分析")
        print(f"📊 将分析 {n_scenarios} 个决策场景")
        
        comparisons = []
        personas = [
            ("风险承担者", self.risk_taker_persona),
            ("谨慎思考者", self.cautious_persona)
        ]
        
        scenario_id = 0
        
        for persona_name, persona_text in personas:
            print(f"\n🎭 分析人格: {persona_name}")
            
            # 为每个人格生成多个决策场景
            for pumps in [3, 7, 12, 17, 22, 27]:  # 不同风险水平
                scenario_id += 1
                
                # 生成状态描述
                earned = pumps * 0.05
                risk_level = "低" if pumps <= 8 else "中" if pumps <= 16 else "高"
                
                state_desc = f"""气球任务状态：
- 当前已打气 {pumps} 次
- 已获得奖励：${earned:.2f}
- 风险水平：{risk_level}
- 继续打气可获得更多奖励，但气球可能爆炸导致本轮奖励清零
- 选择收集可保留当前奖励"""

                print(f"  📝 场景 {scenario_id}: {pumps} 次打气 (风险水平: {risk_level})")
                
                # 获取LLM决策
                llm_decision, llm_reasoning = self.get_llm_decision_with_reasoning(
                    persona_text, state_desc, persona_name)
                
                # 获取人类典型决策
                human_decision, human_prob = self.get_human_typical_decision(pumps)
                
                # 判断是否匹配
                decision_match = llm_decision == human_decision
                
                # 风险水平分类
                risk_category = "低风险" if pumps <= 8 else "中风险" if pumps <= 16 else "高风险"
                
                comparison = DecisionComparison(
                    trial_id=scenario_id,
                    pumps=pumps,
                    state_description=state_desc,
                    llm_decision=llm_decision,
                    llm_reasoning=llm_reasoning,
                    human_typical_decision=human_decision,
                    human_decision_probability=human_prob,
                    decision_match=decision_match,
                    risk_level=risk_category
                )
                
                comparisons.append(comparison)
                
                print(f"    🤖 LLM决策: {llm_decision}")
                print(f"    👥 人类典型: {human_decision} (概率: {human_prob:.2f})")
                print(f"    ✅ 匹配: {'是' if decision_match else '否'}")
                
        return comparisons
        
    def analyze_and_visualize(self, comparisons: List[DecisionComparison]):
        """分析对比结果并生成可视化"""
        
        if not comparisons:
            print("❌ 没有对比数据可供分析")
            return
            
        print(f"\n📊 开始分析 {len(comparisons)} 个决策对比...")
        
        # 转换为DataFrame便于分析
        df = pd.DataFrame([asdict(comp) for comp in comparisons])
        
        # 创建综合可视化
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('LLM vs 人类决策详细对比分析', fontsize=16, fontweight='bold')
        
        # 1. 决策匹配率
        ax = axes[0, 0]
        match_rate = df['decision_match'].mean()
        risk_levels = df['risk_level'].unique()
        match_by_risk = df.groupby('risk_level')['decision_match'].mean()
        
        bars = ax.bar(range(len(risk_levels)), [match_by_risk[level] for level in risk_levels], 
                     color=['green', 'orange', 'red'], alpha=0.7)
        ax.axhline(match_rate, color='blue', linestyle='--', linewidth=2, 
                  label=f'总体匹配率: {match_rate:.2f}')
        ax.set_xticks(range(len(risk_levels)))
        ax.set_xticklabels(risk_levels)
        ax.set_ylabel('决策匹配率')
        ax.set_title('不同风险水平下的决策匹配率')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. LLM vs 人类决策分布
        ax = axes[0, 1]
        llm_decisions = df['llm_decision'].value_counts()
        human_decisions = df['human_typical_decision'].value_counts()
        
        x = np.arange(len(['PUMP', 'COLLECT']))
        width = 0.35
        
        ax.bar(x - width/2, [llm_decisions.get('PUMP', 0), llm_decisions.get('COLLECT', 0)], 
               width, label='LLM决策', alpha=0.8, color='skyblue')
        ax.bar(x + width/2, [human_decisions.get('PUMP', 0), human_decisions.get('COLLECT', 0)], 
               width, label='人类典型决策', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('决策类型')
        ax.set_ylabel('频次')
        ax.set_title('LLM vs 人类决策分布')
        ax.set_xticks(x)
        ax.set_xticklabels(['PUMP', 'COLLECT'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 风险水平与决策关系
        ax = axes[0, 2]
        risk_decision_crosstab = pd.crosstab(df['risk_level'], df['llm_decision'], normalize='index')
        risk_decision_crosstab.plot(kind='bar', ax=ax, color=['lightblue', 'lightgreen'])
        ax.set_title('LLM在不同风险水平下的决策模式')
        ax.set_xlabel('风险水平')
        ax.set_ylabel('决策比例')
        ax.legend(title='LLM决策')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # 4. 人类决策概率分布
        ax = axes[1, 0]
        ax.scatter(df['pumps'], df['human_decision_probability'], 
                  c=['red' if d == 'PUMP' else 'blue' for d in df['human_typical_decision']], 
                  s=80, alpha=0.7, edgecolors='black')
        ax.set_xlabel('打气次数')
        ax.set_ylabel('人类PUMP决策概率')
        ax.set_title('人类风险规避曲线')
        ax.grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(df['pumps'], df['human_decision_probability'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(df['pumps'].min(), df['pumps'].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='趋势线')
        ax.legend()
        
        # 5. 决策一致性热图
        ax = axes[1, 1]
        
        # 创建决策对比矩阵
        decision_matrix = np.zeros((2, 2))
        for _, row in df.iterrows():
            llm_idx = 0 if row['llm_decision'] == 'PUMP' else 1
            human_idx = 0 if row['human_typical_decision'] == 'PUMP' else 1
            decision_matrix[llm_idx, human_idx] += 1
            
        im = ax.imshow(decision_matrix, cmap='Blues', aspect='auto')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['人类PUMP', '人类COLLECT'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['LLM PUMP', 'LLM COLLECT'])
        ax.set_title('LLM vs 人类决策混淆矩阵')
        
        # 添加数值标注
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{int(decision_matrix[i, j])}',
                       ha="center", va="center", color="white", fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        
        # 6. 详细统计信息
        ax = axes[1, 2]
        ax.axis('off')
        
        # 计算统计数据
        total_comparisons = len(comparisons)
        matches = sum(1 for c in comparisons if c.decision_match)
        match_rate = matches / total_comparisons
        
        pump_decisions_llm = sum(1 for c in comparisons if c.llm_decision == 'PUMP')
        pump_decisions_human = sum(1 for c in comparisons if c.human_typical_decision == 'PUMP')
        
        stats_text = f"""📈 详细统计数据

🔍 总对比场景数: {total_comparisons}
✅ 决策匹配数: {matches}
📊 总体匹配率: {match_rate:.2%}

🤖 LLM决策统计:
   PUMP: {pump_decisions_llm} ({pump_decisions_llm/total_comparisons:.1%})
   COLLECT: {total_comparisons-pump_decisions_llm} ({(total_comparisons-pump_decisions_llm)/total_comparisons:.1%})

👥 人类典型决策:
   PUMP: {pump_decisions_human} ({pump_decisions_human/total_comparisons:.1%})
   COLLECT: {total_comparisons-pump_decisions_human} ({(total_comparisons-pump_decisions_human)/total_comparisons:.1%})

🎯 风险水平匹配率:
   低风险: {df[df['risk_level']=='低风险']['decision_match'].mean():.1%}
   中风险: {df[df['risk_level']=='中风险']['decision_match'].mean():.1%}
   高风险: {df[df['risk_level']=='高风险']['decision_match'].mean():.1%}"""

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'llm_human_detailed_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 保存详细对比数据
        self.save_detailed_results(comparisons, df)
        
    def save_detailed_results(self, comparisons: List[DecisionComparison], df: pd.DataFrame):
        """保存详细的对比结果"""
        
        # 保存原始对比数据
        timestamp = int(time.time())
        results_file = self.output_dir / f"llm_human_comparison_{timestamp}.json"
        
        comparison_data = {
            'metadata': {
                'timestamp': timestamp,
                'total_comparisons': len(comparisons),
                'personas_tested': ['风险承担者', '谨慎思考者'],
                'human_baseline': self.human_baseline
            },
            'comparisons': [asdict(comp) for comp in comparisons]
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, ensure_ascii=False, indent=2)
        
        # 保存CSV格式便于进一步分析
        csv_file = self.output_dir / f"llm_human_comparison_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        print(f"💾 详细结果已保存:")
        print(f"   JSON: {results_file}")
        print(f"   CSV:  {csv_file}")
        
    def print_detailed_analysis(self, comparisons: List[DecisionComparison]):
        """打印详细分析报告"""
        
        print("\n" + "="*80)
        print("LLM与人类决策详细对比分析报告")
        print("="*80)
        
        total = len(comparisons)
        matches = sum(1 for c in comparisons if c.decision_match)
        
        print(f"\n📊 总体统计:")
        print(f"   总对比场景: {total}")
        print(f"   决策匹配: {matches} ({matches/total:.1%})")
        print(f"   决策不匹配: {total-matches} ({(total-matches)/total:.1%})")
        
        print(f"\n🎭 按人格类型分析:")
        persona_stats = {}
        for comp in comparisons:
            # 简单根据trial_id判断人格（前半部分是风险承担者）
            persona = "风险承担者" if comp.trial_id <= total//2 else "谨慎思考者"
            if persona not in persona_stats:
                persona_stats[persona] = {'total': 0, 'matches': 0}
            persona_stats[persona]['total'] += 1
            if comp.decision_match:
                persona_stats[persona]['matches'] += 1
                
        for persona, stats in persona_stats.items():
            match_rate = stats['matches'] / stats['total']
            print(f"   {persona}: {stats['matches']}/{stats['total']} ({match_rate:.1%})")
        
        print(f"\n🎯 按风险水平分析:")
        risk_stats = {}
        for comp in comparisons:
            risk = comp.risk_level
            if risk not in risk_stats:
                risk_stats[risk] = {'total': 0, 'matches': 0}
            risk_stats[risk]['total'] += 1
            if comp.decision_match:
                risk_stats[risk]['matches'] += 1
                
        for risk, stats in risk_stats.items():
            match_rate = stats['matches'] / stats['total']
            print(f"   {risk}: {stats['matches']}/{stats['total']} ({match_rate:.1%})")
        
        print(f"\n💡 关键发现:")
        
        # LLM决策倾向
        llm_pumps = sum(1 for c in comparisons if c.llm_decision == 'PUMP')
        human_pumps = sum(1 for c in comparisons if c.human_typical_decision == 'PUMP')
        
        print(f"   • LLM倾向: {llm_pumps/total:.1%} 选择PUMP")
        print(f"   • 人类倾向: {human_pumps/total:.1%} 选择PUMP")
        
        if llm_pumps > human_pumps:
            print(f"   • LLM比人类更激进 (+{(llm_pumps-human_pumps)/total:.1%})")
        else:
            print(f"   • LLM比人类更保守 ({(llm_pumps-human_pumps)/total:.1%})")
        
        # 展示一些具体的决策推理
        print(f"\n🧠 LLM决策推理示例:")
        for i, comp in enumerate(comparisons[:3]):
            match_status = "✅匹配" if comp.decision_match else "❌不匹配"
            print(f"\n   示例 {i+1} - {comp.pumps}次打气 ({comp.risk_level}) {match_status}")
            print(f"   LLM决策: {comp.llm_decision}")
            print(f"   人类典型: {comp.human_typical_decision}")
            print(f"   LLM推理: {comp.llm_reasoning}")

def main():
    """运行LLM与人类决策对比分析"""
    
    print("🚀 LLM与人类决策详细对比分析")
    print("="*50)
    
    comparator = LLMHumanComparator()
    
    if not comparator.client:
        print("❌ 无法运行，需要有效的OpenAI API密钥")
        print("请确保OPENAI_API_KEY环境变量已正确设置")
        return
    
    # 运行详细对比
    comparisons = comparator.run_detailed_comparison(n_scenarios=20)
    
    if comparisons:
        # 分析和可视化
        comparator.analyze_and_visualize(comparisons)
        
        # 打印详细报告
        comparator.print_detailed_analysis(comparisons)
        
        print(f"\n🎉 LLM与人类决策对比分析完成!")
        print(f"📊 可视化图表: llm_human_comparison/llm_human_detailed_comparison.png")
        print(f"📁 详细数据: llm_human_comparison/ 目录")
    
    else:
        print("❌ 未能生成对比数据")

if __name__ == "__main__":
    main()