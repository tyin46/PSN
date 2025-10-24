import matplotlib.pyplot as plt

risk = [90, 80, 70, 60, 50, 40, 30, 20, 10]
pumps = [28, 27, 10, 22, 11, 6, 7, 4, 4]
outcomes = ['exploded', 'exploded', 'safe', 'safe', 'safe', 'safe', 'safe', 'safe', 'safe']
colors = ['red' if o == 'exploded' else 'green' for o in outcomes]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(risk, pumps, linewidth=1, linestyle='-', color='gray')
ax.scatter(risk, pumps, c=colors, s=120, edgecolors='black', zorder=3)
ax.axhline(24, linestyle='--', linewidth=1.2, label='Explosion threshold (24)')
ax.axhline(32, linestyle=':', linewidth=1.2, label='Maximum pumps (32)')
ax.set_xlabel('Risk-taker weight (%)')
ax.set_ylabel('Number of pumps')
ax.set_xticks(risk)
ax.set_title('BART Outcome by Personality Mix')
ax.legend()

ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(risk)
ax2.set_xticklabels([100 - r for r in risk])
ax2.set_xlabel('Cautious-thinker weight (%)')

plt.tight_layout()
plt.show()
