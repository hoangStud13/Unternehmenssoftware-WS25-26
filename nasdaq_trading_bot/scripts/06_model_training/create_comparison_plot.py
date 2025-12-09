import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# HARDCODED METRICS (MSE)
# Bitte hier die Werte aus den Plots/Logs eintragen!
# ---------------------------------------------------------

# 1. Baselines (aus metrics.txt)
LOSS_DUMMY_MEAN = 1.264742
LOSS_LINEAR_REG = 0.846074

# 2. Deep Learning Models (Bitte Werte anpassen!)
# Beispielwerte (bitte ersetzen mit den Werten aus den Bildern):
LOSS_RNN          = 0.56   # PLACEHOLDER: Wert aus 06_rnn_results.png ablesen
LOSS_LSTM         = 0.34   # PLACEHOLDER: Wert aus 06_lstm_results.png ablesen
LOSS_FEED_FORWARD = 0.0037   # PLACEHOLDER: Wert aus 06_feed_forward_loss_curves_relu.png ablesen

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

models = [
    'Dummy (Mean)',  
    'Linear Reg (Baseline)', 
    'Feed Forward', 
    'RNN',
    'LSTM'
]

values = [
    LOSS_DUMMY_MEAN,
    LOSS_LINEAR_REG,
    LOSS_FEED_FORWARD,
    LOSS_RNN,
    LOSS_LSTM
]

colors = [
    'grey',   # Dummy
    'orange', # RNN
    'red',    # Linear (Baseline)
    'purple', # FF
    'green'   # LSTM
]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, values, color=colors, alpha=0.8)

# Baseline Line
plt.axhline(y=LOSS_LINEAR_REG, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (Linear)')

# Werte als Text über den Balken
for bar, val in zip(bars, values):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2., 
        height,
        f'{val:.4f}',
        ha='center', 
        va='bottom',
        fontweight='bold'
    )

plt.title('Validation/Test MSE Loss Comparison', fontsize=14)
plt.ylabel('MSE Loss (Lower is Better)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()

# Speichern
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
output_path = os.path.join(project_root, "nasdaq_trading_bot", "images", "06_model_comparison_final.png")
if not os.path.exists(os.path.dirname(output_path)):
     output_path = os.path.join(project_root, "images", "06_model_comparison_final.png")

plt.tight_layout()
plt.savefig(output_path, dpi=200)
print(f"Comparison plot saved to: {output_path}")
