import matplotlib.pyplot as plt
import numpy as np

def main():
    # Data extracted from inference runs on IMD GFS Data (2024-04-29) - INDIA REGION ONLY
    variables_large = ['MSL', 'Z']
    rmse_orig_large = [146.17, 143.65]
    rmse_fine_large = [87.29, 92.25]
    mae_orig_large = [89.09, 101.93]
    mae_fine_large = [56.90, 64.37]

    variables_temp = ['T2M', 'T']
    rmse_orig_temp = [1.51, 1.24]
    rmse_fine_temp = [0.79, 0.72]
    mae_orig_temp = [1.17, 0.90]
    mae_fine_temp = [0.52, 0.50]

    variables_wind = ['U10', 'V10', 'U', 'V']
    rmse_orig_wind = [1.30, 1.28, 2.65, 2.41]
    rmse_fine_wind = [1.12, 1.16, 2.21, 2.00]
    mae_orig_wind = [0.96, 0.98, 1.88, 1.75]
    mae_fine_wind = [0.84, 0.88, 1.58, 1.43]

    # Plot Setup
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Regional Performance Comparison (India): Original vs Finetuned Pangu-Weather\n(Inference on IMD GFS Data, Valid Time: 2024-04-29)', 
                 fontsize=16, fontweight='bold')

    # Helper function to plot a subplot
    def plot_bar_group(ax, vars_list, r_orig, r_fine, m_orig, m_fine, title, ylabel):
        x = np.arange(len(vars_list))
        width = 0.2
        
        # Plotting
        ax.bar(x - width*1.5, r_orig, width, label='Original RMSE', color='#e74c3c', edgecolor='black')
        ax.bar(x - width*0.5, r_fine, width, label='Finetuned RMSE', color='#3498db', edgecolor='black')
        ax.bar(x + width*0.5, m_orig, width, label='Original MAE', color='#f1c40f', edgecolor='black')
        ax.bar(x + width*1.5, m_fine, width, label='Finetuned MAE', color='#2ecc71', edgecolor='black')
        
        # Annotations
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(vars_list, fontsize=12)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Calculate and show percentage improvement on the chart (RMSE)
        for i in range(len(vars_list)):
            improvement = ((r_orig[i] - r_fine[i]) / r_orig[i]) * 100
            ax.text(x[i] - width, r_orig[i] + (max(r_orig)*0.02), f"-{improvement:.1f}%", 
                    ha='center', va='bottom', fontweight='bold', color='green')

    # 1. Pressure & Height (Large Values)
    plot_bar_group(axes[0], variables_large, rmse_orig_large, rmse_fine_large, 
                   mae_orig_large, mae_fine_large, 'Pressure & Geopotential', 'Error Value')

    # 2. Temperatures
    plot_bar_group(axes[1], variables_temp, rmse_orig_temp, rmse_fine_temp, 
                   mae_orig_temp, mae_fine_temp, 'Temperatures', 'Error Value')

    # 3. Winds
    plot_bar_group(axes[2], variables_wind, rmse_orig_wind, rmse_fine_wind, 
                   mae_orig_wind, mae_fine_wind, 'Wind Components', 'Error Value')

    # Formatting and saving
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_filename = "finetune_performance_comparison_india.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved successfully as '{output_filename}'")

if __name__ == "__main__":
    main()
