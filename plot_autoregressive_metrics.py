import os
import re
import matplotlib.pyplot as plt
from datetime import datetime

def parse_metrics_file(filepath):
    # Data structure: data[model][variable][step] = rmse
    data = {"original": {}, "finetuned": {}}
    step_to_date = {}
    
    current_step = 1
    current_model = None
    current_var = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            m = re.search(r"### STEP (\d+) \| FORECAST TARGET: ([\d\-]+) [\d:]+.*###", line)
            if m:
                current_step = int(m.group(1))
                date_str = m.group(2)
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                step_to_date[current_step] = dt.strftime("%b %d")
                continue
                
            # Model detection
            if "OBSERVED vs ORIGINAL" in line:
                current_model = "original"
            elif "OBSERVED vs FINETUNED" in line:
                current_model = "finetuned"
            elif "ORIGINAL vs FINETUNED" in line:
                current_model = None  # We don't plot this comparison
                
            # Variable detection
            m = re.search(r"Variable: (.+)", line)
            if m and current_model:
                current_var = m.group(1).strip()
                if current_var not in data[current_model]:
                    data[current_model][current_var] = {}
                    
            # RMSE detection
            m = re.search(r"RMSE:\s+([\d\.]+)", line)
            if m and current_model and current_var:
                rmse = float(m.group(1))
                data[current_model][current_var][current_step] = rmse
                
    return data, step_to_date

def plot_metrics(data, step_to_date, output_dir, region_name):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all variables that exist in both models
    vars_orig = set(data["original"].keys())
    vars_fine = set(data["finetuned"].keys())
    common_vars = vars_orig.intersection(vars_fine)
    
    for var in common_vars:
        orig_dict = data["original"][var]
        fine_dict = data["finetuned"][var]
        
        steps = sorted(list(orig_dict.keys()))
        orig_rmse = [orig_dict[s] for s in steps]
        fine_rmse = [fine_dict.get(s, None) for s in steps] # Use get in case of missing data
        
        # Labels like "Day 1\n(May 28)"
        labels = [f"Day {s}\n({step_to_date[s]})" for s in steps]
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, orig_rmse, marker='o', label='Original Pangu-Weather', linewidth=2, color='indianred')
        plt.plot(steps, fine_rmse, marker='s', label='Finetuned Pangu-Weather', linewidth=2, color='steelblue')
        
        # Determine initialization date (one day before step 1 target date)
        if 1 in step_to_date:
            first_target = datetime.strptime(step_to_date[1], "%b %d")
            # Fake year to do timedelta
            first_target = first_target.replace(year=2026)
            init_date = (first_target - timedelta(days=1)).strftime("%b %d, %Y")
        else:
            init_date = "Unknown"

        plt.title(f"Autoregressive Performance Decay: {var} ({region_name})\nInitialization Date: May 27, 2026", fontsize=14, fontweight='bold')
        plt.xlabel("Forecast Lead Time & Target Date", fontsize=12)
        plt.ylabel("RMSE", fontsize=12)
        plt.xticks(steps, labels, fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        filename = f"autoreg_{region_name.lower()}_{var.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close()

if __name__ == "__main__":
    from datetime import timedelta
    global_file = "raw_metrics/autoregressive_metrics.txt"
    india_file = "raw_metrics/autoregressive_metrics_india.txt"
    
    if os.path.exists(global_file):
        data_global, dates_global = parse_metrics_file(global_file)
        plot_metrics(data_global, dates_global, "plots/autoregressive", "Global")
        print("Generated Global autoregressive plots with dates.")
    
    if os.path.exists(india_file):
        data_india, dates_india = parse_metrics_file(india_file)
        plot_metrics(data_india, dates_india, "plots/autoregressive", "India")
        print("Generated India autoregressive plots with dates.")
