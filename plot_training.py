# Copyright 2025 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import re
import pandas as pd
import matplotlib
# Use the 'Agg' backend to prevent the plot from trying to open a window
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import os 

# Configuration
log_file_path = 'k8s/verl_demo.log'
output_dir = 'results'
metrics_to_plot = [
    'critic/score/mean',       # Accuracy (North Star)
    #'actor/ppo_kl',            # Training Stability
   # 'response_length/mean',    # Reasoning Depth
   # 'perf/mfu/actor',          # Hardware efficiency for the actor 
    'perf/throughput',         # Throughput 
   'perf/time_per_step'       # Step time 
    #'actor/pg_loss' 

]

def parse_verl_logs(file_path):
    data = []
    line_pattern = re.compile(r"step:(\d+)\s-\s(.+)")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: Log file not found at {file_path}")
        return pd.DataFrame()

    with open(file_path, 'r') as f:
        for line in f:
            clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            match = line_pattern.search(clean_line)
            if match:
                step_num = int(match.group(1))
                metrics_raw = match.group(2)
                step_dict = {'step': step_num}
                kv_pairs = re.findall(r"([\w\-/]+):([-+]?\d*\.\d+[eE][-+]?\d+|[-+]?\d*\.\d+|\d+)", metrics_raw)
                for key, value in kv_pairs:
                    step_dict[key] = float(value)
                data.append(step_dict)
    
    return pd.DataFrame(data).drop_duplicates(subset=['step']).sort_values('step')

# 1. Parse Data
df = parse_verl_logs(log_file_path)

if df.empty:
    print("❌ No data found in logs. Check your log file path or regex.")
else:
    # 2. Setup Visualization
    sns.set_theme(style="whitegrid")
    
    # Dynamically calculate grid size
    num_metrics = len(metrics_to_plot)
    cols = 3  # Set how many plots you want per row
    rows = (num_metrics + cols - 1) // cols  # Ceiling division to find needed rows

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    fig.suptitle(f"VERL Training Metrics", fontsize=20)
    
    # Flatten axes for easy iteration, even if it's a 1D or 2D array
    axes_flat = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        ax = axes_flat[i]
        if metric in df.columns:
            sns.lineplot(ax=ax, x='step', y=metric, data=df, color='royalblue', linewidth=2)
            ax.set_title(f"Change in {metric}", fontsize=14)
        else:
            ax.text(0.5, 0.5, f"Metric '{metric}'\nnot found", 
                    ha='center', va='center', fontsize=12, color='gray')
    
    # Hide any unused subplot placeholders
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 3. Save Logic (MUST come before plt.show, or just replace it)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_path = os.path.join(output_dir, 'verl_training_dashboard.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    csv_path = os.path.join(output_dir, 'parsed_metrics.csv')
    df.to_csv(csv_path, index=False)

    print(f"✅ Dashboard saved to: {plot_path}")
    print(f"✅ Raw metrics saved to: {csv_path}")

    # Summary Stats
    print("\n" + "="*50)
    print(f"{'TRAINING SUMMARY (Skipping Step 1)':^50}")
    print("="*50)
    
    # Slice to ignore the cold-start step
    df_stable = df.iloc[1:] if len(df) > 1 else df

    for metric in metrics_to_plot:
        if metric in df_stable.columns:
            avg_val = df_stable[metric].mean()
            
            # Metric-specific formatting
            if metric == 'critic/score/mean':
                final_val = df_stable[metric].iloc[-1]
                print(f"{metric:<22} | Avg: {avg_val:10.4f} | Final: {final_val:10.4f}")
            
            # Higher precision for hardware efficiency (MFU)
            elif 'mfu' in metric:
                print(f"{metric:<22} | Avg: {avg_val:10.6f}") # 6 decimals for MFU
            
            else:
                # Standard 2 decimals for throughput/time, 4 for others
                fmt = ".2f" if "perf" in metric or "throughput" in metric else ".4f"
                print(f"{metric:<22} | Avg: {avg_val:{fmt}}")
        else:
            print(f"{metric:<22} | Metric not found")
    print("="*50)