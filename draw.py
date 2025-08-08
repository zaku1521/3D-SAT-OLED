import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.ticker as mticker
import os  # Added for directory creation and path handling

global oled_metrics, oled_per_target, oled_targets, oled_save_path, oled_unit_groups
global qm9_metrics, qm9_per_target, qm9_save_path, qm9_targets, qm9_weights, qm9_unit_groups

# Common configuration for error metrics
error_metrics = ['mae', 'mse', 'rmse', 'median_ae']
error_labels = ['MAE', 'MSE', 'RMSE', 'MedianAE']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def load_metrics(file_path):
    """Load metrics data in JSON format"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def init_oled_draw():
    global oled_metrics, oled_per_target, oled_targets, oled_save_path, oled_unit_groups
    oled_targets = ['plqy', 'e_ad', 'homo', 'lumo']
    oled_metrics_path = os.path.join('./oled_metrics', 'oled_metrics.json')
    os.makedirs(os.path.dirname(oled_metrics_path), exist_ok=True)
    oled_metrics = load_metrics(oled_metrics_path)
    oled_per_target = oled_metrics['per_target']
    oled_save_path = './oled_metrics'
    oled_unit_groups = {
        'Hartree': {
            'units': {'plqy': '-', 'e_ad': 'Hartree', 'homo': 'Hartree', 'lumo': 'Hartree'},
            'mae_scale': 'log'
        },
        'eV': {
            'units': {'plqy': '-', 'e_ad': 'eV', 'homo': 'eV', 'lumo': 'eV'},
            'mae_scale': 'linear'
        }
    }


def init_qm9_draw():
    global qm9_metrics, qm9_per_target, qm9_save_path, qm9_targets, qm9_weights, qm9_unit_groups
    qm9_targets = ['homo', 'lumo', 'gap']
    qm9_weights = [0.25, 0.25, 0.5]
    qm9_metrics_path = os.path.join('./qm9_metrics', 'qm9_metrics.json')
    os.makedirs(os.path.dirname(qm9_metrics_path), exist_ok=True)
    qm9_metrics = load_metrics(qm9_metrics_path)
    qm9_per_target = qm9_metrics['per_target']
    qm9_save_path = './qm9_metrics'  # Directory for saving QM9 plots
    qm9_unit_groups = {
        'Hartree': {
            'units': {'homo': 'Hartree', 'lumo': 'Hartree', 'gap': 'Hartree'},
            'title': 'QM9 Dataset (Unit: Hartree)'
        },
        'eV': {
            'units': {'homo': 'eV', 'lumo': 'eV', 'gap': 'eV'},
            'title': 'QM9 Dataset (Unit: eV)'
        }
    }


def extract_metrics(per_target_dict, target_name, metric, unit):
    """
    Extract data for specified target, metric, and unit from JSON data
    per_target_dict: Target metrics dictionary (e.g., qm9_per_target)
    target_name: Target property name (e.g., 'homo', 'plqy')
    metric: Metric name (e.g., 'mae', 'r2', 'corr')
    unit: Unit (e.g., 'Hartree', 'eV', '-')
    """
    if unit == "-":
        key = f"{target_name}_-"
    else:
        key = f"{target_name}_{unit}"

    if key not in per_target_dict:
        raise KeyError(f"Data not found: Target={target_name}, Unit={unit}, Key={key}")
    return per_target_dict[key][metric]


def plot_metrics_by_unit(dataset_name, per_target_dict, unit_groups, metrics, metric_labels, colors, save_path):
    """Plot comparison of error metrics across different unit groups"""
    for group_name, group_config in unit_groups.items():
        target_units = group_config['units']  # {target: unit}
        targets = list(target_units.keys())  # List of target properties
        title = group_config['title']  # Plot title
        units = [target_units[t] for t in targets]  # List of units

        # Extract data
        data = []
        for metric in metrics:
            metric_data = []
            for target in targets:
                unit = target_units[target]
                metric_data.append(extract_metrics(per_target_dict, target, metric, unit))
            data.append(metric_data)

        # Plot settings
        x = np.arange(len(targets))  # X-axis positions
        width = 0.2  # Bar width
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot bars
        for i in range(len(metrics)):
            ax.bar(
                x + i * width - 1.5 * width,  # Adjust bar positions
                data[i],
                width=width,
                label=metric_labels[i],
                color=colors[i],
                edgecolor='black'
            )

        # Axis and title settings with larger fonts (no bold)
        ax.set_xticks(x)

        ax.set_xticklabels([f'{t} ({u})' for t, u in zip(targets, units)],
                           fontsize=17)
        ax.set_ylabel('Error Value', fontsize=17)
        ax.set_title(title, fontsize=16, pad=15)

        ax.legend(fontsize=16)

        # Use log scale for OLED plots when needed (due to large PLQY error)
        if dataset_name == 'OLED' and group_name in ['mixed_units', 'non_eV']:
            ax.set_yscale('log')
            ax.set_ylabel('Error Value (log scale)', fontsize=17)

        plt.yticks(fontsize=16)

        # Add grid lines
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()

        # Save plot
        plot_filename = f"{dataset_name.lower()}_metrics_{group_name}.png"
        plot_path = os.path.join(save_path, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()


def plot_oled_triple_metrics(oled_per_target, oled_targets, save_path):
    """Plot OLED metrics (correlation, R², MAE) across different unit systems"""

    for system_name, system_config in oled_unit_groups.items():
        target_units = system_config['units']
        mae_scale = system_config['mae_scale']

        # Extract metrics
        corrs, r2s, maes = [], [], []
        for target in oled_targets:
            unit = target_units[target]
            corrs.append(extract_metrics(oled_per_target, target, 'corr', unit))
            r2s.append(extract_metrics(oled_per_target, target, 'r2', unit))
            maes.append(extract_metrics(oled_per_target, target, 'mae', unit))

        x = np.arange(len(oled_targets))

        plt.figure(figsize=(14, 5))
        plt.subplots_adjust(top=0.85)

        # Correlation subplot
        plt.subplot(1, 3, 1)
        plt.bar(x, corrs, color='skyblue')
        plt.xticks(x, oled_targets)
        plt.title('Correlation Coefficient', fontsize=15)
        plt.ylim(0, 1.05)
        plt.ylabel('Correlation', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # R² subplot
        plt.subplot(1, 3, 2)
        plt.bar(x, r2s, color='salmon')
        plt.xticks(x, oled_targets, fontsize=13)
        plt.title('Coefficient of Determination ($R^2$)', fontsize=15)
        plt.ylim(0, 1.05)
        plt.yticks(fontsize=13)
        plt.ylabel('$R^2$', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # MAE subplot
        ax = plt.subplot(1, 3, 3)
        plt.bar(x, maes, color='lightgreen')
        plt.xticks(x, oled_targets, fontsize=13)
        plt.yscale(mae_scale)
        plt.yticks(fontsize=13)

        if mae_scale == 'log':
            min_val, max_val = min(maes), max(maes)
            if min_val <= 0:
                min_val = min([v for v in maes if v > 0]) * 0.5
            lower = 10 ** (int(np.floor(np.log10(min_val))))
            upper = 10 ** (int(np.ceil(np.log10(max_val))))
            ax.set_ylim(lower, upper)

            major_ticks = [3e-3, 8e-3, 3e-2, 8e-2]
            ax.yaxis.set_major_locator(mticker.FixedLocator(major_ticks))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
            ax.yaxis.set_minor_locator(mticker.NullLocator())
            ax.grid(axis='y', which='major', linestyle='--', alpha=0.5)

        scale_label = ' (Log Scale)' if mae_scale == 'log' else ''
        plt.title(f'Mean Absolute Error (MAE){scale_label}', fontsize=15)
        plt.ylabel(f'MAE ({target_units[oled_targets[1]]})', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Overall title
        plt.suptitle(
            'OLED Dataset Performance Metrics',
            fontsize=16,
            y=0.98
        )
        plt.tight_layout(rect=[0, 0, 1, 0.92])

        # Save plot
        plot_filename = f"oled_triple_metrics_{system_name}.png"
        plot_path = os.path.join(save_path, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()


def plot_prediction_scatter(y_valid, y_pred, save_path):
    """Plot scatter plots of predicted vs true values for OLED targets"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (attr, color) in enumerate(zip(oled_targets, colors)):
        ax = axes[i]
        ax.scatter(y_valid[i], y_pred[i], alpha=0.6, color=color, label=attr, s=40)
        ax.plot([y_valid[i].min(), y_valid[i].max()],
                [y_valid[i].min(), y_valid[i].max()],
                'k--')

        unit = '-' if attr == 'plqy' else 'eV'

        corr = extract_metrics(oled_per_target, attr, 'corr', unit)
        r2 = extract_metrics(oled_per_target, attr, 'r2', unit)

        metric_text = (f'Corr={corr:.4f}\n'
                       f'R²={r2:.4f}')

        ax.text(0.65, 0.25, metric_text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=15,
                bbox=dict(facecolor='white', alpha=0.8))

        ax.set_xlabel('True Value', fontsize=18)
        ax.set_ylabel('Predicted Value', fontsize=18)
        ax.set_title(attr, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=15)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(save_path, "oled_prediction_scatter.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_qm9_r2_scores(qm9_per_target, save_path, unit='Hartree'):
    """Plot R² scores and weighted R² for QM9 dataset"""

    # Extract R² scores
    r2_scores = []
    for target in qm9_targets:
        r2 = extract_metrics(qm9_per_target, target, 'r2', unit)
        r2_scores.append(r2)

    # Calculate weighted R²
    weighted_r2 = sum(w * r2 for w, r2 in zip(qm9_weights, r2_scores))
    r2_scores.append(weighted_r2)

    labels = [t.upper() for t in qm9_targets] + ['Weighted $R^2$']

    plt.figure(figsize=(10, 6))

    # Plot line with markers
    plt.plot(labels, r2_scores,
             marker='o',
             markersize=8,
             linewidth=2,
             color='#1f77b4',
             markerfacecolor='white',
             markeredgewidth=2,
             markeredgecolor='#1f77b4')

    # Add value labels
    for x, y in zip(labels, r2_scores):
        plt.text(x, y + 0.0003, f'{y:.4f}',
                 ha='center',
                 va='bottom',
                 fontsize=15,
                 bbox=dict(facecolor='none', alpha=0.8, edgecolor='none', pad=2))

    # Axis limits
    min_r2 = min(r2_scores) - 0.001
    max_r2 = max(r2_scores) + 0.001
    plt.ylim(min_r2, max(max_r2, 1.0005))

    # Labels and title
    plt.ylabel('$R^2$ Value', fontsize=17)
    plt.xlabel('Target Property', fontsize=17)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title(f'QM9 Dataset $R^2$ Scores and Weighted $R^2$', fontsize=18, pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(save_path, f"qm9_r2_scores_{unit.lower()}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_mae_comparison_qm9_meV(per_target_dict, save_path):
    """Plot MAE comparison between different models on QM9 dataset"""
    # Model names and their MAE values (meV)
    models = ['LiNet', 'TFN', 'SE(3)-Tr', 'LieConv', 'EGNN', 'SpinConv']
    homo_meV = [46, 40, 35, 30, 29, 26]
    lumo_meV = [35, 38, 33, 25, 25, 22]
    gap_meV = [68, 58, 53, 49, 48, 47]

    # Add current model's results
    current_model = '3D-SAT-OLED'
    homo_meV.append(extract_metrics(per_target_dict, 'homo', 'mae', 'eV') * 1000)
    lumo_meV.append(extract_metrics(per_target_dict, 'lumo', 'mae', 'eV') * 1000)
    gap_meV.append(extract_metrics(per_target_dict, 'gap', 'mae', 'eV') * 1000)
    models.append(current_model)

    plt.figure(figsize=(12, 6))

    # Plot lines
    plt.plot(models, homo_meV, marker='o', label='HOMO', linewidth=2, markersize=8)
    plt.plot(models, lumo_meV, marker='s', label='LUMO', linewidth=2, markersize=8)
    plt.plot(models, gap_meV, marker='^', label='GAP', linewidth=2, markersize=8)

    # Labels and title
    plt.title('Model Performance Comparison on QM9 (MAE in meV)', fontsize=17, pad=15)
    plt.ylabel('MAE (meV)', fontsize=17)
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(fontsize=16)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(save_path, "qm9_model_mae_comparison_meV.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_mae_comparison_qm9_Ha(per_target_dict, save_path):
    """Plot MAE comparison between different models on QM9 dataset using Hartree units"""
    # Model names and their corresponding MAE values (in Hartree)
    models = ['GC*', 'MPNN*', 'DTNN*', 'Attentive FP', 'PointGAT']
    homo_hartree = [0.00716, 0.00541, 0.00388, 0.00358, 0.00335]
    lumo_hartree = [0.00921, 0.00623, 0.00513, 0.00415, 0.00370]
    gap_hartree = [0.01120, 0.00820, 0.00660, 0.00528, 0.00486]

    # Add results for the current model (extracted from per_target_dict)
    current_model = '3D-SAT-OLED'
    # Assuming the extract_metrics function returns values in Hartree units
    homo_hartree.append(extract_metrics(per_target_dict, 'homo', 'mae', 'Hartree'))
    lumo_hartree.append(extract_metrics(per_target_dict, 'lumo', 'mae', 'Hartree'))
    gap_hartree.append(extract_metrics(per_target_dict, 'gap', 'mae', 'Hartree'))
    models.append(current_model)

    # Create figure and set size
    plt.figure(figsize=(12, 6))

    # Plot lines with markers
    plt.plot(models, homo_hartree, marker='o', label='HOMO', linewidth=2, markersize=8)
    plt.plot(models, lumo_hartree, marker='s', label='LUMO', linewidth=2, markersize=8)
    plt.plot(models, gap_hartree, marker='^', label='GAP', linewidth=2, markersize=8)

    # Set labels and title
    plt.title('Model Performance Comparison on QM9 (MAE in Hartree)', fontsize=17, pad=15)
    plt.ylabel('MAE (Hartree)', fontsize=17)
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(fontsize=16)

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save and display the plot
    plot_path = os.path.join(save_path, "qm9_model_mae_comparison_Hartree.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_mae_radar_oled_eV(per_target_dict, save_path):
    models = ['Deep Potential', 'E3NN', 'Uni-Mol', '3D-SAT-OLED']
    labels = ['PLQY', 'E_ad', 'HOMO', 'LUMO']

    values = np.array([
        [0.145, 0.205, 0.084, 0.294],  # Deep Potential
        [0.117, 0.061, 0.085, 0.119],  # E3NN
        [0.107, 0.054, 0.084, 0.111],  # Uni-Mol
        [
            extract_metrics(per_target_dict, 'plqy', 'mae', '-'),
            extract_metrics(per_target_dict, 'e_ad', 'mae', 'eV'),
            extract_metrics(per_target_dict, 'homo', 'mae', 'eV'),
            extract_metrics(per_target_dict, 'lumo', 'mae', 'eV'),
        ]  # 3D-SAT-OLED
    ])

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)

    for i, model in enumerate(models):
        stats = values[i].tolist()
        stats += stats[:1]
        ax.plot(angles, stats, linewidth=2, label=model)
        ax.fill(angles, stats, alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=16)
    ax.xaxis.set_tick_params(pad=15)


    ax.tick_params(axis='y', labelsize=14)

    ax.set_title('Model Performance Comparison on OLED QM (MAE in eV)', size=18, pad=20)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=16)

    plt.tight_layout()
    plot_path = os.path.join(save_path, "oled_model_mae_comparison_eV.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()


def draw_oled(y_valid, y_pred):
    """Generate and save all OLED dataset plots"""
    # Create save directory if it doesn't exist
    os.makedirs(oled_save_path, exist_ok=True)
    # Generate plots
    plot_oled_triple_metrics(oled_per_target, oled_targets, oled_save_path)
    plot_prediction_scatter(y_valid, y_pred, oled_save_path)


def draw_oled_without_scatter():
    """Generate and save all OLED dataset plots"""
    # Create save directory if it doesn't exist
    os.makedirs(oled_save_path, exist_ok=True)
    # Generate plots
    plot_oled_triple_metrics(oled_per_target, oled_targets, oled_save_path)


def draw_qm():
    """Generate and save all QM9 dataset plots"""
    # Create save directory if it doesn't exist
    os.makedirs(qm9_save_path, exist_ok=True)
    # Generate plots
    plot_metrics_by_unit(dataset_name='QM9',
                         per_target_dict=qm9_per_target,
                         unit_groups=qm9_unit_groups,
                         metrics=error_metrics,
                         metric_labels=error_labels,
                         colors=colors,
                         save_path=qm9_save_path)
    plot_qm9_r2_scores(qm9_per_target, qm9_save_path)
    plot_model_mae_comparison_qm9_meV(qm9_per_target, qm9_save_path)
    plot_model_mae_comparison_qm9_Ha(qm9_per_target, qm9_save_path)


