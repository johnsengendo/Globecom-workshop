# Importing necessary libraries for data manipulation, modeling, and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Defining the base directory path where the dataset is stored
base_dir = "./network_operator_KPIs_time_series_dataset/network_operator_KPIs_time_series_dataset"

# Loading metadata containing information about the time series data
info_path = f"{base_dir}/data_series_info.txt"
info_df = pd.read_csv(info_path, sep=" ", header=None, names=["series_id", "kpi_type"])

# Defining the types of KPIs to be analyzing and mapping each KPI to its corresponding file path
kpi_types = ['internet', 'sessions', 'vpn', 'downstream']
kpi_series_paths = {}
for kpi in kpi_types:
    # Extracting the series_id for each KPI type from the metadata
    kpi_id_row = info_df[info_df["kpi_type"] == kpi].iloc[0]["series_id"]
    kpi_series_paths[kpi] = f"{base_dir}/data_series/{kpi_id_row}.txt"

# Loading and merging the KPI data from different sources into a single DataFrame
combined_df = None
for kpi, path in kpi_series_paths.items():
    # Reading each KPI file and merging it with the combined DataFrame
    df = pd.read_csv(path, sep="\s+", header=None, names=["timestamp", kpi])
    combined_df = df if combined_df is None else combined_df.merge(df, on='timestamp', how='inner')

# Defining the window size for creating sequences of time series data
window_size = 10

# Preparing the dataset by dropping the timestamp column and converting it to a numpy array
data = combined_df.drop(columns=['timestamp']).values

def create_multivariate_dataset(data, window_size):
    """
    Creating sequences of data for time series prediction.

    Parameters:
    - data: The input time series data.
    - window_size: The number of time steps to include in each sequence.

    Returns:
    - X: Input sequences for the model.
    - y: Target values for prediction.
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size][0])  # Predicting the 'internet' KPI
    return np.array(X), np.array(y)

# Generating input sequences and target values for the model
X, y = create_multivariate_dataset(data, window_size)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Building an LSTM model with bidirectional layers for improved sequence learning
model = Sequential([
    # Adding first bidirectional LSTM layer with batch normalization and dropout for regularization
    Bidirectional(LSTM(128, return_sequences=True, input_shape=(window_size, X.shape[2]))),
    BatchNormalization(),
    Dropout(0.2),

    # Adding second bidirectional LSTM layer
    Bidirectional(LSTM(128, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.2),

    # Adding third bidirectional LSTM layer (final recurrent layer)
    Bidirectional(LSTM(64)),
    BatchNormalization(),
    Dropout(0.2),

    # Adding dense layers for final prediction
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

# Configuring the optimizer and compiling the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

# Setting up training callbacks for better performance and early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Training the model with validation split
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Making predictions on the test data
y_pred = model.predict(X_test).flatten()
actual = y_test

# Creating baseline prediction methods for comparison
# Creating static baseline: mean + 2 standard deviations
static_bw = y_train.mean() + 2 * y_train.std()
static_pred = np.full_like(actual, static_bw)

# Creating percentile-based baseline: 95th percentile
p95_val = np.percentile(y_train, 95)
p95_pred = np.full_like(actual, p95_val)

def compute_errors(actual, predicted):
    """
    Computing basic error metrics for model evaluation.

    Parameters:
    - actual: The actual values.
    - predicted: The predicted values.

    Returns:
    - mse: Mean Squared Error.
    - rmse: Root Mean Squared Error.
    - mae: Mean Absolute Error.
    """
    mse = np.mean((predicted - actual) ** 2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    return mse, rmse, mae

# Calculating error metrics for all approaches
mse_dt, rmse_dt, mae_dt = compute_errors(actual, y_pred)
mse_static, rmse_static, mae_static = compute_errors(actual, static_pred)
mse_p95, rmse_p95, mae_p95 = compute_errors(actual, p95_pred)

print(f"MAE - Static: {mae_static:.2f}, P95: {mae_p95:.2f}, DT: {mae_dt:.2f}")
print(f"RMSE - Static: {rmse_static:.2f}, P95: {rmse_p95:.2f}, DT: {rmse_dt:.2f}")

def compute_metrics(actual, allocated):
    """
    Computing efficiency, wastage, and utilization metrics.

    Parameters:
    - actual: The actual values.
    - allocated: The allocated or predicted values.

    Returns:
    - A dictionary containing efficiency, wastage, and utilization metrics.
    """
    allocated = np.clip(allocated, 1e-6, None)
    efficiency = np.minimum(actual / allocated, 1.0)
    wastage = np.maximum(allocated - actual, 0) / allocated
    utilization = actual / allocated
    return {
        "Efficiency": np.mean(efficiency),
        "Wastage": np.mean(wastage),
        "Utilization": np.mean(utilization)
    }

# Calculating advanced metrics for all methods
metrics_static = compute_metrics(actual, static_pred)
metrics_p95 = compute_metrics(actual, p95_pred)
metrics_dt = compute_metrics(actual, y_pred)

print("Advanced Metrics (Static):", metrics_static)
print("Advanced Metrics (P95):", metrics_p95)
print("Advanced Metrics (DT):", metrics_dt)

# Computing over-provisioning analysis
over_static = np.maximum(static_pred - actual, 0)
over_p95 = np.maximum(p95_pred - actual, 0)
over_dt = np.maximum(y_pred - actual, 0)
avg_over_static = np.mean(over_static)
avg_over_p95 = np.mean(over_p95)
avg_over_dt = np.mean(over_dt)

print(f"Avg Over-provision - Static: {avg_over_static:.2f}, P95: {avg_over_p95:.2f}, DT: {avg_over_dt:.2f}")

def compute_comprehensive_metrics(actual, predicted):
    """
    Computing comprehensive metrics for model evaluation.

    Parameters:
    - actual: The actual values.
    - predicted: The predicted values.

    Returns:
    - A dictionary containing various performance metrics.
    """
    mse = np.mean((predicted - actual) ** 2)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1e-6))) * 100
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    nrmse = rmse / (np.max(actual) - np.min(actual)) * 100
    max_error = np.max(np.abs(actual - predicted))
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2,
        'NRMSE (%)': nrmse,
        'Max Error': max_error
    }

# Calculating comprehensive metrics for all methods
static_metrics = compute_comprehensive_metrics(actual, static_pred)
p95_metrics = compute_comprehensive_metrics(actual, p95_pred)
dt_metrics = compute_comprehensive_metrics(actual, y_pred)

# Creating and displaying the performance table
performance_data = {
    'Metric': ['MSE', 'RMSE', 'MAE', 'MAPE (%)', 'R²', 'NRMSE (%)', 'Max Error'],
    'Static Method': [
        f"{static_metrics['MSE']:.4f}",
        f"{static_metrics['RMSE']:.4f}",
        f"{static_metrics['MAE']:.4f}",
        f"{static_metrics['MAPE']:.2f}",
        f"{static_metrics['R²']:.4f}",
        f"{static_metrics['NRMSE (%)']:.2f}",
        f"{static_metrics['Max Error']:.4f}"
    ],
    'P95 Method': [
        f"{p95_metrics['MSE']:.4f}",
        f"{p95_metrics['RMSE']:.4f}",
        f"{p95_metrics['MAE']:.4f}",
        f"{p95_metrics['MAPE']:.2f}",
        f"{p95_metrics['R²']:.4f}",
        f"{p95_metrics['NRMSE (%)']:.2f}",
        f"{p95_metrics['Max Error']:.4f}"
    ],
    'DT Method': [
        f"{dt_metrics['MSE']:.4f}",
        f"{dt_metrics['RMSE']:.4f}",
        f"{dt_metrics['MAE']:.4f}",
        f"{dt_metrics['MAPE']:.2f}",
        f"{dt_metrics['R²']:.4f}",
        f"{dt_metrics['NRMSE (%)']:.2f}",
        f"{dt_metrics['Max Error']:.4f}"
    ],
    'Best Method': []
}

# Determining the best method for each metric
metric_keys = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²', 'NRMSE (%)', 'Max Error']
best_methods = []
for metric in metric_keys:
    static_val = static_metrics[metric]
    p95_val = p95_metrics[metric]
    dt_val = dt_metrics[metric]

    if metric == 'R²':
        best_val = max(static_val, p95_val, dt_val)
        if best_val == static_val:
            best_methods.append('Static')
        elif best_val == p95_val:
            best_methods.append('P95')
        else:
            best_methods.append('DT')
    else:
        best_val = min(static_val, p95_val, dt_val)
        if best_val == static_val:
            best_methods.append('Static')
        elif best_val == p95_val:
            best_methods.append('P95')
        else:
            best_methods.append('DT')

performance_data['Best Method'] = best_methods
performance_df = pd.DataFrame(performance_data)
print(performance_df.to_string(index=False, justify='center'))

def compute_business_metrics(actual, allocated):
    """
    Computing business and efficiency metrics for resource allocation.

    Parameters:
    - actual: The actual values.
    - allocated: The allocated or predicted values.

    Returns:
    - A dictionary containing various business and efficiency metrics.
    """
    allocated = np.clip(allocated, 1e-6, None)
    efficiency = np.minimum(actual / allocated, 1.0)
    wastage = np.maximum(allocated - actual, 0) / allocated
    utilization = actual / allocated
    under_provision = np.maximum(actual - allocated, 0)
    under_provision_rate = np.mean(under_provision > 0) * 100
    avg_under_provision = np.mean(under_provision)
    over_provision = np.maximum(allocated - actual, 0)
    over_provision_rate = np.mean(over_provision > 0) * 100
    avg_over_provision = np.mean(over_provision)
    sla_violations = np.sum(actual > allocated)
    sla_compliance = (1 - sla_violations / len(actual)) * 100
    total_allocated = np.sum(allocated)
    total_actual = np.sum(actual)
    cost_efficiency = (total_actual / total_allocated) * 100
    return {
        'Avg Efficiency': np.mean(efficiency),
        'Avg Wastage (%)': np.mean(wastage) * 100,
        'Avg Utilization (%)': np.mean(utilization) * 100,
        'Under-prov Rate (%)': under_provision_rate,
        'Avg Under-prov': avg_under_provision,
        'Over-prov Rate (%)': over_provision_rate,
        'Avg Over-prov': avg_over_provision,
        'SLA Compliance (%)': sla_compliance,
        'Cost Efficiency (%)': cost_efficiency
    }

# Calculating business metrics for all methods
static_business = compute_business_metrics(actual, static_pred)
p95_business = compute_business_metrics(actual, p95_pred)
dt_business = compute_business_metrics(actual, y_pred)

# Creating business impact table
business_data = {
    'Metric': [
        'Average Efficiency',
        'Average Wastage (%)',
        'Average Utilization (%)',
        'Under-provisioning Rate (%)',
        'Avg Under-provisioning',
        'Over-provisioning Rate (%)',
        'Avg Over-provisioning',
        'SLA Compliance (%)',
        'Cost Efficiency (%)'
    ],
    'Static Method': [
        f"{static_business['Avg Efficiency']:.4f}",
        f"{static_business['Avg Wastage (%)']:.2f}",
        f"{static_business['Avg Utilization (%)']:.2f}",
        f"{static_business['Under-prov Rate (%)']:.2f}",
        f"{static_business['Avg Under-prov']:.4f}",
        f"{static_business['Over-prov Rate (%)']:.2f}",
        f"{static_business['Avg Over-prov']:.4f}",
        f"{static_business['SLA Compliance (%)']:.2f}",
        f"{static_business['Cost Efficiency (%)']:.2f}"
    ],
    'P95 Method': [
        f"{p95_business['Avg Efficiency']:.4f}",
        f"{p95_business['Avg Wastage (%)']:.2f}",
        f"{p95_business['Avg Utilization (%)']:.2f}",
        f"{p95_business['Under-prov Rate (%)']:.2f}",
        f"{p95_business['Avg Under-prov']:.4f}",
        f"{p95_business['Over-prov Rate (%)']:.2f}",
        f"{p95_business['Avg Over-prov']:.4f}",
        f"{p95_business['SLA Compliance (%)']:.2f}",
        f"{p95_business['Cost Efficiency (%)']:.2f}"
    ],
    'DT Method': [
        f"{dt_business['Avg Efficiency']:.4f}",
        f"{dt_business['Avg Wastage (%)']:.2f}",
        f"{dt_business['Avg Utilization (%)']:.2f}",
        f"{dt_business['Under-prov Rate (%)']:.2f}",
        f"{dt_business['Avg Under-prov']:.4f}",
        f"{dt_business['Over-prov Rate (%)']:.2f}",
        f"{dt_business['Avg Over-prov']:.4f}",
        f"{dt_business['SLA Compliance (%)']:.2f}",
        f"{dt_business['Cost Efficiency (%)']:.2f}"
    ],
    'Best Method': []
}

# Determining the best method for business metrics
business_metrics_keys = ['Avg Efficiency', 'Avg Wastage (%)', 'Avg Utilization (%)', 'Under-prov Rate (%)',
                        'Avg Under-prov', 'Over-prov Rate (%)', 'Avg Over-prov', 'SLA Compliance (%)', 'Cost Efficiency (%)']
best_business_methods = []
for metric in business_metrics_keys:
    static_val = static_business[metric]
    p95_val = p95_business[metric]
    dt_val = dt_business[metric]

    if metric in ['Avg Efficiency', 'Avg Utilization (%)', 'SLA Compliance (%)', 'Cost Efficiency (%)']:
        best_val = max(static_val, p95_val, dt_val)
        if best_val == static_val:
            best_business_methods.append('Static')
        elif best_val == p95_val:
            best_business_methods.append('P95')
        else:
            best_business_methods.append('DT')
    else:
        best_val = min(static_val, p95_val, dt_val)
        if best_val == static_val:
            best_business_methods.append('Static')
        elif best_val == p95_val:
            best_business_methods.append('P95')
        else:
            best_business_methods.append('DT')

business_data['Best Method'] = best_business_methods
business_df = pd.DataFrame(business_data)
print(business_df.to_string(index=False, justify='center'))

# Printing summary insights
print("\n" + "="*95)
print("KEY INSIGHTS:")
print("="*95)
print(f"🎯 Best MAE: {min(mae_static, mae_p95, mae_dt):.2f} achieved by {['Static', 'P95', 'DT'][np.argmin([mae_static, mae_p95, mae_dt])]}")
print(f"💰 Lowest Wastage: {min(static_business['Avg Wastage (%)'], p95_business['Avg Wastage (%)'], dt_business['Avg Wastage (%)']):.2f}% achieved by {['Static', 'P95', 'DT'][np.argmin([static_business['Avg Wastage (%)'], p95_business['Avg Wastage (%)'], dt_business['Avg Wastage (%)']])]}")
print(f"📊 Best SLA Compliance: {max(static_business['SLA Compliance (%)'], p95_business['SLA Compliance (%)'], dt_business['SLA Compliance (%)']):.2f}% achieved by {['Static', 'P95', 'DT'][np.argmax([static_business['SLA Compliance (%)'], p95_business['SLA Compliance (%)'], dt_business['SLA Compliance (%)']])]}")
print(f"⚡ Best Cost Efficiency: {max(static_business['Cost Efficiency (%)'], p95_business['Cost Efficiency (%)'], dt_business['Cost Efficiency (%)']):.2f}% achieved by {['Static', 'P95', 'DT'][np.argmax([static_business['Cost Efficiency (%)'], p95_business['Cost Efficiency (%)'], dt_business['Cost Efficiency (%)']])]}")
print("="*95)

# Setting global parameters for ultra-high definition and enhanced styling of visualizations
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 8
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['grid.linewidth'] = 1.5
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.2

# Plotting actual vs predicted values
n_plot = 1000
plt.figure(figsize=(16, 8))
plt.plot(actual[:n_plot], label='Actual Internet', linewidth=4, color='green', alpha=0.9)
plt.plot(y_pred[:n_plot], label='DT Prediction', linewidth=4, color='orange', alpha=0.9)
plt.xlabel('Time Index', fontsize=18, fontweight='bold')
plt.ylabel('KPI Value', fontsize=18, fontweight='bold')
plt.title('Actual vs. Predicted KPI', fontsize=22, fontweight='bold', pad=20)
plt.legend(fontsize=16, loc='upper right', frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.4, linewidth=1.5)
plt.tight_layout()
plt.savefig('actual_vs_predicted_all_methods.png', dpi=600, bbox_inches='tight')
plt.close()

# Plotting MAE & RMSE comparison
methods = ['Baseline 1', 'Baseline 2', 'AI-Enabled-DT']
mae_vals = [mae_static, mae_p95, mae_dt]
rmse_vals = [rmse_static, rmse_p95, rmse_dt]
x = np.arange(len(methods))
width = 0.35
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
bars1 = ax1.bar(x, mae_vals, color=['blue', 'orange', 'green'], alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_xlabel('Methods', fontsize=18, fontweight='bold')
ax1.set_ylabel('MAE', fontsize=18, fontweight='bold')
ax1.set_title('Mean Absolute Error Comparison', fontsize=20, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.4, linewidth=1.5)
bars2 = ax2.bar(x, rmse_vals, color=['blue', 'orange', 'green'], alpha=0.8, edgecolor='black', linewidth=2)
ax2.set_xlabel('Methods', fontsize=18, fontweight='bold')
ax2.set_ylabel('RMSE', fontsize=18, fontweight='bold')
ax2.set_title('Root Mean Square Error Comparison', fontsize=20, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=16, fontweight='bold')
ax2.grid(True, alpha=0.4, linewidth=1.5)
plt.tight_layout()
plt.savefig('mae_rmse_comparison_all_methods.png', dpi=600, bbox_inches='tight')
plt.close()

# Plotting advanced metrics comparison
adv_labels = ['Efficiency', 'Wastage', 'Utilization']
static_adv = [metrics_static['Efficiency'], metrics_static['Wastage'], metrics_static['Utilization']]
p95_adv = [metrics_p95['Efficiency'], metrics_p95['Wastage'], metrics_p95['Utilization']]
dt_adv = [metrics_dt['Efficiency'], metrics_dt['Wastage'], metrics_dt['Utilization']]
x = np.arange(len(adv_labels))
width = 0.25
plt.figure(figsize=(16, 8))
bars1 = plt.bar(x - width, static_adv, width, label='Baseline 1', color='blue', alpha=0.8, edgecolor='black', linewidth=2)
bars2 = plt.bar(x, p95_adv, width, label='Baseline 2', color='orange', alpha=0.8, edgecolor='black', linewidth=2)
bars3 = plt.bar(x + width, dt_adv, width, label='AI-Enabled-DT', color='green', alpha=0.8, edgecolor='black', linewidth=2)
plt.xlabel('Metrics', fontsize=18, fontweight='bold')
plt.ylabel('Value', fontsize=18, fontweight='bold')
plt.title('Advanced Metrics Comparison', fontsize=22, fontweight='bold', pad=20)
plt.xticks(x, adv_labels, fontsize=16, fontweight='bold')
plt.legend(fontsize=16, loc='upper center', frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.4, linewidth=1.5)
plt.tight_layout()
plt.savefig('advanced_metrics_comparison_all_methods.png', dpi=600, bbox_inches='tight')
plt.close()

# Plotting over-provisioning comparison
over_vals = [avg_over_static, avg_over_p95, avg_over_dt]
colors = ['blue', 'orange', 'green']
plt.figure(figsize=(14, 8))
bars = plt.bar(methods, over_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
plt.xlabel('Methods', fontsize=18, fontweight='bold')
plt.ylabel('Avg Over-provisioning (KPI Units)', fontsize=18, fontweight='bold')
plt.title('Average Over-provisioning Comparison', fontsize=22, fontweight='bold', pad=20)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.4, linewidth=1.5)
plt.tight_layout()
plt.savefig('over_provisioning_comparison_all_methods.png', dpi=600, bbox_inches='tight')
plt.close()

# Plotting comprehensive performance radar chart
plt.figure(figsize=(14, 14))

def normalize_for_radar(values, higher_is_better=True):
    """
    Normalizing values for radar chart visualization.

    Parameters:
    - values: The values to normalize.
    - higher_is_better: Whether higher values are better.

    Returns:
    - Normalized values.
    """
    if higher_is_better:
        return [(v - min(values)) / (max(values) - min(values)) for v in values]
    else:
        return [(max(values) - v) / (max(values) - min(values)) for v in values]

radar_metrics = ['MAE', 'RMSE', 'Efficiency', 'SLA Compliance', 'Cost Efficiency']
static_radar = [
    static_metrics['MAE'], static_metrics['RMSE'], static_business['Avg Efficiency'],
    static_business['SLA Compliance (%)'], static_business['Cost Efficiency (%)']
]
p95_radar = [
    p95_metrics['MAE'], p95_metrics['RMSE'], p95_business['Avg Efficiency'],
    p95_business['SLA Compliance (%)'], p95_business['Cost Efficiency (%)']
]
dt_radar = [
    dt_metrics['MAE'], dt_metrics['RMSE'], dt_business['Avg Efficiency'],
    dt_business['SLA Compliance (%)'], dt_business['Cost Efficiency (%)']
]

mae_norm = normalize_for_radar([static_radar[0], p95_radar[0], dt_radar[0]], False)
rmse_norm = normalize_for_radar([static_radar[1], p95_radar[1], dt_radar[1]], False)
eff_norm = normalize_for_radar([static_radar[2], p95_radar[2], dt_radar[2]], True)
sla_norm = normalize_for_radar([static_radar[3], p95_radar[3], dt_radar[3]], True)
cost_norm = normalize_for_radar([static_radar[4], p95_radar[4], dt_radar[4]], True)

static_norm = [mae_norm[0], rmse_norm[0], eff_norm[0], sla_norm[0], cost_norm[0]]
p95_norm = [mae_norm[1], rmse_norm[1], eff_norm[1], sla_norm[1], cost_norm[1]]
dt_norm = [mae_norm[2], rmse_norm[2], eff_norm[2], sla_norm[2], cost_norm[2]]

from math import pi
angles = [n / float(len(radar_metrics)) * 2 * pi for n in range(len(radar_metrics))]
angles += angles[:1]
static_norm += static_norm[:1]
p95_norm += p95_norm[:1]
dt_norm += dt_norm[:1]
fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))
ax.plot(angles, static_norm, 'o-', linewidth=4, label='Baseline 1', color='blue', markersize=10)
ax.fill(angles, static_norm, alpha=0.25, color='blue')
ax.plot(angles, p95_norm, 'o-', linewidth=4, label='Baseline 2', color='orange', markersize=10)
ax.fill(angles, p95_norm, alpha=0.25, color='orange')
ax.plot(angles, dt_norm, 'o-', linewidth=4, label='AI-Enabled-DT', color='green', markersize=10)
ax.fill(angles, dt_norm, alpha=0.25, color='green')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_metrics, fontsize=16, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12, fontweight='bold')
ax.set_title('Comprehensive Performance Comparison',
             fontsize=20, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), fontsize=16, frameon=True,
          fancybox=True, shadow=True)
ax.grid(True, linewidth=2, alpha=0.6)
ax.tick_params(axis='y', labelsize=12)
plt.tight_layout()
plt.savefig('performance_radar_chart.png', dpi=600, bbox_inches='tight')
plt.close()

# Plotting box plot for error distribution comparison
error_static = np.abs(actual - static_pred)
error_p95 = np.abs(actual - p95_pred)
error_dt = np.abs(actual - y_pred)
error_data = [error_static, error_p95, error_dt]
method_labels = ['Baseline 1', 'Baseline 2', 'AI-Enabled-DT']
plt.figure(figsize=(14, 10))
box_plot = plt.boxplot(error_data, labels=method_labels, patch_artist=True,
                       boxprops=dict(linewidth=3), whiskerprops=dict(linewidth=3),
                       capprops=dict(linewidth=3), medianprops=dict(linewidth=4))
colors = ['lightblue', 'lightcoral', 'lightgreen']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
plt.xlabel('Methods', fontsize=18, fontweight='bold')
plt.ylabel('Absolute Prediction Error', fontsize=18, fontweight='bold')
plt.title('Prediction Error Distribution Comparison', fontsize=22, fontweight='bold', pad=20)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.4, linewidth=1.5, axis='y')
median_static = np.median(error_static)
median_p95 = np.median(error_p95)
median_dt = np.median(error_dt)
plt.text(1, median_static + 0.1, f'Median: {median_static:.2f}',
         ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.text(2, median_p95 + 0.1, f'Median: {median_p95:.2f}',
         ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.text(3, median_dt + 0.1, f'Median: {median_dt:.2f}',
         ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('error_distribution_boxplot.png', dpi=600, bbox_inches='tight')
plt.close()

# Plotting box plot for relative error comparison
relative_error_static = np.abs((actual - static_pred) / np.maximum(actual, 1e-6)) * 100
relative_error_p95 = np.abs((actual - p95_pred) / np.maximum(actual, 1e-6)) * 100
relative_error_dt = np.abs((actual - y_pred) / np.maximum(actual, 1e-6)) * 100
relative_error_static = np.clip(relative_error_static, 0, 200)
relative_error_p95 = np.clip(relative_error_p95, 0, 200)
relative_error_dt = np.clip(relative_error_dt, 0, 200)
relative_error_data = [relative_error_static, relative_error_p95, relative_error_dt]
plt.figure(figsize=(14, 10))
box_plot_rel = plt.boxplot(relative_error_data, labels=method_labels, patch_artist=True,
                          boxprops=dict(linewidth=3), whiskerprops=dict(linewidth=3),
                          capprops=dict(linewidth=3), medianprops=dict(linewidth=4))
for patch, color in zip(box_plot_rel['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
plt.xlabel('Methods', fontsize=18, fontweight='bold')
plt.ylabel('Relative Prediction Error (%)', fontsize=18, fontweight='bold')
plt.title('Relative Error Distribution Comparison', fontsize=22, fontweight='bold', pad=20)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.4, linewidth=1.5, axis='y')
median_rel_static = np.median(relative_error_static)
median_rel_p95 = np.median(relative_error_p95)
median_rel_dt = np.median(relative_error_dt)
plt.text(1, median_rel_static + 2, f'Median: {median_rel_static:.1f}%',
         ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.text(2, median_rel_p95 + 2, f'Median: {median_rel_p95:.1f}%',
         ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.text(3, median_rel_dt + 2, f'Median: {median_rel_dt:.1f}%',
         ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('relative_error_boxplot.png', dpi=600, bbox_inches='tight')
plt.close()

# Plotting box plot for resource efficiency comparison
efficiency_static = np.minimum(actual / np.maximum(static_pred, 1e-6), 1.0)
efficiency_p95 = np.minimum(actual / np.maximum(p95_pred, 1e-6), 1.0)
efficiency_dt = np.minimum(actual / np.maximum(y_pred, 1e-6), 1.0)
efficiency_data = [efficiency_static, efficiency_p95, efficiency_dt]
plt.figure(figsize=(14, 10))
box_plot_eff = plt.boxplot(efficiency_data, labels=method_labels, patch_artist=True,
                          boxprops=dict(linewidth=3), whiskerprops=dict(linewidth=3),
                          capprops=dict(linewidth=3), medianprops=dict(linewidth=4))
for patch, color in zip(box_plot_eff['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
plt.xlabel('Methods', fontsize=18, fontweight='bold')
plt.ylabel('Resource Efficiency', fontsize=18, fontweight='bold')
plt.title('Resource Efficiency Distribution Comparison', fontsize=22, fontweight='bold', pad=20)
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.4, linewidth=1.5, axis='y')
median_eff_static = np.median(efficiency_static)
median_eff_p95 = np.median(efficiency_p95)
median_eff_dt = np.median(efficiency_dt)
plt.text(1, median_eff_static + 0.02, f'Median: {median_eff_static:.3f}',
         ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.text(2, median_eff_p95 + 0.02, f'Median: {median_eff_p95:.3f}',
         ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.text(3, median_eff_dt + 0.02, f'Median: {median_eff_dt:.3f}',
         ha='center', va='bottom', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('efficiency_distribution_boxplot.png', dpi=600, bbox_inches='tight')
plt.close()

# Performing statistical significance testing
from scipy import stats
print("\n" + "="*95)
print("STATISTICAL SIGNIFICANCE TESTING")
print("="*95)

# Performing Wilcoxon signed-rank test for paired samples
stat1, p_val1 = stats.wilcoxon(error_static, error_dt)
stat2, p_val2 = stats.wilcoxon(error_p95, error_dt)
print(f"Wilcoxon test (Baseline 1 vs DT): statistic={stat1:.2f}, p-value={p_val1:.2e}")
print(f"Wilcoxon test (Baseline 2 vs DT): statistic={stat2:.2f}, p-value={p_val2:.2e}")

# Performing Mann-Whitney U test for independent samples
stat3, p_val3 = stats.mannwhitneyu(error_static, error_dt, alternative='greater')
stat4, p_val4 = stats.mannwhitneyu(error_p95, error_dt, alternative='greater')
print(f"Mann-Whitney U test (Baseline 1 > DT): statistic={stat3:.2f}, p-value={p_val3:.2e}")
print(f"Mann-Whitney U test (Baseline 2 > DT): statistic={stat4:.2f}, p-value={p_val4:.2e}")

def cohens_d(x, y):
    """
    Calculating Cohen's d for effect size.

    Parameters:
    - x: First set of values.
    - y: Second set of values.

    Returns:
    - Cohen's d value.
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

# Calculating effect size using Cohen's d
d1 = cohens_d(error_static, error_dt)
d2 = cohens_d(error_p95, error_dt)
print(f"Cohen's d (Baseline 1 vs DT): {d1:.3f}")
print(f"Cohen's d (Baseline 2 vs DT): {d2:.3f}")

print("\nAll plots saved successfully!")
print("Generated files:")
print("- actual_vs_predicted_all_methods.png")
print("- mae_rmse_comparison_all_methods.png")
print("- advanced_metrics_comparison_all_methods.png")
print("- over_provisioning_comparison_all_methods.png")
print("- performance_radar_chart.png")
