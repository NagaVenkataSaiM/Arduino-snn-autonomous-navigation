import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. SETUP & DATA GENERATION ---
def generate_sensor_data(samples=100):
    # Generating noisy sensor data (simulating real world TCS3200)
    # RED Object: High Red, random low Green/Blue
    red_X = np.random.normal(loc=[0.9, 0.15, 0.15], scale=0.1, size=(samples, 3))
    red_y = np.zeros(samples, dtype=int) 

    # GREEN Object: High Green
    green_X = np.random.normal(loc=[0.15, 0.9, 0.15], scale=0.1, size=(samples, 3))
    green_y = np.ones(samples, dtype=int)

    # BLUE Object: High Blue
    blue_X = np.random.normal(loc=[0.15, 0.15, 0.9], scale=0.1, size=(samples, 3))
    blue_y = np.full(samples, 2, dtype=int)

    X = np.vstack([red_X, green_X, blue_X])
    y = np.concatenate([red_y, green_y, blue_y])
    X = np.clip(X, 0, 1) # Keep values valid
    return X, y

# --- 2. SNN CLASS (LIF Model) ---
class TrafficLightSNN:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weights = np.random.uniform(0.0, 0.2, (num_inputs, num_outputs))
        self.v_threshold = 25.0
        self.v_decay = 0.8
        self.lr = 0.02

    def forward(self, input_currents, duration_ms=50, train_target=None):
        voltage = np.zeros(self.num_outputs)
        spike_train = np.zeros((duration_ms, self.num_outputs))
        v_trace = np.zeros((duration_ms, self.num_outputs))
        
        for t in range(duration_ms):
            # Input current (Noisy injection)
            noise = np.random.normal(0, 0.05, 3) 
            current_in = np.dot(input_currents + noise, self.weights)
            
            voltage = voltage * self.v_decay + current_in
            spikes = np.where(voltage >= self.v_threshold, 1, 0)
            voltage[spikes == 1] = 0 # Reset
            
            spike_train[t] = spikes
            v_trace[t] = voltage

            # Learning (Supervised Hebbian)
            if train_target is not None:
                target_spikes = np.zeros(self.num_outputs)
                target_spikes[train_target] = 1 
                error = target_spikes - spikes
                for out_idx in range(self.num_outputs):
                    if error[out_idx] != 0:
                        self.weights[:, out_idx] += self.lr * error[out_idx] * input_currents
                        
        return spike_train, v_trace

# --- 3. TRAINING ---
X, y = generate_sensor_data(samples=50)
indices = np.arange(len(X))
np.random.shuffle(indices)
X, y = X[indices], y[indices]

snn = TrafficLightSNN(3, 3)
print("Training SNN...")

for i in range(len(X)):
    snn.forward(X[i], train_target=y[i])

print("Training Complete.")

# --- 4. VISUALIZATION 1: NETWORK DYNAMICS (All Cases) ---
# We define 3 distinct test cases to simulate the rover seeing each color
test_cases = [
    ("Red Input (Stop)", [1.0, 0.1, 0.1]),
    ("Green Input (Move)", [0.1, 1.0, 0.1]),
    ("Blue Input (Backward)", [0.1, 0.1, 1.0])
]

fig1, axes = plt.subplots(3, 3, figsize=(14, 10))
fig1.suptitle("Figure 1: SNN Response to Different Traffic Signals", fontsize=16)

total_spikes_snn = 0

for i, (name, inputs) in enumerate(test_cases):
    spikes, v_trace = snn.forward(np.array(inputs), duration_ms=50)
    total_spikes_snn += np.sum(spikes)
    
    # Voltage Trace (Row 1)
    time = np.arange(50)
    ax_v = axes[0, i]
    ax_v.plot(time, v_trace[:, 0], 'r', label="Stop Neuron")
    ax_v.plot(time, v_trace[:, 1], 'g', label="Move Neuron")
    ax_v.plot(time, v_trace[:, 2], 'b', label="Backward Neuron")
    ax_v.axhline(snn.v_threshold, color='k', linestyle='--', alpha=0.5)
    ax_v.set_title(f"Membrane Potential\n({name})")
    if i==0: ax_v.set_ylabel("Voltage (mV)")
    if i==2: ax_v.legend(loc='upper right', fontsize='small')

    # Raster Plot (Row 2)
    ax_r = axes[1, i]
    # Draw spikes
    colors = ['r', 'g', 'b']
    labels = ['Stop', 'Move', 'Backward']
    for neuron_idx in range(3):
        spike_times = np.where(spikes[:, neuron_idx] == 1)[0]
        ax_r.vlines(spike_times, neuron_idx, neuron_idx+1, color=colors[neuron_idx])
    
    ax_r.set_yticks([0.5, 1.5, 2.5])
    ax_r.set_yticklabels(labels)
    ax_r.set_title(f"Spike Raster ({name})")
    ax_r.set_ylim(0, 3)

    # Input Strength (Row 3 - Visualizing what the sensor sees)
    ax_in = axes[2, i]
    ax_in.bar(['R', 'G', 'B'], inputs, color=['r', 'g', 'b'], alpha=0.6)
    ax_in.set_title("Sensor Input Levels")
    ax_in.set_ylim(0, 1.2)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- 5. VISUALIZATION 2: EFFICIENCY & WEIGHTS ---
fig2 = plt.figure(figsize=(12, 6))
gs = fig2.add_gridspec(1, 2)

# Heatmap
ax_heat = fig2.add_subplot(gs[0, 0])
sns.heatmap(snn.weights, annot=True, fmt=".1f", cmap="viridis", 
            xticklabels=["Stop", "Move", "Backward"], 
            yticklabels=["Red In", "Green In", "Blue In"], ax=ax_heat)
ax_heat.set_title("Learned Synaptic Weights")

# Energy Efficiency Comparison
ax_energy = fig2.add_subplot(gs[0, 1])

# Calculation:
# ANN Ops = (Inputs * Neurons) per step * duration
# SNN Ops = Only when a spike occurs (approx)
ann_ops = (3 * 3) * 50 * 3 # 3 inputs * 3 neurons * 50 steps * 3 test cases
snn_ops = total_spikes_snn * 3 # approx 3 ops per spike (update weights/voltage)

categories = ['Standard ANN', 'Your SNN']
values = [ann_ops, snn_ops]

bars = ax_energy.bar(categories, values, color=['gray', 'green'])
ax_energy.set_title("Computational Cost (Operations)")
ax_energy.set_ylabel("Estimated Operations")

# Add text labels
for bar in bars:
    height = bar.get_height()
    ax_energy.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom')

plt.suptitle("Figure 2: Learning Structure & Energy Efficiency", fontsize=16)
plt.tight_layout()
plt.show()

# --- 6. PRINT WEIGHTS ---
print("\n=== COPY THESE WEIGHTS TO ARDUINO ===")
print("float weights[3][3] = {")
for row in snn.weights:
    print(f"  {{ {row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f} }},")
print("};")


# float weights[3][3] = {
#   { 26.1712, 1.9972, 1.8299 },
#   { 1.5297, 26.2905, 2.2284 },
#   { 1.7687, 1.5462, 26.1924 },
# };