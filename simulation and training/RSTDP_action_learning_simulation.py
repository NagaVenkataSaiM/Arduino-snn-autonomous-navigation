import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- 1. CONFIGURATION ---
SAMPLES = 600
DURATION_MS = 50
LEARNING_RATE = 0.01 
INHIBITION = 5.0      

# --- 2. WINNER-TAKE-ALL SNN ---
class WinnerTakeAllSNN:
    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.weights = np.random.uniform(0.2, 0.6, (n_in, n_out))
        self.eligibility_trace = np.zeros((n_in, n_out))
        self.v_threshold = 4.0 
        self.v_decay = 0.9
        self.v_reset = 0.0
        
    def reset(self):
        self.eligibility_trace.fill(0)

    def forward_and_learn(self, inputs, reward=None, learning=True):
        voltage = np.zeros(self.n_out)
        input_trace = np.zeros(self.n_in)
        
        spike_record = []
        voltage_record = [] # New: Record voltage for plotting
        
        input_currents = inputs * 3.5 
        
        for t in range(DURATION_MS):
            noise = np.random.normal(0, 0.05, self.n_in)
            current = np.dot(input_currents + noise, self.weights)
            voltage = voltage * self.v_decay + current
            
            # HARD WINNER-TAKE-ALL
            potential_spikes = np.where(voltage >= self.v_threshold)[0]
            spikes = np.zeros(self.n_out)
            
            if len(potential_spikes) > 0:
                winner_idx = potential_spikes[np.argmax(voltage[potential_spikes])]
                spikes[winner_idx] = 1
                voltage[winner_idx] = self.v_reset
                voltage -= INHIBITION # Suppress everyone
                voltage[winner_idx] = self.v_reset # Restore winner
            
            spike_record.append(spikes)
            voltage_record.append(voltage.copy())
            
            # R-STDP
            if learning:
                input_trace = input_trace * 0.9 + inputs
                for j in range(self.n_out):
                    if spikes[j] == 1:
                        self.eligibility_trace[:, j] += input_trace
                self.eligibility_trace *= 0.95

        # Weight Update
        if learning and reward is not None:
            d_weights = LEARNING_RATE * reward * self.eligibility_trace
            self.weights += d_weights
            self.weights = np.clip(self.weights, 0.01, 1.0)
            
        return np.array(spike_record), np.array(voltage_record)

# --- 3. TRAINING LOOP ---
def get_stimulus(forced_target=None):
    if forced_target is not None:
        target = forced_target
    else:
        target = np.random.randint(0, 3)
        
    noise = np.random.normal(0, 0.05, 3)
    # 0=Red, 1=Green, 2=Blue
    if target == 0: inputs = np.array([1, 0, 0]) + noise
    elif target == 1: inputs = np.array([0, 1, 0]) + noise
    else: inputs = np.array([0, 0, 1]) + noise
    return np.clip(inputs, 0, 1), target

snn = WinnerTakeAllSNN(3, 3)
accuracy_history = []
correct_counts = 0
window = 50

print("Training SNN...")
for episode in range(SAMPLES):
    snn.reset()
    inputs, target_action = get_stimulus()
    
    # Forward
    spikes, _ = snn.forward_and_learn(inputs, learning=True)
    
    # Action
    spike_counts = np.sum(spikes, axis=0)
    chosen_action = np.argmax(spike_counts) if np.sum(spike_counts) > 0 else -1
        
    # Reward
    if chosen_action == target_action:
        reward = 1.0
        correct_counts += 1
    elif chosen_action == -1:
        reward = -0.1 
    else:
        reward = -1.0 
        
    snn.forward_and_learn(inputs, reward=reward, learning=True)
    
    if (episode + 1) % window == 0:
        acc = correct_counts / window
        accuracy_history.append(acc)
        correct_counts = 0

print("Training Complete.")

# --- 4. ADVANCED VISUALIZATION DASHBOARD ---
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 3)

# Plot 1: Learning Curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(accuracy_history, marker='o', color='blue')
ax1.set_title("1. Accuracy over Time")
ax1.set_xlabel("Blocks (50 eps)")
ax1.set_ylabel("Accuracy")
ax1.grid(True)

# Plot 2: Weight Matrix Heatmap
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(snn.weights, annot=True, cmap="Reds", fmt=".2f", 
            xticklabels=['Stop', 'Move', 'Backward'], 
            yticklabels=['Red In', 'Green In', 'Blue In'], ax=ax2)
ax2.set_title("2. Learned Connectivity")

# Plot 3: Confusion Matrix (Accuracy Check)
ax3 = fig.add_subplot(gs[0, 2])
y_true = []
y_pred = []
for _ in range(100): # Test 100 samples
    inputs, target = get_stimulus()
    spikes, _ = snn.forward_and_learn(inputs, learning=False)
    counts = np.sum(spikes, axis=0)
    pred = np.argmax(counts) if np.sum(counts) > 0 else 3 # 3 = No Action
    y_true.append(target)
    y_pred.append(pred)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', 
            xticklabels=['Stop', 'Move', 'Backward'], 
            yticklabels=['Red', 'Green', 'Blue'], ax=ax3)
ax3.set_title("3. Confusion Matrix (Test Data)")

# Plot 4: Voltage Dynamics (The Competition)
# We simulate ONE specific case (Red Input) to see the voltages
ax4 = fig.add_subplot(gs[1, :]) # Spans whole bottom row
inputs, target = get_stimulus(forced_target=0) # Force RED input
spikes, v_trace = snn.forward_and_learn(inputs, learning=False)

time = np.arange(DURATION_MS)
ax4.plot(time, v_trace[:, 0], color='r', linewidth=2, label='Stop Neuron (Should Win)')
ax4.plot(time, v_trace[:, 1], color='g', linestyle='--', alpha=0.6, label='Move Neuron')
ax4.plot(time, v_trace[:, 2], color='b', linestyle='--', alpha=0.6, label='Backward Neuron')
ax4.axhline(snn.v_threshold, color='k', linestyle=':', label='Threshold')

# Highlight spikes
spike_times = np.where(spikes[:, 0] == 1)[0]
ax4.vlines(spike_times, snn.v_threshold, snn.v_threshold + 1, color='red', alpha=0.5)

ax4.set_title("4. Neuron Competition (Input: RED)")
ax4.set_ylabel("Membrane Potential (v)")
ax4.set_xlabel("Time (ms)")
ax4.legend(loc='upper right')
ax4.grid(True)

plt.tight_layout()
plt.show()

print("\n=== COPY THESE WEIGHTS TO ARDUINO ===")
print("float weights[3][3] = {")
print(len(snn.weights[0]))
for row in snn.weights:
    print(f"  {{ {row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f} }},")
print("};")

# float weights[3][3] = {
#   { 1.0000, 0.2441, 0.7725 },
#   { 0.4130, 1.0000, 0.3887 },
#   { 0.2334, 0.3299, 1.0000 },
# };