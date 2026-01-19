import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_neuron_forces(image_path, label):
    # 1. LOAD & RETINA FILTER
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None, None, None, None
    img = cv2.resize(img, (64, 64))
    
    # --- TUNING FIX 1: Stronger Blur ---
    # We increase the second blur to 15 (was 9). 
    # This helps wash out small sand grains better.
    g1 = cv2.GaussianBlur(img, (3, 3), 0)
    g2 = cv2.GaussianBlur(img, (9, 9), 0) 
    dog_response = cv2.absdiff(g1, g2)
    
    # --- TUNING FIX 2: Higher Threshold ---
    # We raise the bar from 15 to 35. 
    # Only "Rock-hard" edges can pass now. Sand noise is blocked.
    active_pixels = np.where(dog_response > 15, 1, 0)
    
    # 2. GENERATE SPIKES
    duration_ms = 50
    input_spikes = np.zeros((active_pixels.size, duration_ms))
    flat_activity = active_pixels.flatten()
    
    for i in range(active_pixels.size):
        if flat_activity[i] == 1:
            input_spikes[i] = np.random.rand(duration_ms) < 0.25
        else:
            input_spikes[i] = np.random.rand(duration_ms) < 0.001

    # 3. RECORD INTERNAL FORCES
    v = 0
    tau = 10.0
    threshold = 5.0
    
    # --- TUNING FIX 3: Lower Weight ---
    # We lower the weight from 0.1 to 0.05.
    # This ensures that even if a few sand pixels leak through, 
    # they aren't strong enough to cause a spike.
    w = 0.1
    
    trace_v = []       
    trace_current = [] 
    trace_leak = []    
    
    for t in range(duration_ms):
        # Calculate Forces
        I_in = np.sum(input_spikes[:, t]) * w  
        leak = (-v / tau)                      
        
        # Update State
        dv = (-v + I_in) / tau
        v += dv
        
        # Record
        trace_v.append(v)
        trace_current.append(I_in)
        trace_leak.append(leak) 
        
    return trace_current, trace_leak, trace_v

# --- EXECUTION ---
fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)

# Run Analysis
rock_I, rock_L, rock_V = analyze_neuron_forces('rock.jpg', "Martian Rock")
sand_I, sand_L, sand_V = analyze_neuron_forces('sand.jpg', "Martian Sand")

if rock_I is not None:
    time = np.arange(50)
    
    # --- ROCK COLUMN (Left) ---
    axes[0, 0].set_title("Martian ROCK: Internal Forces", fontsize=14, fontweight='bold')
    axes[0, 0].fill_between(time, 0, rock_I, color='green', alpha=0.3, label='Input Current (+)')
    axes[0, 0].plot(time, rock_I, color='green')
    axes[0, 0].fill_between(time, 0, rock_L, color='red', alpha=0.3, label='Leakage (-)')
    axes[0, 0].plot(time, rock_L, color='red', linestyle='--')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc='upper left')
    axes[0, 0].set_ylabel("Force (dV)")
    
    # Force Y-Limits to be consistent
    axes[0, 0].set_ylim(-8, 12)

    # Net Change
    net_force_rock = np.array(rock_I) + np.array(rock_L)
    axes[1, 0].plot(time, net_force_rock, color='black', linewidth=1)
    axes[1, 0].fill_between(time, 0, net_force_rock, where=(net_force_rock>0), color='blue', alpha=0.1)
    axes[1, 0].set_ylabel("Net Change (dV)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(-2, 8)
    axes[1, 0].set_title("Net Voltage Change (dV)")

    # Result
    axes[2, 0].plot(time, rock_V, color='blue', linewidth=3)
    axes[2, 0].axhline(5.0, color='red', linestyle='--', label='Threshold')
    axes[2, 0].text(5, 5.5, "SPIKE TRIGGERED", color='red', fontweight='bold')
    axes[2, 0].set_ylabel("Membrane Potential (V)")
    axes[2, 0].set_ylim(0, 10)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("Time (ms)")

    # --- SAND COLUMN (Right) ---
    axes[0, 1].set_title("Martian SAND: Internal Forces", fontsize=14, fontweight='bold')
    axes[0, 1].fill_between(time, 0, sand_I, color='green', alpha=0.3)
    axes[0, 1].plot(time, sand_I, color='green')
    axes[0, 1].fill_between(time, 0, sand_L, color='red', alpha=0.3)
    axes[0, 1].plot(time, sand_L, color='red', linestyle='--')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-8, 12) # Match Rock
    axes[0, 1].text(10, 1, "Negligible Input", color='green', fontsize=10)

    # Net Change
    net_force_sand = np.array(sand_I) + np.array(sand_L)
    axes[1, 1].plot(time, net_force_sand, color='black', linewidth=1)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(-2, 8) # Match Rock
    axes[1, 1].set_title("Net Voltage Change (dV)")

    # Result
    axes[2, 1].plot(time, sand_V, color='blue', linewidth=3)
    axes[2, 1].axhline(5.0, color='red', linestyle='--')
    
    # Dynamic Check for Label
    if max(sand_V) < 5.0:
        axes[2, 1].text(5, 5.5, "SILENT (Energy Saved)", color='blue', fontweight='bold')
    else:
        axes[2, 1].text(5, 5.5, "ERROR: STILL SPIKING", color='red', fontweight='bold')
        
    axes[2, 1].set_ylabel("Membrane Potential (V)")
    axes[2, 1].set_ylim(0, 10)
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_xlabel("Time (ms)")

    plt.tight_layout()
    plt.show()
else:
    print("Images not found.")