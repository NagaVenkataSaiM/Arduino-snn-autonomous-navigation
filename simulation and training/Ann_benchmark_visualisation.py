import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import csv
import sys

# --- CONFIG ---
SERIAL_PORT = 'COM5'   # <--- CHECK THIS
BAUD_RATE = 115200
MAX_LIVE_POINTS = 60   

# --- DATA STORAGE (PERMANENT SESSION) ---
session_snn = []
session_ann = []
session_r = []
session_g = []
session_b = []
session_act_snn = []
session_act_ann = []

# --- LIVE BUFFERS (SCROLLING VIEW) ---
live_snn = [0] * MAX_LIVE_POINTS
live_ann = [0] * MAX_LIVE_POINTS
live_r = [0] * MAX_LIVE_POINTS
live_g = [0] * MAX_LIVE_POINTS
live_b = [0] * MAX_LIVE_POINTS
live_act_snn = [0] * MAX_LIVE_POINTS
live_act_ann = [0] * MAX_LIVE_POINTS

# --- CONNECT ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    print(f"✅ Connected to {SERIAL_PORT}")
    print("📈 Dashboard Running... Drive mostly on SAND for best results!")
    print("❌ Close the window to stop and generate the full 3-panel report.")
    ser.reset_input_buffer()
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit()

# --- LIVE DASHBOARD SETUP ---
plt.style.use('seaborn-v0_8-darkgrid')
fig_live, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
fig_live.canvas.manager.set_window_title('LIVE ROVER TELEMETRY (Close to Analyze)')

# 1. Latency
line_ann, = ax1.plot([], [], color='#E74C3C', linewidth=2, label='ANN Latency')
line_snn, = ax1.plot([], [], color='#2ECC71', linewidth=2, label='SNN Latency')
ax1.set_ylabel('Time (µs)')
ax1.set_ylim(0, 8000)
ax1.legend(loc='upper right')
ax1.set_title('1. Real-Time Computational Cost')

# 2. Sensors
line_r, = ax2.plot([], [], color='red', label='Red')
line_g, = ax2.plot([], [], color='green', label='Green')
line_b, = ax2.plot([], [], color='blue', label='Blue')
ax2.set_ylabel('Intensity')
ax2.set_ylim(-0.1, 1.1)
ax2.legend(loc='upper right')
ax2.set_title('2. Visual Input (RGB)')

# 3. Actions
line_act_snn, = ax3.step([], [], color='#2ECC71', linewidth=2, label='SNN Action')
line_act_ann, = ax3.step([], [], color='#E74C3C', linestyle='--', label='ANN Prediction')
ax3.set_ylabel('Action')
ax3.set_yticks([0, 1, 2])
ax3.set_yticklabels(['STOP', 'FWD', 'BACK'])
ax3.set_ylim(-0.5, 2.5)
ax3.legend(loc='upper right')
ax3.set_title('3. Motor Decisions')

def update(frame):
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            parts = line.split(',')
            
            if len(parts) >= 7:
                # Parse
                t_snn = float(parts[0])
                t_ann = float(parts[1])
                r_val = float(parts[2])
                g_val = float(parts[3])
                b_val = float(parts[4])
                act_s = float(parts[5])
                act_a = float(parts[6])

                # 1. SAVE TO PERMANENT SESSION (For Final Report)
                session_snn.append(t_snn)
                session_ann.append(t_ann)
                session_r.append(r_val)
                session_g.append(g_val)
                session_b.append(b_val)
                session_act_snn.append(act_s)
                session_act_ann.append(act_a)

                # 2. UPDATE LIVE BUFFERS
                live_snn.append(t_snn)
                live_ann.append(t_ann)
                live_r.append(r_val)
                live_g.append(g_val)
                live_b.append(b_val)
                live_act_snn.append(act_s)
                live_act_ann.append(act_a)

                # Scroll
                if len(live_snn) > MAX_LIVE_POINTS:
                    live_snn.pop(0); live_ann.pop(0)
                    live_r.pop(0); live_g.pop(0); live_b.pop(0)
                    live_act_snn.pop(0); live_act_ann.pop(0)
        except:
            pass

    # Draw Live Lines
    x = range(len(live_snn))
    line_snn.set_data(x, live_snn)
    line_ann.set_data(x, live_ann)
    line_r.set_data(x, live_r)
    line_g.set_data(x, live_g)
    line_b.set_data(x, live_b)
    line_act_snn.set_data(x, live_act_snn)
    line_act_ann.set_data(x, live_act_ann)
    
    ax1.set_xlim(0, MAX_LIVE_POINTS)
    return line_snn, line_ann, line_r, line_g, line_b, line_act_snn, line_act_ann

ani = animation.FuncAnimation(fig_live, update, interval=30, blit=True)
plt.show() # CODE PAUSES HERE

# ==========================================
# FINAL REPORT GENERATION
# ==========================================
if len(session_snn) == 0:
    print("⚠️ No data collected.")
    sys.exit()

print(f"\n📊 Generating Final 3-Panel Report ({len(session_snn)} samples)...")

# 1. Stats
avg_snn = np.mean(session_snn)
avg_ann = np.mean(session_ann)
total_snn = np.sum(session_snn) / 1000.0
total_ann = np.sum(session_ann) / 1000.0

if avg_snn < avg_ann:
    ratio = avg_ann / avg_snn
    verdict = f"SNN is {ratio:.2f}x Faster"
else:
    ratio = avg_snn / avg_ann
    verdict = f"SNN is {ratio:.2f}x Slower"

# 2. Time Axis
time_axis = [i * 0.055 for i in range(len(session_snn))]

# 3. CREATE FINAL FIGURE (3 PANELS)
fig_final, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
fig_final.suptitle('Hardware Benchmark Final Report', fontsize=16, fontweight='bold')

# --- PANEL 1: LATENCY ---
ax1.plot(time_axis, session_ann, color='#E74C3C', linestyle='--', linewidth=2, label='ANN Latency')
ax1.plot(time_axis, session_snn, color='#2ECC71', linewidth=2, label='SNN Latency')
ax1.fill_between(time_axis, session_snn, color='#2ECC71', alpha=0.3)
ax1.set_ylabel('Latency (µs)')
ax1.set_title(f'1. Real-Time Computational Cost ({verdict})')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Stats Box in Panel 1
stats_text = f"AVG ANN: {avg_ann:.0f}µs | AVG SNN: {avg_snn:.0f}µs | Efficiency: {ratio:.2f}x"
ax1.text(0.02, 0.9, stats_text, transform=ax1.transAxes, 
         bbox=dict(facecolor='white', alpha=0.8), fontsize=10)

# --- PANEL 2: INPUTS (RGB) ---
ax2.plot(time_axis, session_r, color='red', alpha=0.8, label='Red')
ax2.plot(time_axis, session_g, color='green', alpha=0.8, label='Green')
ax2.plot(time_axis, session_b, color='blue', alpha=0.8, label='Blue')
ax2.set_ylabel('Sensor Intensity')
ax2.set_title('2. Visual Cortex Inputs')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# --- PANEL 3: ACTIONS ---
ax3.step(time_axis, session_act_snn, where='post', color='#2ECC71', linewidth=2, label='SNN Action')
ax3.step(time_axis, session_act_ann, where='post', color='#E74C3C', linestyle='--', label='ANN Prediction')
ax3.set_ylabel('Motor State')
ax3.set_xlabel('Time (Seconds)')
ax3.set_yticks([0, 1, 2])
ax3.set_yticklabels(['STOP', 'FWD', 'BACK'])
ax3.set_title('3. Action Decisions')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(top=0.93)

# Save Image
plt.savefig('Final_Benchmark_Report.png', dpi=300)
print("✅ Saved chart to 'Final_Benchmark_Report.png'")

# Save CSV
with open('rover_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Time', 'SNN_Lat', 'ANN_Lat', 'R', 'G', 'B', 'SNN_Act', 'ANN_Act'])
    for i in range(len(session_snn)):
        writer.writerow([time_axis[i], session_snn[i], session_ann[i], 
                         session_r[i], session_g[i], session_b[i],
                         session_act_snn[i], session_act_ann[i]])
print("✅ Saved raw data to 'rover_data.csv'")

plt.show()