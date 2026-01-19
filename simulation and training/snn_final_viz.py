import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_and_classify(image_path, label):
    # 1. Load in COLOR
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: return None, None, None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    img_rgb = cv2.resize(img_rgb, (64, 64))
    img_gray = cv2.resize(img_gray, (64, 64))

    # 2. THE RETINA (Updated with your TUNED values)
    g1 = cv2.GaussianBlur(img_gray, (3, 3), 0)
    g2 = cv2.GaussianBlur(img_gray, (9, 9), 0) # Sigma 15
    dog_response = cv2.absdiff(g1, g2) 

    # Threshold: 35 (The "Sand Blocker")
    active_pixels = np.where(dog_response > 15, 1, 0) 
    
    # 3. RUN LIF NEURON
    duration_ms = 50
    input_spikes = np.zeros((active_pixels.size, duration_ms))
    flat_activity = active_pixels.flatten()
    
    for i in range(active_pixels.size):
        if flat_activity[i] == 1:
            input_spikes[i] = np.random.rand(duration_ms) < 0.25 
        else:
            input_spikes[i] = np.random.rand(duration_ms) < 0.001

    v = 0
    tau = 10.0
    threshold = 5.0 
    voltage_trace = []
    
    # Weight: 0.05 (The "Heavy" Neuron)
    w = 0.1
    has_fired = False
    
    for t in range(duration_ms):
        incoming_current = np.sum(input_spikes[:, t]) * w
        dv = (-v + incoming_current) / tau
        v += dv
        voltage_trace.append(v)
        if v >= threshold:
            has_fired = True

    # 4. VISUALIZATION
    final_img = img_rgb.copy()
    
    if has_fired:
        # Green = ACTIVE
        cv2.rectangle(final_img, (0,0), (63,63), (0, 255, 0), 2)
        cv2.putText(final_img, "ACTIVE", (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    else:
        # Red = SILENT
        cv2.rectangle(final_img, (0,0), (63,63), (0, 0, 255), 2)
        cv2.putText(final_img, "SILENT", (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    return final_img, active_pixels, voltage_trace, has_fired

# --- EXECUTION ---
fig, axes = plt.subplots(3, 2, figsize=(10, 10))

rock_img, rock_retina, rock_volt, rock_fired = process_and_classify('rock.jpg', "Martian Rock")
sand_img, sand_retina, sand_volt, sand_fired = process_and_classify('sand.jpg', "Martian Sand")

if rock_img is not None:
    # ROCK
    axes[0, 0].imshow(rock_img); axes[0, 0].set_title("Input: Martian Rock")
    axes[1, 0].imshow(rock_retina, cmap='jet'); axes[1, 0].set_title("Retina Output")
    axes[2, 0].plot(rock_volt, color='red'); axes[2, 0].axhline(5.0, color='black', linestyle='--')
    axes[2, 0].set_ylim(0, 10); axes[2, 0].set_title("Voltage (Spike Triggered)")

    # SAND
    axes[0, 1].imshow(sand_img); axes[0, 1].set_title("Input: Martian Sand")
    axes[1, 1].imshow(sand_retina, cmap='jet'); axes[1, 1].set_title("Retina Output")
    axes[2, 1].plot(sand_volt, color='blue'); axes[2, 1].axhline(5.0, color='black', linestyle='--')
    axes[2, 1].set_ylim(0, 10); axes[2, 1].set_title("Voltage (Sub-threshold)")

    plt.tight_layout()
    plt.show()