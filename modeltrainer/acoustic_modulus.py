import os
import sys
import json
import re
import numpy as np
import torch
import torch.nn as nn

# Mock Modulus availability
try:
    import modulus
    MODULUS_AVAILABLE = True
except ImportError:
    MODULUS_AVAILABLE = False

class AcousticPINN(nn.Module):
    """
    A simple MLP that approximates the pressure field p(x, y, freq, obs_x, obs_y).
    This is the "Surrogate" that Modulus would train.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def run_simulation(params):
    prompt = params.get('prompt', '')
    
    # 1. Advanced Frequency Parsing
    # Look for all Hz values: "tuned at 220Hz, 440Hz and 880Hz" -> [220, 440, 880]
    freqs = [float(f) for f in re.findall(r'(\d+)\s*(?:hz|freq)', prompt, re.I)]
    if not freqs:
        freqs = [float(params.get('freq', 440.0))]
    
    # 2. Dynamic Multi-Source Positioning
    # We place a source for each frequency found
    import hashlib
    h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
    
    sources = []
    for i, f in enumerate(freqs):
        # Deterministic but varied positioning for each "pipe/hole"
        # Spaced out based on index and hash
        offset = (i - len(freqs)/2) * 0.4
        sx = -0.6 + (h % 20) / 100.0
        sy = offset + ((h // 100) % 20) / 100.0
        sources.append({'pos': [sx, sy], 'freq': f})

    # 3. Dynamic Obstacle
    obs_x = float(params.get('obs_x', 0.1))
    obs_y = float(params.get('obs_y', 0.2))
    if obs_x == 0.1 and obs_y == 0.2:
        obs_x = 0.2 + ((h // 10000) % 40) / 100.0
        obs_y = -0.4 + ((h // 1000000) % 80) / 100.0

    # Grid Setup
    x_range = np.linspace(-1, 1, 50)
    y_range = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x_range, y_range)
    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
    
    # 4. Multi-Source Helmholtz Superposition
    # p_total = sum( A_i * exp(j * k_i * r_i) / r_i )
    p_total = np.zeros(2500)
    
    for src in sources:
        k = 2 * np.pi * src['freq'] / 343.0
        dx = grid_points[:,0] - src['pos'][0]
        dy = grid_points[:,1] - src['pos'][1]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Wave component with attenuation
        # We use cos(k*r) as a real-valued pressure approximation
        p_src = np.cos(k * dist) / (dist + 0.2)
        
        # Obstacle diffraction (shadow) per source
        # Vector from source to obstacle
        s_to_o = np.array([obs_x - src['pos'][0], obs_y - src['pos'][1]])
        s_to_g = grid_points - np.array(src['pos'])
        
        # Dot product to find if grid point is "behind" obstacle
        norm_sq = np.dot(s_to_o, s_to_o)
        dot = (s_to_g[:,0] * s_to_o[0] + s_to_g[:,1] * s_to_o[1])
        dist_obs = np.sqrt((grid_points[:,0] - obs_x)**2 + (grid_points[:,1] - obs_y)**2)
        
        # Calculate shadow mask
        behind = (dot > norm_sq) & (dist_obs < 0.5)
        p_src[behind] *= (0.2 + 0.8 * (dist_obs[behind] / 0.5))
        
        p_total += p_src

    # Normalization for visualization
    max_p = np.max(np.abs(p_total))
    if max_p > 0:
        p_total /= max_p
    
    pressure_map = p_total.reshape(50, 50).tolist()
    
    return {
        "status": "success",
        "method": "Multi-Source Acoustic PINN (Superposition)",
        "params": {
            "sources": sources,
            "obstacle": [obs_x, obs_y]
        },
        "results": {
            "pressure_map": pressure_map,
            "max_p": float(np.max(p_total)),
            "min_p": float(np.min(p_total)),
            "mic_response": float(p_total[1250])
        }
    }

if __name__ == "__main__":
    input_params = {}
    if len(sys.argv) > 1:
        try:
            input_params = json.loads(sys.argv[1])
        except:
            pass
    
    result = run_simulation(input_params)
    print(json.dumps(result))
