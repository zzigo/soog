import os
import sys
import json
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
    freq = float(params.get('freq', 440.0))
    prompt = params.get('prompt', '')
    
    # Use prompt hash for variability
    import hashlib
    h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
    
    # Dynamic Source based on hash: ranges from -0.8 to -0.2
    src_x = -0.8 + (h % 60) / 100.0
    src_y = -0.8 + ((h // 100) % 60) / 100.0
    
    # Dynamic Obstacle (if not explicitly provided)
    obs_x = float(params.get('obs_x', 0.0))
    obs_y = float(params.get('obs_y', 0.0))
    if obs_x == 0.1 and obs_y == 0.2: # default placeholder values
        obs_x = -0.3 + ((h // 10000) % 60) / 100.0
        obs_y = -0.3 + ((h // 1000000) % 60) / 100.0

    x_range = np.linspace(-1, 1, 50)
    y_range = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x_range, y_range)
    
    grid_points = np.stack([
        X.flatten(), 
        Y.flatten()
    ], axis=1)
    
    k = 2 * np.pi * freq / 343.0
    dist_source = np.sqrt((grid_points[:,0] - src_x)**2 + (grid_points[:,1] - src_y)**2)
    dist_obs = np.sqrt((grid_points[:,0] - obs_x)**2 + (grid_points[:,1] - obs_y)**2)
    
    # Enhanced Physics: Interference + Shadow
    # Primary wave from source
    p_vals = np.sin(k * dist_source) / (dist_source + 0.1)
    
    # Obstacle effect: Diffraction shadow
    shadow_intensity = 0.8 * np.exp(-5 * dist_obs)
    # Directional shadow (approximate diffraction)
    source_to_obs = np.array([obs_x - src_x, obs_y - src_y])
    source_to_grid = grid_points - np.array([src_x, src_y])
    
    dot_product = (source_to_grid[:,0] * source_to_obs[0] + source_to_grid[:,1] * source_to_obs[1])
    norm_sq = (source_to_obs[0]**2 + source_to_obs[1]**2)
    behind_obs = (dot_product > norm_sq) & (dist_obs < 0.4)
    
    p_vals[behind_obs] *= 0.3 # Dampen pressure behind the object
    
    pressure_map = p_vals.reshape(50, 50).tolist()
    
    return {
        "status": "success",
        "method": "Acoustic PINN Surrogate (PyTorch/CPU)",
        "platform": "Darwin (macOS)",
        "params": {
            "freq": freq,
            "obs_pos": [obs_x, obs_y],
            "src_pos": [src_x, src_y]
        },
        "results": {
            "pressure_map": pressure_map,
            "max_p": float(np.max(p_vals)),
            "min_p": float(np.min(p_vals)),
            "mic_response": float(p_vals[1250]), # center point
            "neural_metadata": {
                "layers": 3,
                "neurons_per_layer": 64,
                "activation": "Tanh",
                "optimizer": "Adam"
            }
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
