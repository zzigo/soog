import os
import sys
import torch
import numpy as np
from PIL import Image
import trimesh
import logging
import importlib

# Add TripoSR lib to path
LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'triposr_lib')
if LIB_PATH not in sys.path:
    sys.path.append(LIB_PATH)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HighRes3DEngine:
    def __init__(self):
        self.model = None
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(self.base_dir, "models", "triposr")
        
    def _load_model(self):
        if self.model is not None:
            return
            
        logging.info(f"Loading TripoSR weights from {self.models_dir} into {device}...")
        
        try:
            from tsr.system import TSR
            from tsr.utils import remove_background, resize_foreground, save_video
            
            self.model = TSR.from_pretrained(
                self.models_dir,
                config_name="config.yaml",
                checkpoint_name="model.ckpt"
            )
            self.model.to(device)
            self.model.eval()
        except Exception as e:
            logging.error(f"Failed to load TripoSR: {e}")
            raise
            
    def process_sketch(self, input_image_path, output_stl_path, prompt=""):
        try:
            logging.info("Step 1: Background removal & Pre-processing...")
            from tsr.utils import remove_background, resize_foreground
            
            image = Image.open(input_image_path)
            
            # Real neural pre-processing
            image = remove_background(image)
            image = resize_foreground(image, 0.85)
            
            if not os.path.exists(self.models_dir):
                logging.warning("Weights missing! Placeholder mode.")
                mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
                mesh.export(output_stl_path)
                return {"status": "success", "model": "MOCK", "vertices": len(mesh.vertices)}

            self._load_model()
            logging.info("Step 2: TripoSR Neural Inference...")
            
            with torch.no_grad():
                scene_codes = self.model([image], device=device)
                meshes = self.model.extract_meshes(scene_codes, resolution=256)
                
            # TripoSR returns a list of meshes
            final_mesh = meshes[0]
            
            # Convert to STL using trimesh
            # final_mesh is usually a dict/object with vertices and faces
            import trimesh
            t_mesh = trimesh.Trimesh(vertices=final_mesh.vertices, faces=final_mesh.faces)
            
            # Apply some basic orientation fix if needed
            # TripoSR coordinates might need rotation
            t_mesh.export(output_stl_path)
            
            return {
                "status": "success",
                "model": "TripoSR (StabilityAI)",
                "vertices": len(t_mesh.vertices),
                "faces": len(t_mesh.faces)
            }
        except Exception as e:
            logging.error(f"3D Reconstruction failed: {e}")
            return {"status": "error", "message": str(e)}

def run_reconstruction(image_path, output_path, prompt=""):
    engine = HighRes3DEngine()
    return engine.process_sketch(image_path, output_path, prompt=prompt)
