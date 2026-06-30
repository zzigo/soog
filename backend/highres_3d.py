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
                weight_name="model.ckpt"
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
            
            # Blend with neutral gray background for TripoSR (expects RGB)
            image_np = np.array(image).astype(np.float32) / 255.0
            if image_np.shape[-1] == 4:
                # Composite: fg * alpha + bg * (1 - alpha)
                image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + (1 - image_np[:, :, 3:4]) * 0.5
                image = Image.fromarray((image_np * 255.0).astype(np.uint8))
            else:
                image = image.convert("RGB")
            
            if not os.path.exists(self.models_dir):
                logging.warning("Weights missing! Placeholder mode.")
                mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
                mesh.export(output_stl_path)
                return {"status": "success", "model": "MOCK", "vertices": len(mesh.vertices)}

            self._load_model()
            logging.info("Step 2: TripoSR Neural Inference...")
            
            with torch.no_grad():
                scene_codes = self.model([image], device=device)
                # Lower resolution (128 instead of 256) to save memory on VPS
                # Enable vertex colors (has_vertex_color=True)
                meshes = self.model.extract_mesh(scene_codes, has_vertex_color=True, resolution=128)
                
            # TripoSR returns a list of meshes
            final_mesh = meshes[0]
            
            # Export using trimesh. TripoSR extract_mesh returns a trimesh.Trimesh object directly.
            # We change extension to .glb to support vertex colors (STL doesn't support them well)
            output_glb_path = output_stl_path.replace(".stl", ".glb")
            final_mesh.export(output_glb_path)
            
            # Manual cleanup to prevent OOM
            del scene_codes
            del meshes
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            return {
                "status": "success",
                "model": "TripoSR (StabilityAI)",
                "vertices": len(final_mesh.vertices),
                "faces": len(final_mesh.faces),
                "has_colors": True,
                "format": "glb"
            }
        except Exception as e:
            logging.error(f"3D Reconstruction failed: {e}")
            return {"status": "error", "message": str(e)}

def run_reconstruction(image_path, output_path, prompt=""):
    engine = HighRes3DEngine()
    return engine.process_sketch(image_path, output_path, prompt=prompt)
