import os
from huggingface_hub import snapshot_download

def download_models():
    # We target the official InstantMesh repository
    repo_id = "TencentARC/InstantMesh"
    # Target directory inside backend/models
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_dir = os.path.join(base_dir, "models", "instantmesh")
    
    print(f"🚀 Downloading InstantMesh weights to: {local_dir}")
    print("This may take a while (~10GB)...")
    
    try:
        snapshot_download(
            repo_id=repo_id, 
            local_dir=local_dir,
            ignore_patterns=["*.msgpack", "*.h5"] # Save space by ignoring non-pytorch weights
        )
        print("✅ Download complete! Engine is ready for neural reconstruction.")
    except Exception as e:
        print(f"❌ Error downloading weights: {e}")

if __name__ == "__main__":
    download_models()
