import os
from huggingface_hub import snapshot_download

def download_models():
    # TripoSR is the most robust and RAM-efficient LRM for a 16GB VPS.
    repo_id = "stabilityai/TripoSR"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_dir = os.path.join(base_dir, "models", "triposr")
    
    print(f"🚀 Downloading TripoSR weights (StabilityAI) to: {local_dir}")
    try:
        snapshot_download(
            repo_id=repo_id, 
            local_dir=local_dir
        )
        print("✅ TripoSR weights ready. Transitioning from Ferrero to Neural 3D.")
    except Exception as e:
        print(f"❌ Error downloading TripoSR: {e}")

if __name__ == "__main__":
    download_models()
