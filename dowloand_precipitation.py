
import os
from pathlib import Path

# === Get r2_download.py helper module (if running on Colab/Kaggle) ===
HELPER_URL = "https://raw.githubusercontent.com/SALA-AI-LATAM/hackathon-participants/main/r2_download.py"

if not Path("r2_download.py").exists():
    print("Downloading r2_download.py...")
    import urllib.request
    urllib.request.urlretrieve(HELPER_URL, "r2_download.py")
    print("Done.")
else:
    print("r2_download.py already present.")

# === Set R2 credentials ===
# Search common locations for the .env file
ENV_NAME = "participant-download.env"
ENV_SEARCH_PATHS = [
    Path(ENV_NAME),               # current directory (notebook folder)
    Path("..") / ENV_NAME,        # parent directory (repo root)
    Path("/workspace") / ENV_NAME,  # RunPod workspace root
]

env_file = None
for p in ENV_SEARCH_PATHS:
    if p.exists():
        env_file = p
        break

if env_file is not None:
    for line in open(env_file):
        line = line.strip()
        if line and not line.startswith("#"):
            key, val = line.removeprefix("export ").split("=", 1)
            os.environ[key] = val.strip('"')
    print(f"Credentials loaded from {env_file}")
elif "R2_ENDPOINT" in os.environ:
    print("Using pre-set environment variables")
else:
    # Option B: Paste credentials directly (organizers will provide these)
    os.environ["R2_ENDPOINT"] = "https://6200702e94592ad231a53daba00f8a5d.r2.cloudflarestorage.com"
    os.environ["R2_ACCESS_KEY_ID"] = "93bb95ebfe47d5ef93c45efe3c108ca8"
    os.environ["R2_SECRET_ACCESS_KEY"] = "cee49fead9c1a8ac2741a4c2703c908efc5d965100a2d8d20c233fce05547a55"
    os.environ["R2_BUCKET"] = "sala-2026-hackathon-data"
    print("Using inline credentials (edit the values above if they say YOUR_...)")

import r2_download as hd

print(f"Environment: {hd._detect_environment()}")
print(f"Data directory: {hd._default_data_dir()}")

# === Download precipitation-nowcasting dataset from R2 ===
client = hd.get_s3_client()
manifest = hd.load_manifest(
    bucket=os.environ["R2_BUCKET"], s3_client=client, cache_path="manifest.json"
)
hd.summarize_manifest(manifest)

stats = hd.download_dataset(manifest, dataset_name="precipitation-nowcasting")
print(f"\nDownloaded: {stats['downloaded']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")