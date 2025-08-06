from huggingface_hub import snapshot_download
import os

def download_internimage_model(model_name="internimage_g_22kto1k_512"):
    
    download_dir = f"models/{model_name}"
    model_path = f"OpenGVLab/{model_name}"
    if not os.path.exists(f"{download_dir}"):
        os.makedirs(download_dir, exist_ok=True)
        snapshot_download(repo_id=model_path, local_dir=download_dir, repo_type="model")
    else:
        print(f"Directory {download_dir} already exists. Skipping download.")

download_internimage_model("internimage_l_22kto1k_384")