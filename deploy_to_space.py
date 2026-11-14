import os
import sys
import io
import argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from huggingface_hub import HfApi

SPACE_REPO_ID = "minhunhooo/video-subtitle-summarization"
DEPLOY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deploy-huggingface")

def main():
    parser = argparse.ArgumentParser(description="Deploy Video Subtitle Summarization App to HF Spaces")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Write Token")
    parser.add_argument("--repo_id", type=str, default=SPACE_REPO_ID)
    parser.add_argument("--message", type=str, default="Refactor layout: top-level navigation, clean text tab, distinct summary cards")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("[!] Token not provided via --token or HF_TOKEN environment variable.")
        token = input("Enter your Hugging Face Write Token: ").strip()

    api = HfApi(token=token)

    print(f"[*] Uploading files from '{DEPLOY_DIR}' to Space '{args.repo_id}'...")
    commit_info = api.upload_folder(
        folder_path=DEPLOY_DIR,
        repo_id=args.repo_id,
        repo_type="space",
        commit_message=args.message,
        ignore_patterns=["__pycache__/*", "*.pyc", "*.DS_Store", ".cache/*"],
        token=token
    )
    print(f"[+] Upload committed successfully! Commit: {commit_info.commit_id[:7] if hasattr(commit_info, 'commit_id') else 'OK'}")
    print(f" Space URL: https://huggingface.co/spaces/{args.repo_id}")

if __name__ == "__main__":
    main()
