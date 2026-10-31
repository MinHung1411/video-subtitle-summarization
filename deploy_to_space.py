"""
Deploy to Hugging Face Spaces Script:
Uploads the modular Gradio application from `deploy-huggingface/`
to `minhunhooo/video-subtitle-summarization` on Hugging Face Spaces.
Cleans up old monolithic BART weights to optimize Space startup.
"""

import os
import sys
import io
import argparse

# Force UTF-8 encoding for console output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from huggingface_hub import HfApi, delete_file

SPACE_REPO_ID = "minhunhooo/video-subtitle-summarization"
DEPLOY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deploy-huggingface")

def main():
    parser = argparse.ArgumentParser(description="Deploy Video Subtitle Summarization App to HF Spaces")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Write Token")
    parser.add_argument("--repo_id", type=str, default=SPACE_REPO_ID, help="HF Space Repo ID")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("[!] Token not provided via --token or HF_TOKEN environment variable.")
        token = input("Enter your Hugging Face Write Token: ").strip()

    if not token:
        print("[x] Error: Token cannot be empty. Aborting.")
        sys.exit(1)

    api = HfApi(token=token)

    # 1. Verify Authentication
    print("[*] Verifying Hugging Face Token...")
    try:
        user_info = api.whoami()
        username = user_info.get("name", "Unknown")
        print(f"[+] Authenticated successfully as: {username}")
    except Exception as e:
        print(f"[x] Authentication failed: {e}")
        sys.exit(1)

    # 2. Inspect Existing Space Files
    print(f"[*] Checking existing files in Space: {args.repo_id}...")
    try:
        existing_files = api.list_repo_files(repo_id=args.repo_id, repo_type="space")
        print(f"[+] Found {len(existing_files)} existing files in Space.")
    except Exception as e:
        print(f"[x] Could not inspect Space {args.repo_id}: {e}")
        sys.exit(1)

    # 3. Upload new modular app and clean up old bart_best weights
    print(f"[*] Uploading files from '{DEPLOY_DIR}' to Space '{args.repo_id}'...")
    try:
        commit_info = api.upload_folder(
            folder_path=DEPLOY_DIR,
            repo_id=args.repo_id,
            repo_type="space",
            commit_message="Deploy production multilingual mT5 pipeline with Whisper STT & hierarchical chunking",
            delete_patterns=["bart_best/*"],
            ignore_patterns=["__pycache__/*", "*.pyc", "*.DS_Store"],
            token=token
        )
        print(f"[+] Upload committed successfully! Commit: {commit_info.commit_id[:7] if hasattr(commit_info, 'commit_id') else 'OK'}")
    except Exception as e:
        print(f"[x] Upload failed: {e}")
        sys.exit(1)

    # 4. Upload summary
    print("[+] Repository synchronized and legacy files purged.")

    # 5. Output Summary and Next Steps
    print("\n" + "=" * 60)
    print(" Deployment Complete!")
    print(f" Space URL: https://huggingface.co/spaces/{args.repo_id}")
    print(" Hugging Face Space is now building container & installing dependencies.")
    print("=" * 60)

if __name__ == "__main__":
    main()

