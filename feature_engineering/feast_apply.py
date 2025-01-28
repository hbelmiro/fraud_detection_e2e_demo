import argparse
import os
import subprocess
import sys
from typing import TextIO

from minio import Minio, S3Error

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")

REMOTE_FEATURE_REPO_DIR = os.getenv("FEATURE_REPO_REMOTE_DIR")
REMOTE_DATA_DIR = os.path.join(REMOTE_FEATURE_REPO_DIR, "data")
REMOTE_INPUT_DIR = os.path.join(REMOTE_DATA_DIR, "input")
REMOTE_OUTPUT_DIR = os.path.join(REMOTE_DATA_DIR, "output")


def feast_apply(feature_repo_path: str):
    print("Will run feast apply in {}".format(feature_repo_path))
    try:
        result = subprocess.run(
            ["feast", "apply"],
            cwd=feature_repo_path,
            check=True,
            capture_output=True,
            text=True
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Feast applied successfully!")

        with open("/tmp/kfp/outputs/features_ok", "w") as f:
            f.write("true")
    except subprocess.CalledProcessError as e:
        print(f"Error executing 'feast apply': {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)

        with open("/tmp/kfp/outputs/features_ok", "w") as f:
            f.write("false")

        sys.exit(e.returncode)


def upload_directory(remote_path, local_dir):
    """Upload all files from a local directory to a MinIO bucket."""
    client = Minio(
        MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_ENDPOINT.startswith("https://")
    )

    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Create the object path by prefixing the directory path
            object_path = os.path.join(remote_path, os.path.relpath(local_file_path, local_dir))

            try:
                print(f"Uploading: {local_file_path} -> {object_path}")
                client.fput_object(MINIO_BUCKET, object_path, local_file_path)
            except S3Error as e:
                print(f"Failed to upload {local_file_path}: {e}")

    print("Upload complete.")


def download_artifacts(directory_path, dest):
    client = Minio(
        MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_ENDPOINT.startswith("https://")
    )

    os.makedirs(dest, exist_ok=True)

    objects = client.list_objects(MINIO_BUCKET, prefix=directory_path, recursive=True)

    for obj in objects:
        file_path = os.path.join(dest, obj.object_name.replace(directory_path, "").lstrip("/"))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        print(f"Downloading: {obj.object_name} -> {file_path}")
        client.fget_object(MINIO_BUCKET, obj.object_name, file_path)

    print("Download complete.")


def main():
    parser = argparse.ArgumentParser(description="Run 'feast apply' in a specified feature repository path.")
    parser.add_argument("--feature-repo-path", required=True, help="Path to the feature repository")
    args = parser.parse_args()

    if not all([MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET]):
        raise ValueError("Missing required environment variables!")

    local_data_dir = os.path.join(args.feature_repo_path, "data")
    local_input_dir = os.path.join(local_data_dir, "input")
    local_output_dir = os.path.join(local_data_dir, "output")

    download_artifacts(REMOTE_INPUT_DIR, local_input_dir)

    os.makedirs(local_output_dir, exist_ok=True)

    feast_apply(args.feature_repo_path)

    upload_directory(REMOTE_OUTPUT_DIR, local_output_dir)


if __name__ == '__main__':
    main()
