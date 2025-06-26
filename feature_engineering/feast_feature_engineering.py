import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta

from minio import Minio, S3Error

MINIO_ENDPOINT = "http://minio-service.fraud-detection.svc.cluster.local:9000"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minio123"
MINIO_BUCKET = "mlpipeline"

REMOTE_FEATURE_REPO_DIR = "artifacts/feature_repo/"
REMOTE_DATA_DIR = REMOTE_FEATURE_REPO_DIR + "data/"
REMOTE_INPUT_DIR = REMOTE_DATA_DIR + "input/"
REMOTE_OUTPUT_DIR = REMOTE_DATA_DIR + "output/"


def feast_apply(feature_repo_path: str) -> bool:
    print(f"Will run feast apply in {feature_repo_path}")
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
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing 'feast apply': {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def feast_materialize(feature_repo_path: str) -> bool:
    print(f"Will run feast materialize in {feature_repo_path}")

    end_date = datetime.utcnow().isoformat()
    start_date = (datetime.utcnow() - timedelta(days=365)).isoformat()

    try:
        result = subprocess.run(
            ["feast", "materialize", start_date, end_date],
            cwd=feature_repo_path,
            check=True,
            capture_output=True,
            text=True
        )
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("Feast materialization completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing 'feast materialize': {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def upload_directory(remote_path, local_dir):
    client = Minio(
        MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_ENDPOINT.startswith("https://")
    )

    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            object_path = os.path.join(remote_path, os.path.relpath(local_file_path, local_dir))

            try:
                print(f"Uploading: {local_file_path} -> {object_path}")
                client.fput_object(MINIO_BUCKET, object_path, local_file_path)
            except S3Error as e:
                raise RuntimeError(f"Failed to upload {local_file_path}: {e}")

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
        if obj.object_name.endswith('/'):
            print(f"Creating directory: {obj.object_name}")
            dir_path = os.path.join(dest, obj.object_name.replace(directory_path, "").lstrip("/"))
            os.makedirs(dir_path, exist_ok=True)
            continue
            
        file_path = os.path.join(dest, obj.object_name.replace(directory_path, "").lstrip("/"))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        print(f"Downloading: {obj.object_name} -> {file_path}")
        client.fget_object(MINIO_BUCKET, obj.object_name, file_path)

    print("Download complete.")


def move_parquet_files_from_temp():
    """Move parquet files from temporary locations to their final expected locations."""
    output_dir = "/app/feature_repo/data/output"
    
    # Look for parquet files in temporary directories
    temp_parquet_dir = os.path.join(output_dir, "features.parquet", "_temporary")
    if os.path.exists(temp_parquet_dir):
        print("Found temporary parquet directory, moving files...")
        
        # Find all parquet files in temporary directories
        parquet_files = []
        for root, dirs, files in os.walk(temp_parquet_dir):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_files.append(os.path.join(root, file))
        
        if parquet_files:
            # Create final parquet directory
            final_parquet_dir = os.path.join(output_dir, "features.parquet")
            os.makedirs(final_parquet_dir, exist_ok=True)
            
            # Move parquet files to final location
            for i, parquet_file in enumerate(parquet_files):
                final_file = os.path.join(final_parquet_dir, f"part-{i:05d}.parquet")
                print(f"Moving {parquet_file} -> {final_file}")
                os.rename(parquet_file, final_file)
            
            print(f"Moved {len(parquet_files)} parquet files to final location")
        else:
            print("No parquet files found in temporary directory")
    else:
        print("No temporary parquet directory found")


def main():
    parser = argparse.ArgumentParser(description="Run feature engineering with Feast")
    parser.add_argument("--feature-repo-path", required=True, help="Path to the feature repository")
    args = parser.parse_args()

    local_data_dir = "/app/feature_repo/data/"
    local_input_dir = local_data_dir + "input/"
    local_output_dir = local_data_dir + "output/"

    download_artifacts(REMOTE_INPUT_DIR, local_input_dir)
    download_artifacts(REMOTE_OUTPUT_DIR, local_output_dir)

    os.makedirs(local_output_dir, exist_ok=True)

    move_parquet_files_from_temp()

    apply_success = feast_apply(args.feature_repo_path)
    if not apply_success:
        with open("/tmp/kfp/outputs/features_ok", "w") as f:
            f.write("false")
        sys.exit(1)

    materialize_success = feast_materialize(args.feature_repo_path)

    print("Contents of output directory after materialization:")
    for root, dirs, files in os.walk(local_output_dir):
        for file in files:
            print(f"  {os.path.join(root, file)}")

    online_store_db = os.path.join(local_output_dir, "online_store.db")
    if os.path.exists(online_store_db):
        print(f"Online store database exists at {online_store_db}")
    else:
        raise RuntimeError(f"WARNING: Online store database not found at {online_store_db}")

    upload_directory(REMOTE_OUTPUT_DIR, local_output_dir)

    with open("/tmp/kfp/outputs/features_ok", "w") as f:
        f.write("true" if apply_success and materialize_success else "false")

    if not (apply_success and materialize_success):
        sys.exit(1)


if __name__ == '__main__':
    main()
