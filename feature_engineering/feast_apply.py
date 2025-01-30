import os
import subprocess
import sys


def feast_apply(feature_repo_path: str):
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
    except subprocess.CalledProcessError as e:
        print(f"Error executing 'feast apply': {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)


def main():
    if len(sys.argv) < 2:
        script_name = os.path.basename(sys.argv[0])
        print(f"Usage: python {script_name} <feature_repo_path>")
        sys.exit(1)
    feature_repo_path = sys.argv[1]
    feast_apply(feature_repo_path)


if __name__ == '__main__':
    main()
