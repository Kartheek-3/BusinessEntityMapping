import argparse
import boto3
import os

def main():
    parser = argparse.ArgumentParser(description="Download final results from S3.")
    parser.add_argument("--s3-uri", default="s3://kartheek-business-er-2026/output/", help="S3 URI to output folder")
    parser.add_argument("--local-dir", required=True, help="Local directory to save files")
    args = parser.parse_args()
    
    os.makedirs(args.local_dir, exist_ok=True)
    
    s3 = boto3.client("s3")
    
    bucket = args.s3_uri.split("/")[2]
    prefix = "/".join(args.s3_uri.split("/")[3:])
    if not prefix.endswith("/"):
        prefix += "/"
        
    print(f"Downloading from bucket {bucket}, prefix {prefix}")
    
    files = ["matching_results.tsv", "candidate_pairs.tsv"]
    for file in files:
        key = prefix + file
        local_path = os.path.join(args.local_dir, file)
        try:
            print(f"Downloading {key} to {local_path}...")
            s3.download_file(bucket, key, local_path)
            print(f"Downloaded {file}")
        except Exception as e:
            print(f"Failed to download {file}: {e}")

if __name__ == "__main__":
    main()
