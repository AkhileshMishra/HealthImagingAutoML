#!/usr/bin/env python3
"""
AWS HealthImaging Pixel Data Fetcher

This script retrieves pixel data from AWS HealthImaging for use in ML pipelines.
It accepts an ImageSet ID and Datastore ID, fetches all image frames,
and saves them to the specified output directory.

Usage:
    python fetch.py --datastore-id <id> --image-set-id <id> --output-dir /opt/ml/processing/output
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_medical_imaging_client(region: str = None):
    """Create a boto3 client for AWS HealthImaging."""
    config = Config(
        retries={"max_attempts": 3, "mode": "adaptive"},
        connect_timeout=30,
        read_timeout=60,
    )
    return boto3.client(
        "medical-imaging",
        region_name=region or os.environ.get("AWS_REGION", "us-east-1"),
        config=config,
    )


def get_image_set_metadata(client, datastore_id: str, image_set_id: str) -> dict:
    """Retrieve metadata for an ImageSet."""
    logger.info(f"Fetching metadata for ImageSet: {image_set_id}")
    
    response = client.get_image_set_metadata(
        datastoreId=datastore_id,
        imageSetId=image_set_id,
    )
    
    # The metadata is returned as a streaming body in gzip format
    import gzip
    metadata_blob = response["imageSetMetadataBlob"].read()
    metadata_json = gzip.decompress(metadata_blob).decode("utf-8")
    return json.loads(metadata_json)


def extract_frame_ids(metadata: dict) -> list:
    """Extract all image frame IDs from ImageSet metadata."""
    frame_ids = []
    
    # Navigate the DICOM metadata structure
    study = metadata.get("Study", {})
    series_dict = study.get("Series", {})
    
    for series_id, series_data in series_dict.items():
        instances = series_data.get("Instances", {})
        for instance_id, instance_data in instances.items():
            image_frames = instance_data.get("ImageFrames", [])
            for frame in image_frames:
                frame_id = frame.get("ID")
                if frame_id:
                    frame_ids.append({
                        "frameId": frame_id,
                        "seriesId": series_id,
                        "instanceId": instance_id,
                    })
    
    logger.info(f"Found {len(frame_ids)} image frames")
    return frame_ids


def fetch_image_frame(
    client,
    datastore_id: str,
    image_set_id: str,
    frame_id: str,
) -> bytes:
    """Fetch a single image frame from HealthImaging."""
    response = client.get_image_frame(
        datastoreId=datastore_id,
        imageSetId=image_set_id,
        imageFrameInformation={"imageFrameId": frame_id},
    )
    return response["imageFrameBlob"].read()


def save_frame(frame_data: bytes, output_path: Path, frame_info: dict):
    """Save frame data to disk."""
    # Create filename from frame info
    filename = f"{frame_info['seriesId']}_{frame_info['instanceId']}_{frame_info['frameId']}.htj2k"
    filepath = output_path / filename
    
    with open(filepath, "wb") as f:
        f.write(frame_data)
    
    return filepath


def save_manifest(output_path: Path, manifest_data: dict):
    """Save a manifest file with metadata about fetched frames."""
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info(f"Saved manifest to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch pixel data from AWS HealthImaging"
    )
    parser.add_argument(
        "--datastore-id",
        required=True,
        help="AWS HealthImaging Datastore ID",
    )
    parser.add_argument(
        "--image-set-id",
        required=True,
        help="AWS HealthImaging ImageSet ID",
    )
    parser.add_argument(
        "--output-dir",
        default="/opt/ml/processing/output",
        help="Output directory for fetched frames",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames to fetch (0 = all)",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region for HealthImaging",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting fetch for ImageSet: {args.image_set_id}")
    logger.info(f"Datastore: {args.datastore_id}")
    logger.info(f"Output directory: {output_path}")
    
    # Initialize client
    client = get_medical_imaging_client(args.region)
    
    # Get ImageSet metadata
    metadata = get_image_set_metadata(client, args.datastore_id, args.image_set_id)
    
    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Extract frame IDs
    frame_ids = extract_frame_ids(metadata)
    
    if args.max_frames > 0:
        frame_ids = frame_ids[:args.max_frames]
        logger.info(f"Limited to {args.max_frames} frames")
    
    # Fetch and save each frame
    manifest = {
        "datastoreId": args.datastore_id,
        "imageSetId": args.image_set_id,
        "totalFrames": len(frame_ids),
        "frames": [],
    }
    
    for i, frame_info in enumerate(frame_ids):
        try:
            logger.info(f"Fetching frame {i+1}/{len(frame_ids)}: {frame_info['frameId']}")
            
            frame_data = fetch_image_frame(
                client,
                args.datastore_id,
                args.image_set_id,
                frame_info["frameId"],
            )
            
            filepath = save_frame(frame_data, output_path, frame_info)
            
            manifest["frames"].append({
                **frame_info,
                "filename": filepath.name,
                "size": len(frame_data),
            })
            
        except Exception as e:
            logger.error(f"Failed to fetch frame {frame_info['frameId']}: {e}")
            manifest["frames"].append({
                **frame_info,
                "error": str(e),
            })
    
    # Save manifest
    save_manifest(output_path, manifest)
    
    logger.info(f"Fetch complete. {len(manifest['frames'])} frames processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
