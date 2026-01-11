# MLOps Pipeline Configuration
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Configuration parameters for the MLOps Pipeline CDK stack.
Update these values according to your environment.
"""

# =============================================================================
# DEPLOYMENT CONFIGURATION
# =============================================================================

# AWS region for deployment
DEPLOY_REGION = "us-east-1"

# Stack name
STACK_NAME = "AHI-MLOps-Pipeline"

# =============================================================================
# AWS HEALTHIMAGING CONFIGURATION
# =============================================================================

# Your HealthImaging datastore ID (required for EventBridge filtering)
# Leave empty to trigger on ALL datastores in the account
DATASTORE_ID = ""

# =============================================================================
# SAGEMAKER CONFIGURATION
# =============================================================================

# SageMaker Pipeline name
PIPELINE_NAME = "ahi-mlops-pipeline"

# S3 bucket for SageMaker artifacts (leave empty to create new bucket)
SAGEMAKER_BUCKET_NAME = ""

# Processing job instance type
PROCESSING_INSTANCE_TYPE = "ml.m5.large"

# Training job instance type (for future use)
TRAINING_INSTANCE_TYPE = "ml.m5.large"

# =============================================================================
# FETCHER CONTAINER CONFIGURATION
# =============================================================================

# Maximum number of frames to fetch per ImageSet (0 = all frames)
MAX_FRAMES_PER_IMAGESET = 0

# Output format for fetched frames: "raw" or "numpy"
OUTPUT_FORMAT = "raw"
