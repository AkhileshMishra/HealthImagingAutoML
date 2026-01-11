# MLOps Pipeline for AWS HealthImaging

This module provides an end-to-end MLOps pipeline that automatically triggers machine learning workflows when new medical images are imported into AWS HealthImaging.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│ AWS HealthImaging│────▶│   EventBridge    │────▶│  SageMaker Pipeline │
│  (Image Import)  │     │  (Event Rule)    │     │                     │
└─────────────────┘     └──────────────────┘     │  ┌───────────────┐  │
                                                  │  │ Fetcher Step  │  │
                                                  │  │ (Processing)  │  │
                                                  │  └───────┬───────┘  │
                                                  │          │          │
                                                  │  ┌───────▼───────┐  │
                                                  │  │ Training Step │  │
                                                  │  │   (Dummy)     │  │
                                                  │  └───────────────┘  │
                                                  └─────────────────────┘
```

## Directory Structure

```
MLOps-Pipeline/
├── README_MLOPS.md           # This file
├── app.py                    # CDK application entry point
├── requirements.txt          # Python dependencies
├── cdk.json                  # CDK configuration
├── config.py                 # Configuration parameters
├── fetcher_image/
│   ├── Dockerfile            # Container definition for fetcher
│   ├── fetch.py              # Script to retrieve pixel data from HealthImaging
│   └── requirements.txt      # Fetcher container dependencies
└── pipeline/
    └── pipeline_definition.py # SageMaker Pipeline definition
```

## Prerequisites

1. **AWS CLI** configured with appropriate credentials
2. **AWS CDK v2** installed (`npm install -g aws-cdk`)
3. **Python 3.9+** with pip
4. **Docker** installed and running
5. **Existing AWS HealthImaging Datastore** with imported images

## Configuration

Edit `config.py` to set your environment-specific values:

```python
# Required
DEPLOY_REGION = "us-east-1"
DATASTORE_ID = "your-datastore-id"  # Your HealthImaging datastore ID

# Optional
SAGEMAKER_BUCKET_NAME = ""  # Leave empty to create new bucket
PIPELINE_NAME = "ahi-mlops-pipeline"
```

## Deployment Steps

### 1. Set up Python Environment

```bash
cd MLOps-Pipeline
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Bootstrap CDK (if first time)

```bash
cdk bootstrap
```

### 3. Deploy the Stack

```bash
cdk deploy
```

### 4. Verify Deployment

After deployment, you'll see outputs including:
- SageMaker Pipeline ARN
- EventBridge Rule ARN
- S3 Bucket for pipeline artifacts

## Testing the Pipeline

### Manual Trigger

You can manually start the pipeline with a specific ImageSet ID:

```bash
aws sagemaker start-pipeline-execution \
    --pipeline-name ahi-mlops-pipeline \
    --pipeline-parameters '[{"Name":"ImageSetId","Value":"your-image-set-id"},{"Name":"DatastoreId","Value":"your-datastore-id"}]'
```

### Automatic Trigger

Import a DICOM study into your HealthImaging datastore. The EventBridge rule will automatically detect the new ImageSet and trigger the pipeline.

## IAM Permissions

The stack creates the following IAM roles:

### SageMaker Execution Role
- `medical-imaging:GetImageSet` - Read ImageSet metadata
- `medical-imaging:GetImageFrame` - Retrieve pixel data
- `s3:GetObject`, `s3:PutObject` - Pipeline artifacts
- `ecr:GetDownloadUrlForLayer`, `ecr:BatchGetImage` - Pull container images

### EventBridge Role
- `sagemaker:StartPipelineExecution` - Trigger the pipeline

## Pipeline Steps

### Step 1: Fetcher (ProcessingStep)
- Runs the custom fetcher container
- Retrieves pixel data from AWS HealthImaging
- Outputs data to S3 for downstream processing

### Step 2: Training (Dummy)
- Placeholder training step
- Replace with your actual ML training logic

## Extending the Pipeline

### Adding Real Training

Replace the dummy training step in `pipeline/pipeline_definition.py`:

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    source_dir="training_scripts",
    role=role,
    instance_count=1,
    instance_type="ml.p3.2xlarge",
    framework_version="2.0",
    py_version="py310",
)

training_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "training": TrainingInput(
            s3_data=fetcher_step.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri
        )
    }
)
```

### Adding Model Registration

```python
from sagemaker.workflow.model_step import ModelStep

model_step = ModelStep(
    name="RegisterModel",
    step_args=model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        model_package_group_name="ahi-models",
    )
)
```

## Cleanup

```bash
cdk destroy
```

## Troubleshooting

### Pipeline Not Triggering
1. Verify EventBridge rule is enabled in the console
2. Check CloudWatch Logs for the EventBridge rule
3. Ensure the HealthImaging event pattern matches

### Fetcher Container Fails
1. Check SageMaker Processing job logs in CloudWatch
2. Verify IAM permissions for HealthImaging access
3. Ensure the ImageSet ID and Datastore ID are valid

### Permission Denied Errors
1. Verify the SageMaker execution role has the required policies
2. Check if the HealthImaging datastore allows access from the role

## Integration with Existing Components

### TLM Proxy
The fetcher can be modified to use the TLM Proxy for progressive image loading:
```python
# In fetch.py, replace direct HealthImaging calls with TLM Proxy calls
TLM_PROXY_URL = os.environ.get("TLM_PROXY_URL")
```

### S3 StoreSCP
Configure the S3 StoreSCP to write to a bucket that triggers HealthImaging import jobs, creating a complete ingestion-to-ML pipeline.
