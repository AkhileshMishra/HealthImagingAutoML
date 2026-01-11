# End-to-End Medical Imaging MLOps Platform - Deployment Guide

This guide walks through deploying the complete AWS HealthImaging MLOps platform, including the existing streaming layer, ingestion services, and the new MLOps pipeline.

## Prerequisites

- AWS CLI configured with appropriate credentials
- AWS CDK v2 (`npm install -g aws-cdk`)
- Node.js 18+ and npm
- Python 3.9+ with pip
- Docker installed and running
- An AWS account with HealthImaging enabled

---

## Phase 1: High-Speed Streaming Layer (Existing Code)

### 1.1 Deploy TLM Proxy Backend

The Tile Level Marker (TLM) Proxy enables progressive image loading from AWS HealthImaging.

```bash
# Navigate to the TLM Proxy directory
cd tile-level-marker-proxy

# Install dependencies
npm install

# Configure the deployment (edit config.ts)
# Required: Set ACM_ARN and COGNITO_USER_POOL_ID
```

**Edit `tile-level-marker-proxy/config.ts`:**
```typescript
const DEPLOY_REGION: string = 'us-east-1';
const ACM_ARN: string = 'arn:aws:acm:us-east-1:ACCOUNT:certificate/CERT-ID';
const COGNITO_USER_POOL_ID: NullableString = 'us-east-1_XXXXXXXXX';
```

```bash
# Bootstrap CDK (first time only)
cdk bootstrap

# Deploy the stack
cdk deploy

# Note the ALB URL from the output:
# TileLevelMarkerProxyStack.LoadBalancerDNS = xxx.us-east-1.elb.amazonaws.com
```

**Output to capture:** `LoadBalancerDNS` - This is your TLM Proxy API endpoint.

### 1.2 Deploy Frontend Viewer (OHIF-based)

The imaging viewer UI is deployed via AWS Amplify.

```bash
# Navigate to the viewer directory
cd imaging-viewer-ui

# Install dependencies
npm install

# Initialize Amplify (if not already done)
amplify init
# - Enter environment name: dev
# - Select default editor
# - Choose AWS profile

# Add hosting
amplify add hosting
# - Select: Amazon CloudFront and S3
# - Select: PROD (S3 with CloudFront using HTTPS)

# Deploy the full-stack app
amplify publish
```

**Alternative: One-Click Deploy**
Use the Amplify button in the README for automatic deployment via AWS Console.

**Configure TLM Proxy URL:**
After deployment, log into the viewer and navigate to Settings to configure:
- TLM Proxy URL: `https://your-tlm-proxy-url`
- HealthImaging Region: `us-east-1`

---

## Phase 2: Ingestion Layer (Existing Code)

### 2.1 Deploy S3 StoreSCP Service

The S3 StoreSCP service receives DICOM via DIMSE and stores to S3.

```bash
# Navigate to the ingestion directory
cd s3-storescp

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Edit `s3-storescp/cfg.py` for HealthImaging compatibility:**
```python
# Required settings for HealthImaging import
ENVIRONMENT = {
    "CREATE_METADATA": False,      # HealthImaging only needs P10 files
    "GZIP_FILES": False,           # No gzip for HealthImaging
    "ADD_STUDYUID_PREFIX": False,  # Flat structure required
    # ... other settings
}
```

```bash
# Deploy the stack
cdk synth
cdk deploy

# Note the NLB DNS from output for DICOM sending
```

### 2.2 Verify IAM Permissions

The S3 StoreSCP task role needs permissions to trigger HealthImaging imports. Add this policy to the ECS task role if integrating with HealthImaging import jobs:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "medical-imaging:StartDICOMImportJob",
                "medical-imaging:GetDICOMImportJob"
            ],
            "Resource": "*"
        }
    ]
}
```

---

## Phase 3: MLOps Pipeline (New Implementation)

### 3.1 Deploy the MLOps Stack

```bash
# Navigate to the MLOps Pipeline directory
cd MLOps-Pipeline

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Edit `MLOps-Pipeline/config.py`:**
```python
DEPLOY_REGION = "us-east-1"
DATASTORE_ID = "your-healthimaging-datastore-id"  # Optional: filter events
PIPELINE_NAME = "ahi-mlops-pipeline"
```

```bash
# Bootstrap CDK (if not done)
cdk bootstrap

# Deploy the MLOps stack
cdk deploy

# Capture outputs:
# - SageMakerBucketName
# - FetcherImageUri
# - SageMakerRoleArn
# - CreatePipelineCommand
```

### 3.2 Create the SageMaker Pipeline

After CDK deployment, run the pipeline creation command from the output:

```bash
cd MLOps-Pipeline

python -c "
from pipeline.pipeline_definition import get_pipeline
pipeline = get_pipeline(
    role='arn:aws:iam::ACCOUNT:role/AHI-MLOps-Pipeline-SageMakerExecutionRole',
    pipeline_name='ahi-mlops-pipeline',
    fetcher_image_uri='ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/...',
    default_bucket='ahi-mlops-pipeline-sagemakerbucket-xxx',
    region='us-east-1',
)
pipeline.upsert(role_arn='arn:aws:iam::ACCOUNT:role/AHI-MLOps-Pipeline-SageMakerExecutionRole')
print('Pipeline created successfully!')
"
```

### 3.3 Test the Pipeline Manually

```bash
# Start a pipeline execution with test parameters
aws sagemaker start-pipeline-execution \
    --pipeline-name ahi-mlops-pipeline \
    --pipeline-parameters '[
        {"Name":"ImageSetId","Value":"your-test-imageset-id"},
        {"Name":"DatastoreId","Value":"your-datastore-id"}
    ]' \
    --region us-east-1
```

---

## Phase 4: Permissions & Wiring

### 4.1 SageMaker Execution Role Permissions

The CDK stack automatically creates a role with these permissions:
- `medical-imaging:GetImageSet`
- `medical-imaging:GetImageSetMetadata`
- `medical-imaging:GetImageFrame`
- `medical-imaging:ListImageSetVersions`
- S3 read/write to the SageMaker bucket
- ECR pull for the fetcher container

### 4.2 EventBridge Role Permissions

The EventBridge role has:
- `sagemaker:StartPipelineExecution` for the specific pipeline

### 4.3 Verify Event Flow

1. **Check EventBridge Rule:**
   ```bash
   aws events describe-rule --name ahi-imageset-created-trigger --region us-east-1
   ```

2. **Check Lambda Trigger:**
   ```bash
   aws lambda get-function --function-name AHI-MLOps-Pipeline-TriggerPipelineLambda --region us-east-1
   ```

3. **Monitor CloudWatch Logs:**
   - Lambda logs: `/aws/lambda/AHI-MLOps-Pipeline-TriggerPipelineLambda`
   - SageMaker Processing logs: `/aws/sagemaker/ProcessingJobs`

---

## End-to-End Test

### 1. Send DICOM to S3 StoreSCP

```bash
# Using dcm4che storescu
storescu -c S3SCP@<NLB-DNS>:11113 --tls /path/to/dicom/files/
```

### 2. Trigger HealthImaging Import

```bash
aws medical-imaging start-dicom-import-job \
    --datastore-id <datastore-id> \
    --data-access-role-arn <import-role-arn> \
    --input-s3-uri s3://<receive-bucket>/DICOM/ \
    --output-s3-uri s3://<receive-bucket>/import-output/
```

### 3. Verify Pipeline Execution

```bash
# List recent pipeline executions
aws sagemaker list-pipeline-executions \
    --pipeline-name ahi-mlops-pipeline \
    --region us-east-1

# Describe a specific execution
aws sagemaker describe-pipeline-execution \
    --pipeline-execution-arn <execution-arn>
```

### 4. View Results in Viewer

Open the Amplify-hosted viewer URL and browse the imported ImageSets.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Medical Imaging MLOps Platform                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐  │
│  │   DICOM      │    │  S3 StoreSCP │    │    AWS HealthImaging         │  │
│  │   Source     │───▶│  (Fargate)   │───▶│    (Import Job)              │  │
│  └──────────────┘    └──────────────┘    └──────────────┬───────────────┘  │
│                                                          │                   │
│                                                          ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐  │
│  │   OHIF       │◀───│  TLM Proxy   │◀───│      EventBridge             │  │
│  │   Viewer     │    │  (Fargate)   │    │   (ImageSet Created)         │  │
│  └──────────────┘    └──────────────┘    └──────────────┬───────────────┘  │
│                                                          │                   │
│                                                          ▼                   │
│                                          ┌──────────────────────────────┐   │
│                                          │    SageMaker Pipeline        │   │
│                                          │  ┌────────────────────────┐  │   │
│                                          │  │ Fetcher (Processing)   │  │   │
│                                          │  └───────────┬────────────┘  │   │
│                                          │              │               │   │
│                                          │  ┌───────────▼────────────┐  │   │
│                                          │  │ Training (ML Model)    │  │   │
│                                          │  └────────────────────────┘  │   │
│                                          └──────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Cleanup

```bash
# Remove MLOps Pipeline
cd MLOps-Pipeline && cdk destroy

# Remove S3 StoreSCP
cd s3-storescp && cdk destroy

# Remove TLM Proxy
cd tile-level-marker-proxy && cdk destroy --all

# Remove Amplify App (via console or CLI)
amplify delete
```

---

## Troubleshooting

### Pipeline Not Triggering
- Verify EventBridge rule is enabled
- Check CloudWatch Logs for the trigger Lambda
- Ensure HealthImaging events are being emitted (check CloudTrail)

### Fetcher Container Fails
- Check SageMaker Processing job logs
- Verify IAM permissions for HealthImaging access
- Ensure ImageSet ID and Datastore ID are valid

### TLM Proxy Connection Issues
- Verify ACM certificate is valid and in the correct region
- Check Cognito user pool configuration
- Verify security group allows inbound traffic

### Import Job Failures
- Check the import job output S3 location for error details
- Verify DICOM files are valid P10 format
- Ensure IAM role has correct permissions
