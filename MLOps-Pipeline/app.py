#!/usr/bin/env python3
"""
AWS HealthImaging MLOps Pipeline - CDK Application

This CDK application deploys:
1. ECR repository for the fetcher container
2. S3 bucket for SageMaker artifacts
3. SageMaker Pipeline for ML workflows
4. EventBridge rule to trigger pipeline on new ImageSets
5. IAM roles with appropriate permissions

Usage:
    cdk deploy
"""

import os
from constructs import Construct
from aws_cdk import (
    App,
    CfnOutput,
    Duration,
    Environment,
    RemovalPolicy,
    Stack,
    aws_ecr as ecr,
    aws_ecr_assets as ecr_assets,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_s3 as s3,
    aws_sagemaker as sagemaker,
    CustomResource,
    custom_resources as cr,
    aws_lambda as lambda_,
)

import config


class MLOpsPipelineStack(Stack):
    """CDK Stack for AWS HealthImaging MLOps Pipeline."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        region = self.region
        account = self.account

        # =====================================================================
        # S3 Bucket for SageMaker Artifacts
        # =====================================================================
        
        if config.SAGEMAKER_BUCKET_NAME:
            sagemaker_bucket = s3.Bucket.from_bucket_name(
                self, "SageMakerBucket",
                bucket_name=config.SAGEMAKER_BUCKET_NAME,
            )
        else:
            sagemaker_bucket = s3.Bucket(
                self, "SageMakerBucket",
                encryption=s3.BucketEncryption.S3_MANAGED,
                removal_policy=RemovalPolicy.DESTROY,
                auto_delete_objects=True,
                block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
                enforce_ssl=True,
            )

        # =====================================================================
        # ECR Repository and Docker Image for Fetcher
        # =====================================================================
        
        fetcher_image = ecr_assets.DockerImageAsset(
            self, "FetcherImage",
            directory="fetcher_image",
            platform=ecr_assets.Platform.LINUX_AMD64,
        )

        # =====================================================================
        # SageMaker Execution Role
        # =====================================================================
        
        sagemaker_role = iam.Role(
            self, "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                ),
            ],
        )

        # HealthImaging permissions
        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "medical-imaging:GetImageSet",
                    "medical-imaging:GetImageSetMetadata",
                    "medical-imaging:GetImageFrame",
                    "medical-imaging:ListImageSetVersions",
                ],
                resources=["*"],  # Scope down to specific datastore if needed
            )
        )

        # S3 permissions for the SageMaker bucket
        sagemaker_bucket.grant_read_write(sagemaker_role)

        # ECR permissions for pulling the fetcher image
        fetcher_image.repository.grant_pull(sagemaker_role)

        # =====================================================================
        # SageMaker Pipeline
        # =====================================================================
        
        # Pipeline definition JSON - we'll create this via a custom resource
        # that invokes the SageMaker SDK to create/update the pipeline
        
        pipeline_name = config.PIPELINE_NAME
        
        # Create a Lambda function to manage the SageMaker Pipeline
        pipeline_manager_role = iam.Role(
            self, "PipelineManagerRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
            ],
        )
        
        pipeline_manager_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "sagemaker:CreatePipeline",
                    "sagemaker:UpdatePipeline",
                    "sagemaker:DeletePipeline",
                    "sagemaker:DescribePipeline",
                    "iam:PassRole",
                ],
                resources=["*"],
            )
        )

        # For simplicity, we'll output the pipeline creation command
        # In production, use a Custom Resource with Lambda

        # =====================================================================
        # EventBridge Rule for HealthImaging Events
        # =====================================================================
        
        # EventBridge role for triggering SageMaker Pipeline
        eventbridge_role = iam.Role(
            self, "EventBridgeRole",
            assumed_by=iam.ServicePrincipal("events.amazonaws.com"),
        )

        eventbridge_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["sagemaker:StartPipelineExecution"],
                resources=[
                    f"arn:aws:sagemaker:{region}:{account}:pipeline/{pipeline_name}",
                ],
            )
        )

        # Event pattern for HealthImaging ImageSet creation
        event_pattern = {
            "source": ["aws.medical-imaging"],
            "detail-type": ["AWS API Call via CloudTrail"],
            "detail": {
                "eventSource": ["medical-imaging.amazonaws.com"],
                "eventName": ["StartDICOMImportJob"],
            },
        }
        
        # Alternative: Use HealthImaging native events if available
        # This pattern catches import job completions
        event_pattern_native = events.EventPattern(
            source=["aws.medical-imaging"],
            detail_type=["HealthImaging ImageSet State Change"],
            detail={
                "state": ["ACTIVE"],
            },
        )

        # Create the EventBridge rule
        event_rule = events.Rule(
            self, "ImageSetCreatedRule",
            rule_name="ahi-imageset-created-trigger",
            description="Triggers MLOps pipeline when new ImageSets are created in HealthImaging",
            event_pattern=event_pattern_native,
        )

        # Add SageMaker Pipeline as target
        # Note: SageMaker Pipeline target requires the pipeline to exist first
        # We'll use a Lambda function as an intermediary for flexibility
        
        trigger_lambda_role = iam.Role(
            self, "TriggerLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                ),
            ],
        )
        
        trigger_lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["sagemaker:StartPipelineExecution"],
                resources=[
                    f"arn:aws:sagemaker:{region}:{account}:pipeline/{pipeline_name}",
                ],
            )
        )

        trigger_lambda = lambda_.Function(
            self, "TriggerPipelineLambda",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="index.handler",
            role=trigger_lambda_role,
            timeout=Duration.seconds(30),
            environment={
                "PIPELINE_NAME": pipeline_name,
            },
            code=lambda_.Code.from_inline("""
import boto3
import json
import os

def handler(event, context):
    print(f"Received event: {json.dumps(event)}")
    
    pipeline_name = os.environ['PIPELINE_NAME']
    sagemaker = boto3.client('sagemaker')
    
    # Extract ImageSet and Datastore IDs from the event
    detail = event.get('detail', {})
    image_set_id = detail.get('imageSetId', '')
    datastore_id = detail.get('datastoreId', '')
    
    if not image_set_id or not datastore_id:
        print("Missing imageSetId or datastoreId in event")
        return {'statusCode': 400, 'body': 'Missing required parameters'}
    
    # Start pipeline execution
    response = sagemaker.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineParameters=[
            {'Name': 'ImageSetId', 'Value': image_set_id},
            {'Name': 'DatastoreId', 'Value': datastore_id},
        ],
        PipelineExecutionDescription=f'Triggered by ImageSet {image_set_id}',
    )
    
    print(f"Started pipeline execution: {response['PipelineExecutionArn']}")
    return {'statusCode': 200, 'body': response['PipelineExecutionArn']}
"""),
        )

        # Add Lambda as EventBridge target
        event_rule.add_target(targets.LambdaFunction(trigger_lambda))

        # =====================================================================
        # Outputs
        # =====================================================================
        
        CfnOutput(
            self, "SageMakerBucketName",
            value=sagemaker_bucket.bucket_name,
            description="S3 bucket for SageMaker pipeline artifacts",
        )

        CfnOutput(
            self, "FetcherImageUri",
            value=fetcher_image.image_uri,
            description="ECR URI for the fetcher container image",
        )

        CfnOutput(
            self, "SageMakerRoleArn",
            value=sagemaker_role.role_arn,
            description="SageMaker execution role ARN",
        )

        CfnOutput(
            self, "PipelineName",
            value=pipeline_name,
            description="SageMaker Pipeline name",
        )

        CfnOutput(
            self, "EventBridgeRuleArn",
            value=event_rule.rule_arn,
            description="EventBridge rule ARN for ImageSet triggers",
        )

        # Output the command to create the SageMaker Pipeline
        CfnOutput(
            self, "CreatePipelineCommand",
            value=f"""
# Run this command after CDK deployment to create the SageMaker Pipeline:
cd MLOps-Pipeline && python -c "
from pipeline.pipeline_definition import get_pipeline
pipeline = get_pipeline(
    role='{sagemaker_role.role_arn}',
    pipeline_name='{pipeline_name}',
    fetcher_image_uri='{fetcher_image.image_uri}',
    default_bucket='{sagemaker_bucket.bucket_name}',
    region='{region}',
)
pipeline.upsert(role_arn='{sagemaker_role.role_arn}')
print('Pipeline created/updated successfully!')
"
""",
            description="Command to create the SageMaker Pipeline after CDK deployment",
        )


# =============================================================================
# App Entry Point
# =============================================================================

app = App()

MLOpsPipelineStack(
    app,
    config.STACK_NAME,
    description="AWS HealthImaging MLOps Pipeline - Automated ML workflows for medical imaging",
    env=Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION", config.DEPLOY_REGION),
    ),
)

app.synth()
