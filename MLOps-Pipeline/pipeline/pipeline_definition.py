"""
SageMaker Pipeline Definition for AWS HealthImaging MLOps

This module defines a SageMaker Pipeline that:
1. Fetches pixel data from AWS HealthImaging using a custom container
2. Runs a placeholder training step (to be replaced with actual ML training)

The pipeline is triggered by EventBridge when new ImageSets are created.
"""

import os
from typing import Optional

import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.sklearn.estimator import SKLearn


def create_pipeline(
    role: str,
    pipeline_name: str,
    fetcher_image_uri: str,
    default_bucket: str,
    region: str,
    processing_instance_type: str = "ml.m5.large",
    training_instance_type: str = "ml.m5.large",
) -> Pipeline:
    """
    Create a SageMaker Pipeline for HealthImaging MLOps.
    
    Args:
        role: SageMaker execution role ARN
        pipeline_name: Name for the pipeline
        fetcher_image_uri: ECR URI for the fetcher container
        default_bucket: S3 bucket for pipeline artifacts
        region: AWS region
        processing_instance_type: Instance type for processing jobs
        training_instance_type: Instance type for training jobs
    
    Returns:
        SageMaker Pipeline object
    """
    
    # ==========================================================================
    # Pipeline Parameters
    # ==========================================================================
    
    # These parameters can be overridden at pipeline execution time
    image_set_id = ParameterString(
        name="ImageSetId",
        default_value="",
    )
    
    datastore_id = ParameterString(
        name="DatastoreId",
        default_value="",
    )
    
    # ==========================================================================
    # Step 1: Fetcher Processing Step
    # ==========================================================================
    
    # Create processor using the custom fetcher container
    fetcher_processor = Processor(
        role=role,
        image_uri=fetcher_image_uri,
        instance_count=1,
        instance_type=processing_instance_type,
        base_job_name="ahi-fetcher",
        env={
            "AWS_REGION": region,
        },
    )
    
    # Define the processing step
    fetcher_step = ProcessingStep(
        name="FetchImageData",
        processor=fetcher_processor,
        outputs=[
            ProcessingOutput(
                output_name="output",
                source="/opt/ml/processing/output",
                destination=f"s3://{default_bucket}/pipeline-output/fetched-data",
            ),
        ],
        job_arguments=[
            "--datastore-id", datastore_id,
            "--image-set-id", image_set_id,
            "--output-dir", "/opt/ml/processing/output",
            "--region", region,
        ],
    )
    
    # ==========================================================================
    # Step 2: Dummy Training Step
    # ==========================================================================
    
    # Create a simple sklearn estimator as a placeholder
    # Replace this with your actual training logic
    dummy_estimator = SKLearn(
        entry_point="dummy_train.py",
        source_dir=os.path.join(os.path.dirname(__file__), "training_scripts"),
        role=role,
        instance_count=1,
        instance_type=training_instance_type,
        framework_version="1.2-1",
        base_job_name="ahi-training",
        hyperparameters={
            "epochs": 1,
        },
    )
    
    training_step = TrainingStep(
        name="TrainModel",
        estimator=dummy_estimator,
        inputs={
            "training": sagemaker.inputs.TrainingInput(
                s3_data=fetcher_step.properties.ProcessingOutputConfig.Outputs[
                    "output"
                ].S3Output.S3Uri,
                content_type="application/octet-stream",
            ),
        },
    )
    
    # ==========================================================================
    # Create Pipeline
    # ==========================================================================
    
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            image_set_id,
            datastore_id,
        ],
        steps=[
            fetcher_step,
            training_step,
        ],
    )
    
    return pipeline


def get_pipeline(
    role: str,
    pipeline_name: str,
    fetcher_image_uri: str,
    default_bucket: str,
    region: str,
) -> Pipeline:
    """
    Convenience function to get a configured pipeline.
    
    This is the main entry point for creating the pipeline.
    """
    return create_pipeline(
        role=role,
        pipeline_name=pipeline_name,
        fetcher_image_uri=fetcher_image_uri,
        default_bucket=default_bucket,
        region=region,
    )
