"""
Jupyter environment configuration
Run this cell at the start of your notebook
"""

import os

# SageMaker Configuration
SAGEMAKER_CONFIG = {
    "SAGEMAKER_ENDPOINT_NAME": "rfdetr-rfdetr-large-pytorch-2024-12-29-19-30-45-123456",
    "SAGEMAKER_ROLE": "arn:aws:iam::123456789012:role/SageMakerExecutionRole",
    "AWS_DEFAULT_REGION": "us-east-1",
    "RFDETR_VARIANT": "large",
    "RFDETR_CONF": "0.25",
}

# Set environment variables
for key, value in SAGEMAKER_CONFIG.items():
    os.environ[key] = value
    print(f"Set {key} = {value}")

print("✅ Environment configured!")
