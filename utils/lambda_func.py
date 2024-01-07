import json
import boto3
import base64

endpoint_name = "demo-img-clsfr-pneuomia2"

sagemaker_runtime_client = boto3.client("runtime.sagemaker")


def lambda_handler(event, context):
    """
    AWS Lambda handler function for predicting pneumonia presence in an image.

    This function is triggered by an AWS Lambda event. It decodes the base64-encoded
    image contained within the event, logs the image data, and calls the 
    `_predict_pneumonia` function to obtain a prediction. It's designed to be used
    as an entry point for AWS Lambda to handle requests.

    Parameters:
    event (dict): The event dictionary containing the image data under the "image" key.
                  The image data should be base64-encoded.
    context: The runtime information provided by AWS Lambda (unused in this function).

    Returns:
    str: The result of the pneumonia prediction as returned by `_predict_pneumonia`.
    """
    print(event) # for Cloudwatch logs
    
    image = base64.b64decode(event["image"])
    print(image) # for Cloudwatch logs
    
    return _predict_pneumonia(image)



def _predict_pneumonia(image):
    """
    Invokes a SageMaker endpoint to predict the presence of pneumonia from an image.

    This function sends an image to a pre-trained model hosted on a SageMaker endpoint.
    The model predicts whether the image indicates the presence of pneumonia. It returns
    a string indicating the prediction and the probability of the prediction.

    Parameters:
    image (bytes): The image to be analyzed, passed as a byte array.

    Returns:
    str: A string indicating whether pneumonia is predicted to be present or not,
         along with the probability of that prediction.
    """