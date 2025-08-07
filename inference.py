import datetime
import glob
import os

import requests
import valohai
from PIL import Image
from ultralytics import YOLO
from whylabs_client import ApiClient, Configuration
from whylabs_client.api import dataset_profile_api
from whylabs_client.model.reference_profile_item_response import ReferenceProfileItemResponse
from whylogs.api.writer.whylabs import WhyLabsWriter
from whylogs.core import DatasetProfileView
from whylogs.extras.image_metric import log_image
from whylogs.viz import NotebookProfileVisualizer
from whylogs.viz.drift.column_drift_algorithms import calculate_drift_scores

from helpers import unpack_dataset


def inference_yolo(data_path):
    model_path = valohai.inputs("model").path()
    model = YOLO(model_path)

    # Make sure that we get all the images from the folder
    jpg_paths = glob.glob(f"{data_path}/*.jpg")
    png_paths = glob.glob(f"{data_path}/*.png")

    data_list = jpg_paths + png_paths

    # Run batched inference on a list of images
    results = model(data_list, save=True)  # return a list of Results
    inference_profile = None
    for res in results:
        path = res.path
        img = res.orig_img
        # Log to WhyLabs
        date = datetime.datetime.now(datetime.timezone.utc)
        pil = Image.fromarray(img)
        profile = log_image(pil).profile()
        profile.set_dataset_timestamp(date)

        profile_view = profile.view()
        if inference_profile is None:
            inference_profile = profile_view
        else:
            inference_profile = inference_profile.merge(profile_view)

        if valohai.parameters("save_results").value:
            # Save the result to Valohai
            image_name = os.path.basename(path)
            im_array = res.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            out_path = valohai.outputs().path(image_name[:-4] + "_result.jpg")
            im.save(out_path)  # save imag

    print(f"Inference profile {len(results)} images")
    writer = WhyLabsWriter()
    writer.write(inference_profile)
    return inference_profile


def load_reference_data_from_inputs():
    print("----Loading reference data from inputs 'ref_data'...")

    jpg_paths = glob.glob("/valohai/inputs/ref_data/*.jpg")
    png_paths = glob.glob("/valohai/inputs/ref_data/*.png")
    data_list = jpg_paths + png_paths

    reference_profile = None

    for file_path in data_list:
        date = datetime.datetime.now(datetime.timezone.utc)

        img = Image.open(file_path)
        profile = log_image(img).profile()
        profile.set_dataset_timestamp(date)
        profile_view = profile.view()

        # merge each profile while looping
        if reference_profile is None:
            reference_profile = profile_view
        else:
            reference_profile = reference_profile.merge(profile_view)

    print(f"Reference profile {len(data_list)} images is created")
    writer = WhyLabsWriter()
    writer.option(reference_profile_name=valohai.parameters("reference_profile_output_name").value)
    writer.write(reference_profile)
    return reference_profile


def load_reference_data_from_whylabs():
    print("----Loading reference data from WhyLabs...")
    # Load necessary values from environment variables
    ORG_ID = os.getenv("WHYLABS_DEFAULT_ORG_ID")
    MODEL_ID = os.getenv("WHYLABS_DEFAULT_DATASET_ID")
    API_KEY = os.getenv("WHYLABS_API_KEY")
    REF_ID = os.getenv("WHYLABS_REF_ID")

    if not all([ORG_ID, MODEL_ID, API_KEY, REF_ID]):
        raise EnvironmentError(
            "Missing one or more required environment variables: "
            "WHYLABS_DEFAULT_ORG_ID, WHYLABS_DEFAULT_DATASET_ID, WHYLABS_API_KEY, REF_ID",
        )

    # Configure the API client
    configuration = Configuration(
        host="https://api.whylabsapp.com",
    )
    configuration.api_key["ApiKeyAuth"] = API_KEY

    # Enter a context with an instance of the API client
    with ApiClient(configuration) as api_client:
        # Create an instance of the DatasetProfileApi class
        api_instance = dataset_profile_api.DatasetProfileApi(api_client)

        try:
            # Call the API to get the reference profile metadata
            api_response: ReferenceProfileItemResponse = api_instance.get_reference_profile(
                org_id=ORG_ID,
                model_id=MODEL_ID,
                reference_id=REF_ID,
            )

            # Download the profile from the provided URL
            download_url = api_response.download_url
            response = requests.get(download_url)

            if response.status_code == 200:
                # Deserialize the binary content into a DatasetProfileView
                reference_profile = DatasetProfileView.deserialize(response.content)
                print(f"Successfully loaded reference profile: {REF_ID}")
                return reference_profile
            else:
                raise Exception(f"Failed to download reference profile: {response.status_code} {response.text}")

        except Exception as e:
            print(f"Exception when calling DatasetProfileApi->get_reference_profile: {e}")
            raise


def generate_data_drift_report(inference_profile, reference_profile):
    visualization = NotebookProfileVisualizer()
    visualization.set_profiles(
        target_profile_view=inference_profile,
        reference_profile_view=reference_profile,
    )

    # generate and save the drift report
    report = visualization.summary_drift_report()
    visualization.write(
        report,
        preferred_path="/valohai/outputs/",
        html_file_name="summary_drift_report",
    )
    print("----Saved Generated Report to /valohai/outputs/")

    scores = calculate_drift_scores(
        target_view=inference_profile,
        reference_view=reference_profile,
        with_thresholds=True,
    )
    print_report_results(scores)


def print_report_results(scores):
    feature_category_dict = {
        key: value.get("drift_category", None) if isinstance(value, dict) else value for key, value in scores.items()
    }
    print("\n---Feature - Category report results")
    for key, value in feature_category_dict.items():
        print(f"{key}: {value}")

    drift_counts = {"NO_DRIFT": 0, "POSSIBLE_DRIFT": 0, "DRIFT": 0}
    for value in scores.values():
        if isinstance(value, dict) and "drift_category" in value:
            drift_category = value["drift_category"]
            drift_counts[drift_category] += 1

    print("\n---Counts of drift types: ")
    for key, value in drift_counts.items():
        with valohai.metadata.logger() as logger:  # Log to Valohai
            logger.log(key.lower(), value)
        print(f"{key}: {value}")

    if drift_counts["DRIFT"] > 0:
        valohai.set_status_detail("Drift Detected")


if __name__ == "__main__":
    # Get dataset for the inference
    dataset_packed = valohai.inputs("data").path(process_archives=False)
    data_path = "/valohai/repository/data"
    unpack_dataset(dataset_packed, data_path)

    print("----Running YOLO inference")
    inference_profile = inference_yolo(data_path)

    if valohai.parameters("use_whylabs_reference_profile").value:
        reference_profile = load_reference_data_from_whylabs()
    else:
        reference_profile = load_reference_data_from_inputs()

    generate_data_drift_report(inference_profile, reference_profile)
