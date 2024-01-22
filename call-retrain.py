import os

import requests

api_token = os.environ.get("VALOHAI_API_TOKEN")
resp = requests.request(
    url="https://app.valohai.com/api/v0/pipelines/",
    method="POST",
    headers={"Authorization": f"Token {api_token}"},
    json={
        "edges": [
            {
                "source_node": "preprocess",
                "source_key": "image_size",
                "source_type": "parameter",
                "target_node": "training",
                "target_type": "parameter",
                "target_key": "image_size",
            },
            {
                "source_node": "training",
                "source_key": "*best.pt",
                "source_type": "output",
                "target_node": "evaluation",
                "target_type": "input",
                "target_key": "model",
            },
        ],
        "nodes": [
            {
                "name": "preprocess",
                "type": "execution",
                "template": {
                    "environment": "017d329a-e56f-d451-040b-ccb895513c40",
                    "commit": "0aadea924a866f2fd95b5ea304c2a8f7fd0cc189",
                    "step": "prepare_data",
                    "image": "ultralytics/yolov5",
                    "command": "pip install valohai-utils\npython preprocess.py {parameters}",
                    "inputs": {
                        "dataset": [
                            "s3://valohai-demo-library-data/drift-detection/ships-aerial-images.zip",
                        ],
                    },
                    "parameters": {
                        "train_size": 15,
                        "valid_size": 10,
                        "test_size": 10,
                        "image_size": 768,
                    },
                    "runtime_config": {},
                    "runtime_config_preset": "",
                    "inherit_environment_variables": True,
                    "environment_variable_groups": [],
                    "tags": [],
                    "time_limit": 0,
                    "environment_variables": {},
                },
                "on_error": "stop-all",
            },
            {
                "name": "training",
                "type": "execution",
                "template": {
                    "environment": "017d329a-e56f-d451-040b-ccb895513c40",
                    "commit": "0aadea924a866f2fd95b5ea304c2a8f7fd0cc189",
                    "step": "train_model",
                    "image": "ultralytics/yolov5",
                    "command": "pip install valohai-utils\npython train.py {parameters}",
                    "inputs": {
                        "train": ["dataset://drift-demo-ships-aerial/dev_train"],
                        "test": ["dataset://drift-demo-ships-aerial/dev_test"],
                        "valid": ["dataset://drift-demo-ships-aerial/dev_valid"],
                    },
                    "parameters": {
                        "yolo_model_name": "yolov8x.pt",
                        "epochs": 3,
                        "batch_size": 8,
                        "image_size": 768,
                        "optimizer": "SGD",
                        "seed": 42,
                        "project": "/valohai/outputs/",
                    },
                    "runtime_config": {},
                    "runtime_config_preset": "",
                    "inherit_environment_variables": True,
                    "environment_variable_groups": [],
                    "tags": [],
                    "time_limit": 0,
                    "environment_variables": {},
                },
                "on_error": "stop-all",
            },
            {
                "name": "evaluation",
                "type": "execution",
                "template": {
                    "environment": "017d329a-e56f-d451-040b-ccb895513c40",
                    "commit": "0aadea924a866f2fd95b5ea304c2a8f7fd0cc189",
                    "step": "evaluation",
                    "image": "ultralytics/yolov5",
                    "command": "pip install valohai-utils\npython evaluation.py {parameters}",
                    "inputs": {
                        "model": ["datum://model-current-best"],
                        "data_yaml": ["datum://data_yaml"],
                        "valid": ["dataset://drift-demo-ships-aerial/dev_valid"],
                        "test": ["dataset://drift-demo-ships-aerial/dev_test"],
                    },
                    "parameters": {},
                    "runtime_config": {},
                    "runtime_config_preset": "",
                    "inherit_environment_variables": True,
                    "environment_variable_groups": [],
                    "tags": [],
                    "time_limit": 0,
                    "environment_variables": {},
                },
                "on_error": "stop-all",
            },
        ],
        "project": "018c8779-9475-09e1-d481-e295ab4de428",
        "tags": [],
        "parameters": {},
        "title": "train-val-pipeline",
    },
)
if resp.status_code == 400:
    raise RuntimeError(resp.json())
resp.raise_for_status()
data = resp.json()
