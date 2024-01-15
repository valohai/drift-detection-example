import requests

resp = requests.request(
    url="https://staging.valohai.com/api/v0/pipelines/",
    method="POST",
    headers={"Authorization": "Token L2kXt2Frkj5NBItvUH92zhly7B2GjRp24nj2yLFv"},
    json={
        "edges": [
            {
                "source_node": "preprocess",
                "source_key": "image_size",
                "source_type": "parameter",
                "target_node": "training",
                "target_type": "parameter",
                "target_key": "image_size"
            },
            {
                "source_node": "training",
                "source_key": "*best.pt",
                "source_type": "output",
                "target_node": "evaluation",
                "target_type": "input",
                "target_key": "model"
            }
        ],
        "nodes": [
            {
                "name": "evaluation",
                "type": "execution",
                "template": {
                    "environment": "017d329a-e56f-d451-040b-ccb895513c40",
                    "commit": "~4bfc31b2e318f91a624078c87546f0c283f967b75dd5f7812f4f17ee7b9a8a5a",
                    "step": "evaluation",
                    "image": "ultralytics/yolov5",
                    "command": "pip install valohai-utils\npython evaluation.py {parameters}",
                    "inputs": {
                        "model": [
                            "datum://model-current-best"
                        ],
                        "data_yaml": [
                            "datum://data_yaml"
                        ],
                        "valid": [
                            "dataset://drift-demo-ships-aerial/dev_valid"
                        ],
                        "test": [
                            "dataset://drift-demo-ships-aerial/dev_test"
                        ]
                    },
                    "parameters": {},
                    "runtime_config": {},
                    "runtime_config_preset": "",
                    "inherit_environment_variables": True,
                    "environment_variable_groups": [],
                    "tags": [],
                    "time_limit": 0,
                    "environment_variables": {}
                },
                "on_error": "stop-all"
            },
            {
                "name": "preprocess",
                "type": "execution",
                "template": {
                    "environment": "017d329a-e56f-d451-040b-ccb895513c40",
                    "commit": "~4bfc31b2e318f91a624078c87546f0c283f967b75dd5f7812f4f17ee7b9a8a5a",
                    "step": "prepare_data",
                    "image": "ultralytics/yolov5",
                    "command": "pip install valohai-utils\npython preprocess.py {parameters}",
                    "inputs": {
                        "dataset": [
                            "s3://valohai-demo-library-data/drift-detection/ships-aerial-images.zip"
                        ]
                    },
                    "parameters": {
                        "train_size": 15,
                        "valid_size": 10,
                        "test_size": 10,
                        "image_size": 768
                    },
                    "runtime_config": {},
                    "runtime_config_preset": "",
                    "inherit_environment_variables": True,
                    "environment_variable_groups": [],
                    "tags": [],
                    "time_limit": 0,
                    "environment_variables": {}
                },
                "on_error": "stop-all"
            },
            {
                "name": "training",
                "type": "execution",
                "template": {
                    "environment": "017d329a-e56f-d451-040b-ccb895513c40",
                    "commit": "~4bfc31b2e318f91a624078c87546f0c283f967b75dd5f7812f4f17ee7b9a8a5a",
                    "step": "train_model",
                    "image": "ultralytics/yolov5",
                    "command": "pip install valohai-utils\npython train.py {parameters}",
                    "inputs": {
                        "train": [
                            "dataset://drift-demo-ships-aerial/dev_train"
                        ],
                        "test": [
                            "dataset://drift-demo-ships-aerial/dev_test"
                        ],
                        "valid": [
                            "dataset://drift-demo-ships-aerial/dev_valid"
                        ]
                    },
                    "parameters": {
                        "yolo_model_name": "yolov8x.pt",
                        "epochs": 3,
                        "batch_size": 8,
                        "image_size": 768,
                        "optimizer": "SGD",
                        "seed": 42,
                        "project": "/valohai/outputs/"
                    },
                    "runtime_config": {},
                    "runtime_config_preset": "",
                    "inherit_environment_variables": True,
                    "environment_variable_groups": [],
                    "tags": [],
                    "time_limit": 0,
                    "environment_variables": {}
                },
                "on_error": "stop-all"
            }
        ],
        "project": "018c8779-9475-09e1-d481-e295ab4de428",
        "tags": [],
        "copy_source": "018cf921-0e19-c1a8-a21f-6ed7c193212e",
        "parameters": {},
        "title": "train-val-pipeline (copy)"
    },
)
if resp.status_code == 400:
    raise RuntimeError(resp.json())
resp.raise_for_status()
data = resp.json()