import os

import requests

api_token = os.environ["VALOHAI_API_TOKEN"]
project_id = os.environ["VH_PROJECT_ID"]

resp = requests.request(
    url="https://app.valohai.com/api/v0/pipelines/create-from-template/",
    method="POST",
    headers={"Authorization": f"Token {api_token}"},
    json={
        "project": project_id,
        "commit": "0aadea924a866f2fd95b5ea304c2a8f7fd0cc189",
        "name": "train-val-pipeline"
    }
)
if resp.status_code == 400:
    raise RuntimeError(resp.json())
resp.raise_for_status()
data = resp.json()

