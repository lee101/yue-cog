#!/usr/bin/env python3
"""Create or update the RunPod serverless endpoint for this image.

Usage:
  RUNPOD_API_KEY=... scripts/runpod_deploy.py --image ghcr.io/lee101/yue-cog:sls
      --name yue2 [--endpoint-id existing]

Defaults: 10s scaledown, zero workers until explicitly enabled, RTX 4090.
"""

import argparse
import json
import os
import urllib.request

REST = "https://rest.runpod.io/v1"


def api(api_key: str, method: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(REST + path, data=json.dumps(payload).encode(), method=method,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key,
                 "User-Agent": "Omniserve-YuE/1"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def save_template(api_key, name, image, env, disk_gb, registry_auth_id):
    payload = {"name": name + "-tpl", "imageName": image, "isServerless": True,
        "containerDiskInGb": disk_gb, "volumeInGb": 0,
        "dockerEntrypoint": ["python", "-u", "/src/rp_handler.py"],
        "dockerStartCmd": [], "env": env}
    if registry_auth_id:
        payload["containerRegistryAuthId"] = registry_auth_id
    return api(api_key, "POST", "/templates", payload)["id"]


def save_endpoint(api_key, name, template_id, gpu_ids, workers_max, idle, endpoint_id=None):
    if workers_max not in (0, 1):
        raise ValueError("YuE deployment supports a maximum of one worker")
    gpu_names = {"ADA_24": "NVIDIA GeForce RTX 4090", "AMPERE_24": "NVIDIA GeForce RTX 3090"}
    payload = {"name": name, "templateId": template_id, "computeType": "GPU", "gpuCount": 1,
        "allowedCudaVersions": ["12.8", "12.9", "13.0"],
        "gpuTypeIds": [gpu_names.get(gpu.strip(), gpu.strip()) for gpu in gpu_ids.split(",")],
        "workersMax": workers_max or 1, "workersMin": 0, "idleTimeout": idle,
        "executionTimeoutMs": 480000, "flashboot": True, "scalerType": "QUEUE_DELAY", "scalerValue": 4}
    path = "/endpoints" + ("/" + endpoint_id if endpoint_id else "")
    result = api(api_key, "PATCH" if endpoint_id else "POST", path, payload)
    identity = result["id"]
    result = api(api_key, "PATCH", "/endpoints/" + identity,
                 {"workersMin": 0, "workersMax": workers_max})
    if result.get("workersMin") != 0 or result.get("workersMax") != workers_max:
        raise RuntimeError("worker limit verification failed for endpoint " + identity)
    return identity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", default="yue2")
    ap.add_argument("--endpoint-id", default="")
    ap.add_argument("--gpu-ids", default="ADA_24")
    ap.add_argument("--workers-max", type=int, default=0)
    ap.add_argument("--idle", type=int, default=10)
    ap.add_argument("--disk-gb", type=int, default=40)
    ap.add_argument("--registry-auth-id", default="")
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    args = ap.parse_args()
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY is required")
    env = dict(kv.split("=", 1) for kv in args.env)
    env.setdefault("YUE_HF_HOME", "/tmp/yue-hf")
    env.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
    template_id = save_template(api_key, args.name, args.image, env, args.disk_gb,
                                args.registry_auth_id or None)
    endpoint_id = save_endpoint(api_key, args.name, template_id, args.gpu_ids,
                                args.workers_max, args.idle, args.endpoint_id or None)
    print(json.dumps({"template_id": template_id, "endpoint_id": endpoint_id}))


if __name__ == "__main__":
    main()
