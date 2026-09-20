#!/usr/bin/env python3
"""Create or update the RunPod serverless endpoint for this image.

Usage:
  RUNPOD_API_KEY=... scripts/runpod_deploy.py --image ghcr.io/lee101/yue-cog:sls
      --name yue2 [--endpoint-id existing]

Fast/cheap defaults: 10s scaledown, max 1 worker, 4090-class GPUs.
"""

import argparse
import json
import os
import urllib.request

GRAPHQL = "https://api.runpod.io/graphql"


def gql(api_key, query, variables):
    body = json.dumps({"query": query, "variables": variables}).encode()
    req = urllib.request.Request(GRAPHQL, data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    if payload.get("errors"):
        raise RuntimeError(payload["errors"])
    return payload["data"]


def save_template(api_key, name, image, env, disk_gb, registry_auth_id):
    inp = {"name": f"{name}-tpl", "imageName": image, "dockerArgs": "python -u /src/rp_handler.py",
           "containerDiskInGb": disk_gb, "volumeInGb": 40, "volumeMountPath": "/runpod-volume",
           "env": [{"key": k, "value": v} for k, v in env.items()]}
    if registry_auth_id:
        inp["registryAuthId"] = registry_auth_id
    data = gql(api_key, "mutation saveTemplate($input: SaveTemplateInput!) { saveTemplate(input: $input) { id } }",
               {"input": inp})
    return data["saveTemplate"]["id"]


def save_endpoint(api_key, name, template_id, gpu_ids, workers_max, idle, endpoint_id=None):
    inp = {"name": name, "templateId": template_id, "gpuIds": gpu_ids,
           "workersMax": workers_max, "workersMin": 0, "idleTimeout": idle,
           "executionTimeoutMs": 1800000, "flashboot": True}
    if endpoint_id:
        inp["id"] = endpoint_id
    data = gql(api_key, "mutation saveEndpoint($input: EndpointInput!) { saveEndpoint(input: $input) { id } }",
               {"input": inp})
    return data["saveEndpoint"]["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", default="yue2")
    ap.add_argument("--endpoint-id", default="")
    ap.add_argument("--gpu-ids", default="ADA_24,AMPERE_24")
    ap.add_argument("--workers-max", type=int, default=1)
    ap.add_argument("--idle", type=int, default=10)
    ap.add_argument("--disk-gb", type=int, default=40)
    ap.add_argument("--registry-auth-id", default="")
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    args = ap.parse_args()
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY is required")
    env = dict(kv.split("=", 1) for kv in args.env)
    env.setdefault("YUE_HF_HOME", "/runpod-volume/hf")
    env.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
    template_id = save_template(api_key, args.name, args.image, env, args.disk_gb,
                                args.registry_auth_id or None)
    endpoint_id = save_endpoint(api_key, args.name, template_id, args.gpu_ids,
                                args.workers_max, args.idle, args.endpoint_id or None)
    print(json.dumps({"template_id": template_id, "endpoint_id": endpoint_id}))


if __name__ == "__main__":
    main()
