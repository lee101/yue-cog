from __future__ import annotations

import base64
import os
import urllib.parse
import urllib.request
import uuid


def prepare_output_upload(fmt: str) -> dict:
    import boto3
    from botocore.config import Config

    account = os.getenv("R2_ACCOUNT_ID", "")
    bucket = os.getenv("CLOUDFLARE_BUCKET", "")
    access = os.getenv("CLOUDFLARE_R2_ACCESS_KEY_ID", "")
    secret = os.getenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY", "")
    domain = os.getenv("CLOUDFLARE_CDN_DOMAIN", "netwrckstatic.netwrck.com")
    if not all((account, bucket, access, secret)):
        raise RuntimeError("R2 output storage must be configured before RunPod generation")
    content_type = {"flac": "audio/flac", "mp3": "audio/mpeg", "wav": "audio/wav"}[fmt]
    key = f"static/uploads/music/yue2/renders/{uuid.uuid4().hex}.{fmt}"
    client = boto3.client("s3", endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
        aws_access_key_id=access, aws_secret_access_key=secret, region_name="auto",
        config=Config(signature_version="s3v4"))
    url = client.generate_presigned_url("put_object", Params={"Bucket": bucket, "Key": key,
        "ContentType": content_type}, ExpiresIn=3600)
    return {"put_url": url, "audio_url": "https://" + domain.removeprefix("https://").rstrip("/") + "/" + key,
            "content_type": content_type}


def deliver_audio(result: dict, target: dict | None) -> dict:
    if target is None:
        if len(result["audio_b64"]) > 6 * 1024**2:
            raise RuntimeError("large audio requires a presigned R2 output_upload target")
        return result
    if not isinstance(target, dict):
        raise ValueError("invalid audio upload target")
    put = urllib.parse.urlparse(target.get("put_url", ""))
    public = urllib.parse.urlparse(target.get("audio_url", ""))
    if (put.scheme != "https" or not (put.hostname or "").endswith(".r2.cloudflarestorage.com")
            or public.scheme != "https" or not public.hostname
            or target.get("content_type") != result["content_type"]):
        raise ValueError("invalid audio upload target")
    audio = base64.b64decode(result["audio_b64"], validate=True)
    request = urllib.request.Request(target["put_url"], data=audio, method="PUT",
        headers={"Content-Type": result["content_type"]})
    with urllib.request.urlopen(request, timeout=120) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError("audio upload failed")
    return {**{key: value for key, value in result.items() if key != "audio_b64"},
            "audio_url": target["audio_url"]}
