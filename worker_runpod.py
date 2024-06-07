import os, json, requests, runpod

import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

discord_token = os.getenv('com_camenduru_discord_token')
web_uri = os.getenv('com_camenduru_web_uri')
web_token = os.getenv('com_camenduru_web_token')

with torch.inference_mode():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_config = get_pretrained_model("audo/stable-audio-open-1.0")
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    model = model.to(device)

@torch.inference_mode()
def generate(input):
    values = input["input"]

    prompt = values['prompt']
    seconds_start = values['seconds_start']
    seconds_total = values['seconds_total']
    steps = values['steps']
    cfg_scale = values['cfg_scale']
    sigma_min = values['sigma_min']
    sigma_max = values['sigma_max']
    sampler_type = values['sampler_type']
    
    conditioning = [{
        "prompt": prompt,
        "seconds_start": seconds_start, 
        "seconds_total": seconds_total
    }]
    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sampler_type=sampler_type,
        device=device
    )
    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    torchaudio.save("/content/output.wav", output, sample_rate)

    result = "/content/output.wav"
    response = None
    try:
        source_id = values['source_id']
        del values['source_id']
        source_channel = values['source_channel']     
        del values['source_channel']
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        files = {default_filename: open(result, "rb").read()}
        payload = {"content": f"{json.dumps(values)} <@{source_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{source_channel}/messages",
            data=payload,
            headers={"authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if os.path.exists(result):
            os.remove(result)

    if response and response.status_code == 200:
        try:
            payload = {"jobId": job_id, "result": response.json()['attachments'][0]['url']}
            requests.post(f"{web_uri}/api/notify", data=json.dumps(payload), headers={'Content-Type': 'application/json', "authorization": f"{web_token}"})
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            return {"result": response.json()['attachments'][0]['url']}
    else:
        return {"result": "ERROR"}

runpod.serverless.start({"handler": generate})