import argparse
import torch

from q_align.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from q_align.conversation import conv_templates, SeparatorStyle
from q_align.model.builder import load_pretrained_model
from q_align.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import numpy as np
from PIL import Image
from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

import json
from tqdm import tqdm
from collections import defaultdict

import os

def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr



def wa5(logits):
    import numpy as np
    logprobs = np.array([logits["excellent"], logits["good"], logits["fair"], logits["poor"], logits["bad"]])
    probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
    return np.inner(probs, np.array([1,0.75,0.5,0.25,0.]))

def load_video(video_file):
    from decord import VideoReader
    vr = VideoReader(video_file)

    # Get video frame rate
    fps = vr.get_avg_fps()

    # Calculate frame indices for 1fps
    frame_indices = [int(fps * i) for i in range(int(len(vr) / fps))]
    frames = vr.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(frames[i]) for i in range(int(len(vr) / fps))]

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    
    
    import json

    
    image_paths = [
                  "/workspace/qa/VideoFeedback/test/videos_annotated/",
                  ]
                  

    json_prefix = "/code/vqa/e-bench-db/"
    jsons = [
        '/workspace/qa/VideoFeedback/test/data_annotated.json',
    ]
    # image_paths = ['/userhome/wx-hy/']
    # jsons = [
    #         # '/userhome/wanx-test.json',
    #         '/userhome/hunyuan-test.json'
    # ]

    os.makedirs(f"results-crave/{os.path.basename(args.model_path)}/", exist_ok=True)


    conv_mode = "mplug_owl2"
    
    inp = "How would you rate the quality of this video?"
        
    conv = conv_templates[conv_mode].copy()
    inp =  inp + "\n" + DEFAULT_IMAGE_TOKEN
    conv.append_message(conv.roles[0], inp)
    image = None
        
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt() + " The quality of the video is"
    
    toks = ["good", "poor", "high", "fair", "low", "excellent", "bad", "fine", "moderate",  "decent", "average", "medium", "acceptable"]
    print(toks)
    ids_ = [id_[1] for id_ in tokenizer(toks)["input_ids"]]
    print(ids_)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)
    
    for image_path, json_ in zip(image_paths, jsons):
        with open(json_) as f:
            iqadata = json.load(f) 
            prs, gts = [], []
            for i, llddata in enumerate(tqdm(iqadata)):
                try:
                    filename = llddata["id"] + ".mp4"
                    llddata["logits"] = defaultdict(float)

                    image = load_video(image_path + filename)
                    def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                    image = [expand2square(img, tuple(int(x*255) for x in image_processor.image_mean)) for img in image]
                    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(args.device)

                    if True:
                        with torch.inference_mode():
                            output_logits = model(input_ids,
                                images=[image_tensor])["logits"][:,-1]
                            for tok, id_ in zip(toks, ids_):
                                llddata["logits"][tok] += output_logits.mean(0)[id_].item()
                            llddata["score"] = wa5(llddata["logits"])
                            prs.append(llddata["score"])
                            gt_score = (float(llddata['visual quality']) + float(llddata['temporal consistency']) + float(llddata['dynamic degree']) + float(llddata['text-to-video alignment']) + float(llddata['factual consistency'])) / 5
                            gts.append(gt_score*2)
                            # print(llddata)
                            json_ = json_.replace("combined/", "combined-")
                            with open(f"results-crave/{os.path.basename(args.model_path)}/json_", "a") as wf:
                                json.dump(llddata, wf)

                    if i > 0 and i % 200 == 0:
                        print(spearmanr(prs,gts)[0], pearsonr(prs,gts)[0])
                except Exception as e:
                    print(e)
                    continue
            prs=rescale(prs,gts)
            print("Spearmanr", spearmanr(prs,gts)[0], "Pearson", pearsonr(prs,gts)[0],"Kendallr",kendallr(prs,gts)[0])
            r = np.sqrt((( np.array(prs) -  np.array(gts) ) ** 2).mean())
            print("Spearmanr", spearmanr(prs,gts)[0], "Pearson", pearsonr(prs,gts)[0],"Kendallr",kendallr(prs,gts)[0],"RMSE",r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="q-future/q-align-agi-lora-1")
    parser.add_argument("--model-base", type=str, default=None) # default="q-future/one-align")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)