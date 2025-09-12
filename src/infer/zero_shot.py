import torch
import numpy as np

def zero_shot_scores(pil_img, labels, prefix, model, tokenizer, preprocess):
    with torch.no_grad():
        image = preprocess(pil_img).unsqueeze(0)
        if torch.cuda.is_available(): image = image.cuda()
        prompts = [f"{prefix} {lbl} shoe" for lbl in labels]
        text = tokenizer(prompts)
        if torch.cuda.is_available(): text = text.cuda()

        img_feat = model.encode_image(image)
        txt_feat = model.encode_text(text)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        logits = 100.0 * img_feat @ txt_feat.T
        probs = logits.softmax(dim=-1).cpu().numpy()[0]
        return probs

def top1(pil_img, labels, prefix, model, tokenizer, preprocess):
    probs = zero_shot_scores(pil_img, labels, prefix, model, tokenizer, preprocess)
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx]), probs
