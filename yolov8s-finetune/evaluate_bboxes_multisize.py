"""Minimal evaluator: loop over GT and predicted boxes and save cropped patches.

For each image in data/images/val:
 - read GT boxes from data/labels/val (YOLO normalized format)
 - run the YOLO model to get predicted boxes
 - crop and save each GT and predicted box to out/gt and out/pred

This keeps the code intentionally minimal.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import compute_map


# Load VAE
class SelfAttention(nn.Module):
    def __init__(self, n_heads, embd_dim, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.in_proj = nn.Linear(embd_dim, 3 * embd_dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(embd_dim, embd_dim, bias=out_proj_bias)
        self.d_heads = embd_dim // n_heads
    def forward(self, x, casual_mask=False):
        batch_size, seq_len, d_embed = x.shape
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_heads)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.view(interim_shape)
        k = k.view(interim_shape)
        v = v.view(interim_shape)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        weight = q @ k.transpose(-1, -2)
        if casual_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        weight /= math.sqrt(self.d_heads)
        weight = F.softmax(weight, dim=-1)
        output = weight @ v
        output = output.transpose(1, 2)
        output = output.reshape((batch_size, seq_len, d_embed))
        output = self.out_proj(output)
        return output

# %%
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    def forward(self, x):
        residual = x.clone()
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))
        x += residual
        return x

# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    def forward(self, x):
        residue = x.clone()
        x = self.groupnorm1(x)
        x = F.selu(x)
        x = self.conv1(x)
        x = self.groupnorm2(x)
        x = self.conv2(x)
        return x + self.residual_layer(residue)

# %%
class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            ResidualBlock(128, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            ResidualBlock(256, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    def forward(self, x):
        for module in self:
            if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        return x

# %%
class Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )
    def forward(self, x):
        x /= 0.18215
        for module in self:
            x = module(x)
        return x

# %%
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.mu = nn.Linear(512, 128)
        self.logvar = nn.Linear(512, 128)
        self.restarter = nn.Linear(128, 256)
    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        mean = self.mu(encoded)
        log_variance = self.logvar(encoded)
        log_variance = torch.clamp(log_variance, -30, 20)
        std = torch.exp(0.5 * log_variance)
        eps = torch.randn_like(std)
        mean = mean + eps * std
        restarted = self.restarter(mean)
        restarted = restarted.view(restarted.size(0), 4, 8, 8)
        decoded = self.decoder(restarted)
        return decoded, mean, log_variance


# %%

import pickle
import gzip
from pathlib import Path
import importlib
import json

# Try multiple ways to load the saved regressor. Common formats include pickle, joblib
# and sometimes compressed (gzip) or torch-saved objects. This tries a cascade of
# loaders and raises a clear error if none succeed.
# Utility: try to load a saved estimator from several common formats and return
# the inner object that implements .predict() (or None if not found).
def _find_predictor(obj):
    if obj is None:
        return None
    if hasattr(obj, 'predict') and callable(getattr(obj, 'predict')):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            p = _find_predictor(v)
            if p is not None:
                return p
    if isinstance(obj, (list, tuple)):
        for v in obj:
            p = _find_predictor(v)
            if p is not None:
                return p
    return None


def _load_estimator_from_path(p: Path):
    if not p.exists():
        return None
    # try pickle
    try:
        with p.open('rb') as f:
            obj = pickle.load(f)
            pred = _find_predictor(obj)
            if pred is not None:
                return pred
    except Exception:
        pass
    # try joblib
    try:
        joblib = importlib.import_module('joblib')
        try:
            obj = joblib.load(p)
            pred = _find_predictor(obj)
            if pred is not None:
                return pred
        except Exception:
            pass
    except Exception:
        pass
    # try gzip+pickle
    try:
        with gzip.open(p, 'rb') as gf:
            obj = pickle.load(gf)
            pred = _find_predictor(obj)
            if pred is not None:
                return pred
    except Exception:
        pass
    # try torch.load
    try:
        import torch
        obj = torch.load(str(p), map_location='cpu')
        pred = _find_predictor(obj)
        if pred is not None:
            return pred
    except Exception:
        pass
    return None


# Load predictors for each size group (fallbacks included). We expect models
# to be saved as `{name}_model_{size}.pkl` by the multisize trainer. If those
# files are absent, try a generic `{name}_model.pkl` as a fallback.
predictors_by_size = {}
base_models_dir = Path('reg_models/classifiers')
classifier_basename = 'hist_gb'  # default classifier base name used previously
sizes = ['small', 'medium', 'large']
generic_path = base_models_dir / f"{classifier_basename}_model.pkl"
for s in sizes:
    p = base_models_dir / f"{classifier_basename}_model_{s}.pkl"
    pred = _load_estimator_from_path(p)
    if pred is None and generic_path.exists():
        pred = _load_estimator_from_path(generic_path)
    if pred is None:
        print(f"Warning: no predictor found for size '{s}' (tried {p} and {generic_path})")
    else:
        # compatibility guard for sklearn HistGradientBoostingClassifier
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            if isinstance(pred, HistGradientBoostingClassifier) and not hasattr(pred, '_preprocessor'):
                setattr(pred, '_preprocessor', None)
        except Exception:
            pass
    predictors_by_size[s] = pred


def read_yolo_labels(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        x_c = float(parts[1])
        y_c = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        x1 = int(round((x_c - w / 2.0) * img_w))
        y1 = int(round((y_c - h / 2.0) * img_h))
        x2 = int(round((x_c + w / 2.0) * img_w))
        y2 = int(round((y_c + h / 2.0) * img_h))
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    return x1, y1, x2, y2

import tqdm

def evaluate_bboxes(conf_threshold=0.5, conf=0.001):
    # Hardcoded paths for model, data, and output
    model_path = "/Users/aditya/Desktop/yolov8s-finetune/weights/best.pt"
    data_root = Path("/Users/aditya/Desktop/InProgress/Radio Galaxies/dataset")
    out_dir = Path("/Users/aditya/Desktop/yolov8s-finetune/results/crops")
    imgsz = 640
    device = torch.device("mps")
    # Load VAE checkpoint. Newer PyTorch defaults to weights_only=True which will
    # refuse to unpickle custom classes; try safe weights-only load first and fall
    # back to full load if necessary (only if checkpoint is trusted).
    vae_path = Path("VAEModelCubicFit.pth")
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_path}")
    try:
        # try the safer weights-only load first
        vae = torch.load(str(vae_path), map_location=device)
    except (pickle.UnpicklingError, RuntimeError) as e:
        print("Weights-only load failed, attempting full load with weights_only=False (trusted file)")
        try:
            vae = torch.load(str(vae_path), map_location=device, weights_only=False)
            # print out the number of parameters
            total_params = sum(p.numel() for p in vae.parameters())
            print(f"VAE has {total_params} parameters")
        except Exception:
            # if that still fails, re-raise the original exception for debugging
            raise

    model = YOLO(str(model_path))

    images_val = data_root / 'images' / 'val'
    labels_val = data_root / 'labels' / 'val'
    out_gt = out_dir / 'gt'
    out_pred = out_dir / 'pred'
    out_gt.mkdir(parents=True, exist_ok=True)
    out_pred.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in images_val.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    # Prepare collections for JSON output (COCO-style-ish)
    images_out = []
    annotations_out = []
    categories_seen = set()
    image_id = 0
    ann_id = 1

    for img_path in tqdm.tqdm(imgs):
        img = cv2.imread(str(img_path))
        if img is None:
            print('skip', img_path)
            continue
        h, w = img.shape[:2]

        image_id += 1
        images_out.append({
            'id': image_id,
            'file_name': img_path.name,
            'width': w,
            'height': h
        })

        # GT boxes
        gt_file = labels_val / (img_path.stem + '.txt')
        gt_boxes = read_yolo_labels(gt_file, w, h)
        for i, (cls, x1, y1, x2, y2) in enumerate(gt_boxes):
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            outp = out_gt / f"{img_path.stem}_gt_{i}.png"
            cv2.imwrite(str(outp), crop)

        # Predictions
        res = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, device=device, verbose=False)
        if len(res) == 0:
            continue
        r = res[0]
        if hasattr(r, 'boxes') and r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else [0.0] * len(xyxy)
            clss = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, 'cls') else [-1] * len(xyxy)
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = [int(round(v)) for v in xyxy[i]]
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img[y1:y2, x1:x2]
                confv = float(confs[i])

                #Send it throguh the VAE
                #Fit to 64x64
                # Preserve aspect ratio and center on a 64x64 canvas (overwrite `crop` so subsequent resize is a no-op)

                h0, w0 = crop.shape[:2]
                target = 64
                scale = min(target / w0, target / h0)
                new_w = max(1, int(round(w0 * scale)))
                new_h = max(1, int(round(h0 * scale)))

                interp = cv2.INTER_CUBIC
                resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

                canvas = np.full((target, target, 3), 0, dtype=resized.dtype)  # black background
                top = (target - new_h) // 2
                left = (target - new_w) // 2
                canvas[top:top + new_h, left:left + new_w] = resized

                crop_resized = canvas
                crop_resized = crop_resized.astype(np.float32) / 255.0

                crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    recon, mu, sigma = vae(crop_tensor)
                
                #Send mu through the random forest regressor to get a class prediction
                mu_np = mu.cpu().numpy()
                # sklearn expects 2D array for predict: (n_samples, n_features)
                if mu_np.ndim == 1:
                    mu_in = mu_np.reshape(1, -1)
                else:
                    mu_in = mu_np

                # choose predictor based on box area
                w_box = x2 - x1
                h_box = y2 - y1
                area_box = int(w_box * h_box)
                # same thresholds as training script (small <=24, medium 24<area<=48, large>48)
                if area_box <= 24**2:
                    size_name = 'small'
                else:
                    size_name = 'medium'

                predictor = predictors_by_size.get(size_name)
                # fallback to any available predictor
                if predictor is None:
                    for p in predictors_by_size.values():
                        if p is not None:
                            predictor = p
                            break
                if predictor is None:
                    raise RuntimeError('No classifier predictor available for any size groups')

                # Use predict_proba if available to derive a confidence; otherwise use predict()
                if hasattr(predictor, 'predict_proba') and callable(getattr(predictor, 'predict_proba')):
                    probs = predictor.predict_proba(mu_in)
                    confidence = float(np.max(probs))
                    class_pred = int(np.argmax(probs, axis=1)[0])
                else:
                    # fall back to hard predict (confidence unknown)
                    class_pred = int(np.round(predictor.predict(mu_in)[0]))
                    confidence = 1.0
                categories_seen.add(int(class_pred))
                
                # Save annotation for JSON output (COCO-style bbox [x,y,w,h])
                if w_box <= 0 or h_box <= 0:
                    continue
                if confidence < conf_threshold:
                    continue
                annotations_out.append({
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': int(class_pred),
                    'bbox': [int(x1), int(y1), int(w_box), int(h_box)],
                    'score': float(confv)
                })
                ann_id += 1
        
    # Write out JSON file similar to a COCO/train.json structure
    annotations_dir = data_root / 'annotations'
    annotations_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = annotations_dir / 'train.json'

    categories_out = []
    for cid in sorted(categories_seen):
        categories_out.append({'id': int(cid), 'name': str(cid)})

    out_dict = {
        'images': images_out,
        'annotations': annotations_out,
        'categories': categories_out
    }

    with out_json_path.open('w', encoding='utf-8') as jf:
        json.dump(out_dict, jf, indent=2)

    print(f'Wrote {len(annotations_out)} predicted annotations for {len(images_out)} images to: {out_json_path}')

    # Also write the predictions JSON to the path expected by compute_map.main
    preds_dir = data_root / 'predictions'
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_json_path = preds_dir / 'predictions.json'
    # compute_map expects either a list of detections or a dict with 'annotations'
    preds_obj = {'annotations': annotations_out}
    with preds_json_path.open('w', encoding='utf-8') as pf:
        json.dump(preds_obj, pf, indent=2)

    print(f'Wrote predictions JSON to: {preds_json_path}')

    # Call compute_map to evaluate mAP using the predictions file we just created
    map = compute_map.main(conf)
    return map

if __name__ == "__main__":
    # evaluate_bboxes(conf_threshold=0.125, conf=0.001)
    # 2-D optimizer: tune both conf_threshold (linear 0-1) and conf (linear 0-1)
    import numpy as _np

    # search ranges (both treated as linear in [0,1])
    ct_a, ct_b = 0.05, 0.95           # conf_threshold range (linear)
    conf_a, conf_b = 0.001, 0.999     # conf range (linear 0..1)

    tol_ct = 1e-3
    tol_conf = 1e-3
    max_evals = 30   # total allowed expensive evaluate_bboxes calls
    tried = {}       # (ct_rounded, conf_rounded) -> map (None if failed)

    def _key(ct, conf):
        return (round(float(ct), 6), round(float(conf), 8))

    def eval_pair(ct, conf):
        k = _key(ct, conf)
        if k in tried:
            return tried[k]
        try:
            m = evaluate_bboxes(conf_threshold=float(ct), conf=float(conf))
            try:
                m_f = float(m)
            except Exception:
                m_f = None
        except Exception:
            import traceback
            print(f"Error while evaluating ct={ct}, conf={conf}:")
            traceback.print_exc()
            m_f = None
        tried[k] = m_f
        return m_f

    # iterative zooming grid search (coarse->fine)
    eval_count = 0
    iteration = 0
    max_iters = 10
    nx, ny = 4, 4  # grid resolution per iteration

    while eval_count < max_evals and iteration < max_iters:
        iteration += 1
        # generate grid (both ct and conf linear)
        ct_points = _np.linspace(ct_a, ct_b, nx)
        conf_points = _np.linspace(conf_a, conf_b, ny)

        # evaluate grid points until budget exhausted
        for ct in ct_points:
            for conf in conf_points:
                if len(tried) >= max_evals:
                    break
                if _key(ct, conf) in tried:
                    continue
                eval_pair(ct, conf)
        eval_count = len(tried)

        # find best valid point (treat None as -inf)
        best_k = None
        best_val = -float("inf")
        for (k_ct, k_conf), v in tried.items():
            if v is None:
                continue
            if float(v) > best_val:
                best_val = float(v)
                best_k = (float(k_ct), float(k_conf))

        if best_k is None:
            # no successful evals yet; stop to avoid endless loops
            print("No successful evaluations yet; stopping search.")
            break

        best_ct, best_conf = best_k

        # shrink search window around best point
        def shrink_interval(a, b, center, shrink_factor=0.5):
            half = (b - a) * 0.5 * shrink_factor
            na = max(a, center - half)
            nb = min(b, center + half)
            # ensure we keep a non-zero window
            if nb - na < 1e-6:
                na = max(a, center - 1e-6)
                nb = min(b, center + 1e-6)
            return na, nb

        ct_a, ct_b = shrink_interval(ct_a, ct_b, best_ct, shrink_factor=0.5)
        conf_a, conf_b = shrink_interval(conf_a, conf_b, best_conf, shrink_factor=0.5)

        # stopping criteria
        if (ct_b - ct_a) < tol_ct and (conf_b - conf_a) < tol_conf:
            break

    # pick best tried configuration
    best_conf_thresh = None
    best_conf_det = None
    best_map = -1.0
    trials = []
    for (k_ct, k_conf), m in sorted(tried.items()):
        trials.append({"conf_threshold": float(k_ct), "conf": float(k_conf), "map": (None if m is None else float(m))})
        if m is not None and float(m) > best_map:
            best_map = float(m)
            best_conf_thresh = float(k_ct)
            best_conf_det = float(k_conf)

    print(f"Optimization complete. Best conf_threshold={best_conf_thresh}, conf={best_conf_det} -> map={best_map}")
