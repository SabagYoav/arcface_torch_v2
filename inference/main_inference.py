from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop, estimate_norm

from backbones import get_model
from skimask_classifier import is_ski_mask


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_ckpt(path: Path, map_location="cpu") -> Dict:
    ckpt = torch.load(str(path), map_location=map_location)
    if isinstance(ckpt, dict):
        # common wrappers
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
    # if it's already a state_dict-like mapping or something torch.load returns
    return ckpt


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # handles "module." from DDP
    if not isinstance(state_dict, dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        out[nk] = v
    return out

def _to_bgr_uint8(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Convert PIL RGB or numpy RGB/BGR to uint8 BGR for InsightFace."""
    if isinstance(img, Image.Image):
        rgb = np.array(img.convert("RGB"))
        bgr = rgb[:, :, ::-1].copy()
        return bgr.astype(np.uint8)

    arr = img
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    # Heuristic: assume RGB if last dim==3 (common), convert to BGR
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr[:, :, ::-1].copy()
    return arr


def _clip_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(1, min(int(x2), w))
    y2 = max(1, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _crop_np(img_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return img_bgr[y1:y2, x1:x2]

@dataclass
class GalleryIndex:
    labels: List[str]                   # len = num_ids
    mean_embs: torch.Tensor             # [num_ids, D] normalized
    # optional per-id instances (not needed for search; useful for debugging)
    instances: Optional[Dict[str, List[torch.Tensor]]] = None


class PartialFullFaceRecognizer:
    """
    Partial ↔ Full Face Recognizer

    - Detect + align using InsightFace landmarks
    - Encode using full or partial backbone
    - Build gallery: mean embedding per id (normalized mean)
    - Search: cosine similarity (dot product after L2 norm)
    """

    def __init__(
        self,
        fullface_model_path: Path,
        partialface_model_path: Path,
        full_network: str = "r50",
        partial_network: str = "vit_s_dp005_mask_0",
        full_num_features: int = 512,
        partial_num_features: int = 512,
        device: Optional[str] = None,
        det_size: Tuple[int, int] = (640, 640),
        det_gpu: bool = True,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # face detector + landmarks
        self.face_app = FaceAnalysis(name="buffalo_l")
        ctx_id = 0 if (det_gpu and torch.cuda.is_available()) else -1
        self.face_app.prepare(ctx_id=ctx_id, det_size=det_size)

        # image -> tensor
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        # models
        self.full_encoder = self._load_model(fullface_model_path, full_network, full_num_features)
        self.partial_encoder = self._load_model(partialface_model_path, partial_network, partial_num_features)

        # indices (built later)
        self.full_gallery: Optional[GalleryIndex] = None
        self.partial_gallery: Optional[GalleryIndex] = None

    def _load_model(self, model_path: Path, network: str, num_features: int) -> torch.nn.Module:
        model = get_model(network, dropout=0.0, fp16=False, num_features=num_features).to(self.device)
        sd = _load_ckpt(model_path, map_location="cpu")
        sd = _strip_module_prefix(sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[WARN] load_state_dict({model_path.name}) missing={len(missing)} unexpected={len(unexpected)}")
        model.eval()
        return model

    # ----------------------------
    # Detection + alignment
    # ----------------------------
    def _detect_largest_face(self, pil_img: Image.Image):
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        rgb = np.array(pil_img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        faces = self.face_app.get(bgr)
        if not faces:
            return None, bgr

        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        return face, bgr

    # def _align_112(self, bgr: np.ndarray, face) -> Image.Image:
    #     # norm_crop returns aligned BGR 112x112
    #     aligned_bgr = norm_crop(bgr, face.kps, image_size=112)
    #     aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    #     return Image.fromarray(aligned_rgb)

   


    def _align_112(self, bgr: np.ndarray, face):
        ARC_FACE_TEMPLATE_112 = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)
        aligned_bgr = norm_crop(bgr, face.kps, image_size=112)  # works for you
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

        # After alignment, kps are at canonical template (approx). Use template directly.
        aligned_kps = ARC_FACE_TEMPLATE_112.copy()

        return Image.fromarray(aligned_rgb), aligned_kps


    def _make_partial_from_aligned(self, aligned_rgb: Image.Image, kps: np.ndarray, margin: int = 20) -> Image.Image:
        """
        Build an eyes-only partial image from an aligned 112x112 RGB face.

        Logic:
        1) Use kps (InsightFace 5pt: [L_eye, R_eye, nose, L_mouth, R_mouth])
        2) Create eyes ROI around the two eye points with margin (pixels)
        3) Paste that ROI into a zero image of shape 112x112
        4) Return PIL RGB

        Args:
        aligned_rgb: PIL image (expected 112x112)
        kps: np.ndarray shape (5,2) in aligned image coordinates
        margin: pixels to expand around eyes bounding box
        """
        arr = np.array(aligned_rgb, dtype=np.uint8)  # (112,112,3) RGB
        H, W = arr.shape[:2]
        if (H, W) != (112, 112):
            # If not 112x112, we still work in that space, but you likely want it fixed earlier.
            pass

        kps = np.asarray(kps, dtype=np.float32)
        if kps.ndim != 2 or kps.shape[0] < 2 or kps.shape[1] != 2:
            raise ValueError(f"kps must be (N,2) with N>=2, got {kps.shape}")

        # Eyes are first two points in 5pt convention
        le = kps[0]
        re = kps[1]

        x1 = int(np.floor(min(le[0], re[0]) - margin))
        y1 = int(np.floor(min(le[1], re[1]) - margin))
        x2 = int(np.ceil (max(le[0], re[0]) + margin))
        y2 = int(np.ceil (max(le[1], re[1]) + margin))

        # Clip ROI to image bounds
        cx1 = max(0, x1); cy1 = max(0, y1)
        cx2 = min(W, x2); cy2 = min(H, y2)

        # If ROI is invalid (eyes out of bounds / bad kps), return all zeros
        out = np.zeros((112, 112, 3), dtype=np.uint8)

        if cx2 <= cx1 or cy2 <= cy1:
            return Image.fromarray(out)

        roi = arr[cy1:cy2, cx1:cx2, :]  # RGB

        # Where to paste ROI inside the zero image (preserve original coords)
        px1 = cx1
        py1 = cy1
        px2 = px1 + (cx2 - cx1)
        py2 = py1 + (cy2 - cy1)

        out[py1:py2, px1:px2, :] = roi
        return Image.fromarray(out)

    def preprocess(self, pil_img: Image.Image, mode: str = "full") -> torch.Tensor:
        """
        mode: "full" | "partial" | "auto"
        """
        face, bgr = self._detect_largest_face(pil_img)
        if face is None or getattr(face, "kps", None) is None:
            # fallback: no alignment
            x = self.transform(pil_img.convert("RGB")).unsqueeze(0).to(self.device, non_blocking=True)
            return x

        aligned, aligned_kps = self._align_112(bgr, face)

        if mode == "partial":
            aligned = self._make_partial_from_aligned(aligned, kps=aligned_kps, margin=20)
        # elif mode == "auto":
        #     # simple heuristic: if bbox touches borders a lot -> partial
        #     x1, y1, x2, y2 = map(int, face.bbox)
        #     H, W = bgr.shape[:2]
        #     border = min(x1, y1, W - x2, H - y2)
        #     mode = "partial" if border < 5 else "full"
        #     if mode == "partial":
        #         aligned = self._make_partial_from_aligned(aligned)

        x = self.transform(aligned).unsqueeze(0).to(self.device, non_blocking=True)
        return x

    # ----------------------------
    # Encoding
    # ----------------------------
    @torch.no_grad()
    def encode(self, pil_img: Image.Image, mode: str = "full") -> torch.Tensor:
        """
        Returns: [D] CPU, L2-normalized
        mode: "full" | "partial" | "auto"
        """
        x = self.preprocess(pil_img, mode=mode)
        if mode == "partial":
            emb = self.partial_encoder(x)
        elif mode == "full":
            emb = self.full_encoder(x)
        # elif mode == "auto":
        #     # preprocess may flip auto->partial based on heuristic,
        #     # but we need to be consistent: re-run the heuristic decision
        #     # by checking if partial crop was applied. easiest: do a cheap check again:
        #     face, bgr = self._detect_largest_face(pil_img)
        #     use_partial = False
        #     if face is not None:
        #         x1, y1, x2, y2 = map(int, face.bbox)
        #         H, W = bgr.shape[:2]
        #         border = min(x1, y1, W - x2, H - y2)
        #         use_partial = border < 5
        #     emb = self.partial_encoder(x) if use_partial else self.full_encoder(x)


        emb = F.normalize(emb, dim=1)          # [1,D]
        return emb.squeeze(0).detach().cpu()   # [D]

    # ----------------------------
    # Gallery building
    # ----------------------------
    def build_gallery(self, gallery_root: Path, store_instances: bool = False) -> None:
        """
        Expects:
            gallery_root/
                id1/ img1.jpg img2.jpg ...
                id2/ ...
        Builds FULL-FACE gallery only (mean embedding per id).
        """
        labels: List[str] = []
        full_means: List[torch.Tensor] = []
        instances_full: Dict[str, List[torch.Tensor]] = {}

        id_dirs = [p for p in gallery_root.iterdir() if p.is_dir()]
        id_dirs.sort(key=lambda p: p.name)

        for id_dir in id_dirs:
            label = id_dir.name.lower()
            img_paths = [p for p in id_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
            if not img_paths:
                continue

            embs_full: List[torch.Tensor] = []

            for p in img_paths:
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    continue

                ef = self.encode(img, mode="full")   # [D] CPU
                embs_full.append(ef)

            if not embs_full:
                continue

            Ef = torch.stack([F.normalize(e, dim=0) for e in embs_full], dim=0)  # [N,D]
            mf = F.normalize(Ef.mean(dim=0), dim=0)                               # [D]

            labels.append(label)
            full_means.append(mf)

            if store_instances:
                instances_full[label] = embs_full

        if not labels:
            raise RuntimeError(f"No identities found under: {gallery_root}")

        full_mat = torch.stack(full_means, dim=0)   # [M,D]
        self.full_gallery = GalleryIndex(
            labels=labels,
            mean_embs=F.normalize(full_mat, dim=1),
            instances=instances_full if store_instances else None,
        )

    # ----------------------------
    # Search
    # ----------------------------
    def search(self, query_img: Image.Image, threshold: float = 0.2, mode: str = "full", topk: int = 5):
        """
        mode: "full" | "partial" | "auto"
        Gallery is FULL ONLY.
        Returns: list of (label, score) sorted desc
        """
        if self.full_gallery is None:
            raise RuntimeError("Full gallery not built. Call build_gallery(gallery_root) first.")

        q = self.encode(query_img, mode=mode)                 # [D] CPU
        q = F.normalize(q, dim=0).unsqueeze(0)                # [1,D]
        scores = (q @ self.full_gallery.mean_embs.T).squeeze(0)
        return self._postprocess_scores(self.full_gallery.labels, scores, threshold, topk)

    @staticmethod
    def _postprocess_scores(labels: List[str], scores: torch.Tensor, threshold: float, topk: int):
        scores = scores.detach().cpu()
        idx = torch.argsort(scores, descending=True)
        out = []
        for i in idx[: max(topk, 1)]:
            s = float(scores[i])
            if s >= threshold:
                out.append((labels[int(i)], s))
        return out
    
    def detect_full_or_partial(self, pil_img: Image.Image) -> str:
            """
            Simple heuristic to detect if the face is full or partial (ski mask).
            Returns "full" or "partial".
            """
            if is_ski_mask(pil_img):
                return "partial"
            return "full"

if __name__ == "__main__":
    gallery_root = Path("inference/faces_gallary")
    fullface_model_path = Path("work_dirs/config_glint360k_subset_fullface_best_18_01_26/best_model.pt")
    partial_face_model_path = Path("work_dirs/clip_ratio_20/best_model.pt")

    system = PartialFullFaceRecognizer(
        fullface_model_path=fullface_model_path,
        partialface_model_path=partial_face_model_path,
        full_network="r50",
        partial_network="vit_s_dp005_mask_0",
    )

    test_image = Image.open("inference/faces_query/Yoav_with_ski_mask.jpeg").convert("RGB")
    mode = system.detect_full_or_partial(test_image)
    print("Detected mode:", mode)
    system.build_gallery(gallery_root)
    matches = system.search(test_image, threshold=0.1, mode=mode, topk=5)
    print("Matches:", matches)

    #TODO: 
    # for partial-face add alignment and cropping the eyes roi