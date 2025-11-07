import os, sys, glob, time
import numpy as np
import cv2
import argparse
from plyfile import PlyData, PlyElement

import time

start = time.perf_counter()

# ========= 설정 =========
# -d colmap path
# -g gaussian path
# ex) python gaussian_mapping.py -d data/oseok3/ -g output/db0a2cc5-9
parser = argparse.ArgumentParser(description="바닥인 가우시안 찾기")
parser.add_argument("-d", "--data", required=True, help="Colmap path")
parser.add_argument("-g", "--gaussian", required=True, help="Gaussian Splatting path")
args = parser.parse_args()

SPARSE_DIR     = os.path.join(args.data, "sparse/0") # cameras.bin, images.bin
IMAGE_ROOT     = os.path.join(args.data, "images") # 원본 이미지 폴더
colmap_folder = os.path.basename(os.path.normpath(args.data))
MASK_ROOT      = f"output/{colmap_folder}/masks"                   # 마스크 루트(IMG_0001/* or IMG_0001.png)

GAUSS_PLY_IN   = os.path.join(args.gaussian, "point_cloud/iteration_30000/scene_point_cloud.ply")

# 표결 파라미터
FLOOR_THR        = 0.5    # floor_prob >= THR → 바닥으로 간주(마스크 목록 저장용)
# ==========================

# ---- read_write_model.py 찾기 (COLMAP) ----
script_dir = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(os.environ.get("CONDA_PREFIX",""), "share/colmap/scripts/python"),
    "/usr/local/share/colmap/scripts/python",
    "/usr/share/colmap/scripts/python",
    os.path.join(script_dir, "../colmap/scripts/python/"),
    script_dir,
    os.path.abspath(os.path.join(script_dir, "..")),
    os.path.abspath(os.path.join(script_dir, "../..")),
]
rw = None
for d in CANDIDATES:
    p = os.path.join(d, "read_write_model.py")
    if os.path.exists(p):
        sys.path.insert(0, d)
        try:
            import read_write_model as rw
            break
        except Exception:
            pass
if rw is None:
    print("[ERR] read_write_model.py not found. Put it in ./tools or $CONDA_PREFIX/share/colmap/scripts/python")
    sys.exit(1)

# ---- 마스크 로드 (파일/폴더 + 합집합) ----
# 이진화를 또 한다?
def _load_binary_mask(p):
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if g is None: return None
    return (g >= 128).astype(np.uint8)

# 이미지 resize
def _resize_nearest(m, HW):
    H,W = HW
    if m.shape[:2] == (H,W): return m
    return cv2.resize(m, (W,H), interpolation=cv2.INTER_NEAREST)

def get_union_mask(img_name, mask_root, target_hw, cache):
    H,W = target_hw
    stem, _ = os.path.splitext(img_name)
    # 1) 단일 파일
    for cand in [img_name, stem+"_mask.png", stem+"_mask.jpg", stem+"_mask.jpeg"]:
        p = os.path.join(mask_root, cand)
        if os.path.exists(p):
            m = cache.get(p)
            if m is None:
                m = _load_binary_mask(p); cache[p]=m
            return _resize_nearest(m, (H,W)) if m is not None else None
    # # 2) 폴더 합집합 -> 필요없을듯
    # folder = os.path.join(mask_root, stem)
    # if os.path.isdir(folder):
    #     union=None
    #     for fp in sorted(glob.glob(os.path.join(folder, "*"))):
    #         ext = os.path.splitext(fp)[1].lower()
    #         if ext not in [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]:
    #             continue
    #         m = cache.get(fp)
    #         if m is None:
    #             m = _load_binary_mask(fp); cache[fp]=m
    #         if m is None: continue
    #         m = _resize_nearest(m, (H,W))
    #         union = m if union is None else (union | m)
    #     if union is not None: return union
    return None

# ---- 카메라 모델 투영 ----
def qvec2rotmat(q):
    w,x,y,z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z,   2*x*z+2*w*y],
        [2*x*y+2*w*z,   1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y,   2*y*z+2*w*x,   1-2*x*x-2*y*y]], dtype=float)

def project_point(Xw, R, t, cam):
    Xc = R @ Xw + t
    if Xc[2] <= 0: return None  # 뒤쪽
    x = Xc[0]/Xc[2]; y = Xc[1]/Xc[2]
    m = cam["model"]; W=cam["width"]; H=cam["height"]; p=cam["params"]
    if m in ("SIMPLE_PINHOLE","SIMPLE_RADIAL"):
        f,cx,cy = p[0], p[1], p[2]
        if len(p)>=4 and m=="SIMPLE_RADIAL":
            k1=p[3]; r2=x*x+y*y; s=(1+k1*r2); x*=s; y*=s
        u = f*x + cx; v = f*y + cy
    elif m=="PINHOLE":
        fx,fy,cx,cy = p[:4]; u = fx*x + cx; v = fy*y + cy
    elif m=="OPENCV":
        fx,fy,cx,cy,k1,k2,p1,p2,k3 = p[:9]
        r2 = x*x + y*y
        x_d = x*(1+k1*r2+k2*r2*r2+k3*r2*r2*r2) + 2*p1*x*y + p2*(r2+2*x*x)
        y_d = y*(1+k1*r2+k2*r2*r2+k3*r2*r2*r2) + p1*(r2+2*y*y) + 2*p2*x*y
        u = fx*x_d + cx; v = fy*y_d + cy
    else:
        return None
    if 0 <= u < W and 0 <= v < H:
        return int(round(u)), int(round(v))
    return None

# ---- 메인 함수 ----
def main():
    # 0) COLMAP 모델/이미지 로드
    cams   = rw.read_cameras_binary(os.path.join(SPARSE_DIR, "cameras.bin"))
    images = rw.read_images_binary( os.path.join(SPARSE_DIR, "images.bin"))

    # per-image 뷰 준비(+마스크)
    mask_cache = {}
    views = []
    for idx, I in enumerate(sorted(images.values(), key=lambda x: x.id)):
        cam = cams[I.camera_id]
        camd = dict(model=cam.model, width=cam.width, height=cam.height, params=list(cam.params))
        R = qvec2rotmat(I.qvec)
        t = I.tvec.astype(float)
        ip = os.path.join(IMAGE_ROOT, I.name)
        im = cv2.imread(ip)
        if im is None:
            print(f"[WARN] image read fail: {ip}")
            continue
        H,W = im.shape[:2]
        m = get_union_mask(I.name, MASK_ROOT, (H,W), cache=mask_cache)
        if m is None:
            continue
        views.append((camd, R, t, m))
    print(f"[INFO] usable views: {len(views)}")
    if not views:
        raise RuntimeError("no views with masks")

    # 1) 가우시안 PLY 로드 → centers
    ply = PlyData.read(GAUSS_PLY_IN)
    v = ply['vertex']
    G = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float64)
    N = len(G)
    print(f"[G] gaussians: {N}")

    # 2) 각 가우시안에 대해 마스크 표결
    floor_prob = np.zeros(N, float)

    for i in range(N):
        X = G[i]
        votes = []
        for (camd, R, t, m) in views:
            pr = project_point(X, R, t, camd)
            if pr is None: continue
            u,vp = pr
            if 0 <= vp < m.shape[0] and 0 <= u < m.shape[1]:
                votes.append(1 if m[vp,u]>0 else 0)
        floor_prob[i] = np.mean(votes) if votes else 0.0

    # 3) 저장
    out_dir = os.path.dirname(GAUSS_PLY_IN) or "."
    os.makedirs(out_dir, exist_ok=True)

    idxs = np.arange(N, dtype=int)
    prob_csv = os.path.join(out_dir, "gaussian_floor_prob.csv")
    np.savetxt(prob_csv, np.c_[idxs, floor_prob],
               delimiter=",", header="index,floor_prob", comments="", fmt=["%d","%.6f"])
    print(f"[SAVE] floor prob -> {prob_csv}")

    mask_idx_path = os.path.join(out_dir, f"gaussian_floor_indices_thr{FLOOR_THR:.2f}.txt")
    sel = np.where(floor_prob >= FLOOR_THR)[0]
    np.savetxt(mask_idx_path, sel, fmt="%d")
    print(f"[SAVE] floor indices (thr={FLOOR_THR}) -> {mask_idx_path}")

if __name__ == "__main__":
    main()

end = time.perf_counter()

print(f"실행 시간: {end - start:.6f}초")