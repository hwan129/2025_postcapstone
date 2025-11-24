# floor segmentation
# input : colmap에서 사용된 multi-view images
# colmap의 images 파일

# output : Binary images

from ultralytics import YOLO
import cv2
import numpy as np
import os
from glob import glob
import argparse

import time

start = time.perf_counter()

# 경로 설정
# 콜맵의 결과물을 path로 입력하면 됨
# ex) python floor_seg_1.py -d data/oseok4/
parser = argparse.ArgumentParser(description="바닥 Segmentation")
parser.add_argument("-d", "--data", required=True, help="Colmap path")
args = parser.parse_args()

colmap_folder = os.path.basename(os.path.normpath(args.data))   # colmap 폴더 이름을 가져옴
image_dir = os.path.join(args.data, "images")   # colmap에서 선택된 이미지 path
output_dir = os.path.join("output", colmap_folder)  # colmap 폴더의 이름과 같은 output 폴더
model_path = "../yolo11/best.pt" # yolo11 사용


# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)
mask_dir = os.path.join(output_dir, "masks")
os.makedirs(mask_dir, exist_ok=True)

# 모델 로드
model = YOLO(model_path)

# 이미지 확장자 목록
img_exts = [".JPG", ".jpeg", ".png"]

# 이미지 파일 리스트
image_files = [f for ext in img_exts for f in glob(os.path.join(image_dir, f"*{ext}"))]

print(f"총 {len(image_files)}개의 이미지에서 YOLO 세그멘테이션 수행")

for img_path in image_files:
    print(f"\n처리 중: {img_path}")
    filename = os.path.splitext(os.path.basename(img_path))[0]

    # 이미지 읽기
    img = cv2.imread(img_path)
    if img is None:
        print(f"[경고] 이미지 읽기 실패: {img_path}")
        continue

    H, W = img.shape[:2]
    out = img.copy()
    all_masks = []

    # YOLO 세그멘테이션 수행
    results = model(img, conf=0.5)

    for r in results:
        if r.masks is None:
            continue

        union_mask = np.zeros((H, W), dtype=np.uint8)

        # 마스크가 위로 올라가는 현상 수정
        for seg in r.masks.xy:
            pts = np.array(seg, dtype=np.int32)
            cv2.fillPoly(union_mask, [pts], 1)

        # 시각화 -> 확인용
        # overlay = out.copy()
        # overlay[union_mask == 1] = (255, 0, 0)
        # out = np.where(union_mask[..., None] == 1,
        #             (0.25 * overlay + 0.75 * out).astype(np.uint8),
        #             out)

        # 저장
        cv2.imwrite(os.path.join(mask_dir, f"{filename}_mask.png"), union_mask * 255)


    # 시각화 저장 -> 확인용
    # cv2.imwrite(os.path.join(output_dir, f"{filename}_overlay.png"), out)
    # cv2.imwrite(os.path.join(output_dir, f"{filename}_overlay_q100.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 100])

print("\n모든 이미지 처리 완료.")

end = time.perf_counter()

print(f"실행 시간: {end - start:.6f}초")