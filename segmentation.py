# saga에서 segementation 결과 .pt를 .ply로 변환

# input -> .pt(선택된 스플랫인지 아닌지, saga의 output임), 원본.ply
# output -> projection된 gaussian splatting.ply, 추정된 평면의 네개 꼭짓점, 비바닥.ply,

import torch
import numpy as np

INPUT_PLY = "output/oseok3/point_cloud/iteration_30000/scene_point_cloud.ply"
MASK_PATH = "./segmentation_res/oseok3.pt"
OUTPUT_PLY = "./segment/oseok3.ply"


# saga에서 나온 .pt 파일 불러오기
def load_ply_binary(path):
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii")
            header.append(line)
            if line.strip() == "end_header":
                break
        
        # vertex count 찾기
        for h in header:
            if h.startswith("element vertex"):
                vertex_count = int(h.split()[-1])
        
        # 한 vertex의 property 개수 = float property 개수
        # header에서 property float count 하기
        prop_lines = [h for h in header if h.startswith("property float")]
        prop_count = len(prop_lines)

        # binary 데이터 읽기
        # float32 개수 = vertex_count * prop_count
        data = np.fromfile(f, dtype=np.float32, count=vertex_count * prop_count)
        data = data.reshape((vertex_count, prop_count))

    return header, data, vertex_count, prop_count

# ply 파일 저장하기
def save_ply_binary(path, header, data):
    # header 수정: element vertex N 값 바꾸기
    new_header = []
    for h in header:
        if h.startswith("element vertex"):
            new_header.append(f"element vertex {len(data)}\n")
        else:
            new_header.append(h)

    with open(path, "wb") as f:
        # header 저장
        for h in new_header:
            f.write(h.encode("ascii"))
        
        # binary 데이터 저장
        data.astype(np.float32).tofile(f)

# Load original PLY
header, data, vertex_count, prop_count = load_ply_binary(INPUT_PLY) # saga에서 나온 .pt 파일 불러오기
print(f"Loaded {vertex_count} vertices with {prop_count} properties per vertex.")

# Load segmentation mask
mask = torch.load(MASK_PATH).view(-1).cpu().numpy()
assert len(mask) == vertex_count, "mask length != vertex count"

# Ture : 선택된 스플랫만(바닥만), False : 선택 안된 애들(비바닥)
nonfloor_data = data[mask == False] 
floor_data = data[mask == True] 

# 비바닥.ply 저장
print(f"Selected {len(nonfloor_data)} / {vertex_count} vertices")
save_ply_binary(OUTPUT_PLY, header, nonfloor_data)
print(f"Saved filtered PLY → {OUTPUT_PLY}")

# ransac
