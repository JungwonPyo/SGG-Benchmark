# 🤖 Robot Scene Graph Planner

REACT++ 기반 실시간 장면 그래프 생성 → GCN 상황 판단 → 경로 수정

## 전체 파이프라인
```
이미지
  ↓ tools/auto_label/auto_detect.py   (GroundingDINO + SAM2 → bbox 자동 생성)
  ↓ tools/manual_label/relation_labeler.py  (Streamlit → 관계 수동 입력)
  ↓ tools/convert/jsonl_to_h5.py      (H5 변환 → REACT++ 학습용)
  ↓ tools/convert/jsonl_to_coco.py    (COCO 변환 → YOLO 학습용)
  ↓ reactpp/scripts/train.sh          (REACT++ 관계 예측 학습)
  ↓ reactpp/scripts/run_inference.py  (실시간 장면 그래프 생성)
  ↓ graph_net/train.py                (GCN 상황 분류 학습)
  ↓ graph_net/infer_realtime.py       (실시간 상황 판단 → 경로 수정)
```

## 설치
```bash
git clone <your_repo>
cd robot-sgg-planner
pip install -r requirements.txt
# SGG-Benchmark (REACT++) 설치
git clone https://github.com/Maelic/SGG-Benchmark.git
cd SGG-Benchmark && pip install -e . && cd ..
```

## Step 1: 자동 라벨링 (bbox)
```bash
python tools/auto_label/auto_detect.py --img_dir data/raw_images --situation S2
```

## Step 2: 수동 관계 라벨링 (GUI)
```bash
streamlit run tools/manual_label/relation_labeler.py
```

## Step 3: 데이터 변환
```bash
python tools/convert/jsonl_to_h5.py   --jsonl data/jsonl/manual_labeled.jsonl
python tools/convert/jsonl_to_coco.py --jsonl data/jsonl/manual_labeled.jsonl
```

## Step 4: REACT++ 학습
```bash
bash reactpp/scripts/train.sh predcls  # 1단계
bash reactpp/scripts/train.sh sgcls    # 2단계
bash reactpp/scripts/train.sh sgdet    # 3단계 (최종)
```

## Step 5: GCN 학습
```bash
python graph_net/train.py --jsonl data/jsonl/manual_labeled.jsonl --epochs 50
```

## Step 6: 실시간 실행
```bash
python graph_net/infer_realtime.py
```

## 물체 10종 / 관계 8종 / 상황 5종
| | 물체 | | 관계 | | 상황 |
|---|---|---|---|---|---|
| O1 | 부품 박스 | 1 | on | S1 | 손 진입 → STOP |
| O2 | 플라스틱 트레이 | 2 | inside | S2 | 접근로 점유 → DETOUR |
| O3 | 공정 부품 | 3 | next_to | S3 | 팔 궤적 간섭 → RETARGET |
| O4 | 드라이버 | 4 | above | S4 | 인간 접촉 → WAIT |
| O5 | 작업자 손 | 5 | touching | S5 | 배치 점유 → NORMAL |
| O6 | 조립 지그 | 6 | blocking | | |
| O7 | 폐기 박스 | 7 | near | | |
| O8 | 렌치 | 8 | beside | | |
| O9 | 케이블 묶음 | | | | |
| O10 | 보호 고글 | | | | |
