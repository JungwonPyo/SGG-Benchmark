# tools/manual_label/relation_labeler.py
"""
streamlit run tools/manual_label/relation_labeler.py
"""
import streamlit as st
import cv2, json, os, numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Robot SGG Labeler", layout="wide")

CLASSES = [
    "부품 박스","플라스틱 트레이","공정 부품","드라이버","작업자 손",
    "조립 지그","폐기 박스","렌치","케이블 묶음","보호 고글"
]
RELATIONS = ["on","inside","next_to","above","touching","blocking","near","beside"]
SITUATIONS = {
    "S1: 손 진입 (above → stop)":        ("S1","stop"),
    "S2: 접근로 점유 (blocking → detour)": ("S2","detour"),
    "S3: 팔 궤적 간섭 (near → retarget)": ("S3","retarget"),
    "S4: 인간 접촉 (touching → wait)":    ("S4","wait"),
    "S5: 배치 점유 (on → normal)":        ("S5","normal"),
}
COLORS = {
    "부품 박스":(255,0,0), "플라스틱 트레이":(0,255,0), "공정 부품":(0,0,255),
    "드라이버":(255,165,0), "작업자 손":(128,0,128), "조립 지그":(0,128,128),
    "폐기 박스":(255,20,147), "렌치":(139,69,19), "케이블 묶음":(128,128,0),
    "보호 고글":(0,191,255)
}

def draw_scene(img, objects, relations):
    disp = img.copy()
    for obj in objects:
        b  = obj["bbox"]; c = COLORS.get(obj["class"],(0,255,0))
        cv2.rectangle(disp,(b[0],b[1]),(b[2],b[3]),c,2)
        cv2.putText(disp,f"{obj['id']}:{obj['class']}",(b[0],b[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,c,1)
    id2center = {o["id"]:(int((o["bbox"][0]+o["bbox"][2])/2),
                           int((o["bbox"][1]+o["bbox"][3])/2)) for o in objects}
    for r in relations:
        if r["subject"] in id2center and r["object"] in id2center:
            p1,p2 = id2center[r["subject"]], id2center[r["object"]]
            cv2.arrowedLine(disp,p1,p2,(255,255,0),1,tipLength=0.2)
            mx,my = (p1[0]+p2[0])//2,(p1[1]+p2[1])//2
            cv2.putText(disp,r["predicate"],(mx,my),
                        cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,0),1)
    return disp

# ─── Session state init ───────────────────────────────────────────
for k,v in [("objects",[]),("relations",[]),("drawing",False),
            ("pt1",None),("selected_class",CLASSES[0])]:
    if k not in st.session_state: st.session_state[k]=v

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("🤖 SGG Labeler")
    jsonl_src = st.text_input("Auto JSONL (입력)","data/jsonl/auto_labeled.jsonl")
    out_path  = st.text_input("Manual JSONL (출력)","data/jsonl/manual_labeled.jsonl")
    img_file  = st.file_uploader("이미지 업로드",type=["jpg","jpeg","png"])
    sit_key   = st.selectbox("상황 선택",list(SITUATIONS.keys()))
    sit_code, path_mod = SITUATIONS[sit_key]

    st.markdown("---")
    st.subheader("📦 객체 추가")
    sel_cls = st.selectbox("클래스",CLASSES)
    col1,col2 = st.columns(2)
    x1_=col1.number_input("x1",0,2000,100)
    y1_=col2.number_input("y1",0,2000,100)
    x2_=col1.number_input("x2",0,2000,200)
    y2_=col2.number_input("y2",0,2000,200)
    if st.button("➕ 객체 추가"):
        oid = f"O{len(st.session_state.objects)+1}"
        st.session_state.objects.append(
            {"id":oid,"class":sel_cls,"bbox":[x1_,y1_,x2_,y2_],"confidence":1.0})
    if st.button("🗑️ 객체 전체 삭제"):
        st.session_state.objects = []
        st.session_state.relations = []

    st.markdown("---")
    if jsonl_src and os.path.exists(jsonl_src):
        if st.button("📂 Auto 결과 불러오기"):
            with open(jsonl_src) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            if lines:
                st.session_state.objects   = lines[0]["objects"]
                st.session_state.relations = lines[0].get("relationships",[])
                st.success(f"{len(lines[0]['objects'])}개 객체 로드")

# ─── Main panel ───────────────────────────────────────────────────
if img_file:
    img = np.array(Image.open(img_file).convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h,w    = img.shape[:2]
    col_img, col_ctrl = st.columns([3,2])

    with col_img:
        disp = draw_scene(img_bgr, st.session_state.objects, st.session_state.relations)
        st.image(cv2.cvtColor(disp,cv2.COLOR_BGR2RGB), use_column_width=True, caption="현재 장면")

    with col_ctrl:
        st.subheader("🔗 관계 추가")
        obj_ids = [o["id"] for o in st.session_state.objects]
        if len(obj_ids) >= 2:
            subj = st.selectbox("Subject",
                [f"{o['id']}:{o['class']}" for o in st.session_state.objects])
            rel  = st.selectbox("Predicate", RELATIONS)
            obj_ = st.selectbox("Object",
                [f"{o['id']}:{o['class']}" for o in st.session_state.objects])
            if st.button("➕ 관계 추가"):
                st.session_state.relations.append({
                    "subject":subj.split(":")[0],
                    "predicate":rel,
                    "object":obj_.split(":")[0]
                })

        st.subheader("📋 현재 관계")
        for i,r in enumerate(st.session_state.relations):
            c1,c2 = st.columns([4,1])
            c1.write(f"`{r['subject']}` **{r['predicate']}** `{r['object']}`")
            if c2.button("🗑️",key=f"delrel{i}"):
                st.session_state.relations.pop(i); st.rerun()

        st.markdown("---")
        st.subheader("💾 저장")
        if st.button("✅ 이 장면 저장"):
            scene = {
                "scene_id":  f"{sit_code}_{img_file.name.split('.')[0]}_001",
                "situation": sit_code,
                "image_path":img_file.name,
                "objects":   st.session_state.objects,
                "relationships": st.session_state.relations,
                "path_modification": path_mod,
                "goal_position": [w//2, h//2],
                "goal_changed": False
            }
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path,"a",encoding="utf-8") as f:
                f.write(json.dumps(scene,ensure_ascii=False)+"\n")
            st.success(f"✅ 저장 완료 → {out_path}")
            st.session_state.objects = []
            st.session_state.relations = []
else:
    st.info("👈 왼쪽 사이드바에서 이미지를 업로드 하세요.")
