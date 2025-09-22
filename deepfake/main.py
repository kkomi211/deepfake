import io
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, JSONResponse
from PIL import Image

from wm_core import (
    embed_pipeline, extract_pipeline,
    array_to_png_bytes_with_meta, extract_meta_from_png,
)

app = FastAPI(title="DWT-SVD Invisible Watermark API (Fixed Params)", version="1.0.0")

# 고정 파라미터
WAVELET_DEFAULT = "db2"
LEVEL_DEFAULT   = 2
BAND_DEFAULT    = "HL"
ALPHA_DEFAULT   = 0.12

# 서버에 두는 기본 워터마크 파일 경로 (클라이언트는 host만 업로드)
WATERMARK_DEFAULT_PATH = Path("hanshin.png")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed_fixed_single")
async def embed_fixed_single(
    host: UploadFile = File(..., description="원본 이미지만 업로드 (워터마크는 서버의 hanshin.png 사용)")
):
    # 기본 워터마크 존재 확인
    if not WATERMARK_DEFAULT_PATH.exists():
        return JSONResponse(
            status_code=500,
            content={"detail": f"기본 워터마크 파일을 찾을 수 없습니다: {WATERMARK_DEFAULT_PATH.resolve()}"}
        )

    # 이미지 로드
    host_img = Image.open(host.file)
    wm_img   = Image.open(str(WATERMARK_DEFAULT_PATH))

    # 삽입
    watermarked_arr, meta, psnr_db = embed_pipeline(
        host_img=host_img,
        wm_img=wm_img,
        wavelet=WAVELET_DEFAULT,
        level=LEVEL_DEFAULT,
        band=BAND_DEFAULT,
        alpha=ALPHA_DEFAULT
    )

    # 메타 저장(추출용) + 고정 파라미터 기록
    meta["psnr_db"] = psnr_db
    meta["fixed_params"] = {
        "wavelet": WAVELET_DEFAULT,
        "level": LEVEL_DEFAULT,
        "band": BAND_DEFAULT,
        "alpha": ALPHA_DEFAULT,
    }
    meta["wm_src"] = str(WATERMARK_DEFAULT_PATH)

    # PNG 바이너리(메타 포함)로 반환
    png_bytes = array_to_png_bytes_with_meta(watermarked_arr, meta)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="face_marked.png"'}
    )

@app.post("/extract_fixed")
async def extract_fixed(
    watermarked_png: UploadFile = File(..., description="워터마크 삽입 PNG(내부에 wm_meta 포함)")
):
    # 업로드 파일 읽기
    data = await watermarked_png.read()
    bio  = io.BytesIO(data)

    # PNG 메타에서 삽입 시 메타 복원
    meta, im = extract_meta_from_png(bio)

    # 추출
    wm_est_arr = extract_pipeline(watermarked_img=im, meta=meta)

    # PNG로 반환
    out = io.BytesIO()
    Image.fromarray(wm_est_arr.astype("uint8"), mode="L").save(out, format="PNG")
    return Response(
        content=out.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="wm_extracted.png"'}
    )
