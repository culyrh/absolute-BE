"""
S3 관련 API 엔드포인트 (나중에 추가할 예정)
"""

from fastapi import APIRouter, Body, HTTPException
from typing import Dict, Any

router = APIRouter(
    prefix="/api/s3",
    tags=["storage"],
    responses={404: {"description": "Not found"}},
)


@router.post("/presigned", response_model=Dict[str, str])
async def get_presigned_url(
    filename: str = Body(..., embed=True, description="파일명"),
    content_type: str = Body("image/jpeg", embed=True, description="컨텐츠 타입")
):
    """
    S3 업로드 URL 발급 API
    
    이미지 업로드를 위한 S3 Presigned URL을 발급합니다.
    
    - **filename**: 파일명 (필수)
    - **content_type**: 컨텐츠 타입 (기본값: "image/jpeg")
    
    **Response**:
    - uploadUrl: 업로드를 위한 Presigned URL
    - fileUrl: 업로드 완료 후 접근 가능한 파일 URL
    """
    try:
        # 나중에 AWS S3 SDK를 사용하여 실제 구현 예정
        # 임시로 더미 데이터 반환
        return {
            "uploadUrl": f"https://example.com/upload/{filename}?content_type={content_type}",
            "fileUrl": f"https://example.com/files/{filename}"
        }
    except Exception as e:
        print(f"⚠️ S3 Presigned URL 발급 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"S3 Presigned URL 발급 중 오류가 발생했습니다: {str(e)}")