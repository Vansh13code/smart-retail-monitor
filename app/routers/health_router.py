from fastapi import APIRouter

router = APIRouter(
    prefix="/health",
    tags=["Health"]
)


@router.get("/")
def health():
    return {
        "status": "ok",
        "message": "Smart Retail Shelf Monitoring API is running"
    }