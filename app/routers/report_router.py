from fastapi import APIRouter

router = APIRouter(
    prefix="/generate-report",
    tags=["Reports"]
)


@router.get("/")
def report():
    return {
        "status": "ready",
        "message": "Reporting module integrated successfully."
    }