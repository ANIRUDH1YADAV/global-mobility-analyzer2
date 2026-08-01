from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.logger import logger
from src.pipeline.prediction_pipeline import PredictionPipeline, VisaData

app = FastAPI(
    title="Global Mobility Analyzer",
    description="Machine Learning application for Global Mobility (Visa) Prediction",
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

pipeline = PredictionPipeline()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    continent: str = Form(...),
    education_of_employee: str = Form(...),
    has_job_experience: str = Form(...),
    no_of_employees: int = Form(...),
    yr_of_estab: int = Form(...),
    region_of_employment: str = Form(...),
    prevailing_wage: float = Form(...),
    unit_of_wage: str = Form(...),
):
    try:
        logger.info("Prediction request received")

        visa_data = VisaData(
            continent=continent,
            education_of_employee=education_of_employee,
            has_job_experience=has_job_experience,
            requires_job_training="Y",
            no_of_employees=no_of_employees,
            yr_of_estab=yr_of_estab,
            region_of_employment=region_of_employment,
            prevailing_wage=prevailing_wage,
            unit_of_wage=unit_of_wage,
            full_time_position="Y",
        )

        input_df = visa_data.get_data_as_dataframe()
        prediction, confidence = pipeline.predict(input_df)

        logger.info(f"Prediction: {prediction} | Confidence: {confidence}%")

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": prediction,
                "confidence": confidence,
            },
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": f"Error: {str(e)}",
            },
        )