import sys
import logging

# ✅ VERY IMPORTANT (forces logs to show)
logging.basicConfig(level=logging.INFO, force=True)

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from us_visa.pipeline.prediction_pipeline import PredictionPipeline, VisaData
from us_visa.logger import logging as custom_logging


app = FastAPI(title="US Visa Approval Predictor")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Load model + preprocessor once
pipeline = PredictionPipeline()


# -------------------- HOME --------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request},
    )


# -------------------- PREDICT --------------------
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    continent: str             = Form(...),
    education_of_employee: str = Form(...),
    has_job_experience: str    = Form(...),
    no_of_employees: int       = Form(...),
    yr_of_estab: int           = Form(...),
    region_of_employment: str  = Form(...),
    prevailing_wage: float     = Form(...),
    unit_of_wage: str          = Form(...),
):
    try:
        # Step 1: Create input object
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

        # Step 2: Convert to DataFrame
        df = visa_data.get_data_as_dataframe()
        logging.info(f"\n📥 Input Data:\n{df}")

        # Step 3: Predict
        result = pipeline.predict(df)

        logging.info(f"\n🎯 Final Prediction: {result}\n")

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "result": result,
            },
        )

    except Exception as e:
        logging.error(f"❌ Prediction error: {str(e)}")

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "result": f"Error: {str(e)}"
            },
        )