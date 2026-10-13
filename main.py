from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import sys

try:
    from sklearn._loss import _loss as sklearn_loss
    sys.modules.setdefault("_loss", sklearn_loss)
except ImportError:
    pass

app = FastAPI()
model_data = joblib.load("heart.joblib")
model = model_data["model"]
feature_names = model_data["features"]

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"result": None})


@app.post("/predict")
async def predict(
    request: Request,
    age: int = Form(...),
    sex: int = Form(...),
    cp: int = Form(...),
    trestbps: int = Form(...),
    chol: int = Form(...),
    fbs: int = Form(...),
    restecg: int = Form(...),
    thalach: int = Form(...),
    exang: int = Form(...),
    oldpeak: float = Form(...),
    slope: int = Form(...),
    ca: int = Form(...),
    thal: int = Form(...),
):
    payload = pd.DataFrame(
        [{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }], columns=feature_names
    )

    prediction = int(model.predict(payload)[0])
    result = "Patient is likely to have heart disease." if prediction == 1 else "Patient is unlikely to have heart disease."
    return templates.TemplateResponse(request, "index.html", {"result": result})