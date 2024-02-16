from fastapi import FastAPI ,From,request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from pipeline.prediction_pipeline import predictpipeline,CustomData

app=FastAPI()
templates=Jinja2Templates(directory="templates")

class PredictRequest(BaseModel):
    carat: float
    depth: float
    table: float
    x: float
    y: float
    z: float
    cut: str
    color: str
    clarity: str


@app.get("/")
def homepage(request:Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.post("/predict")
def predict_datapoint(request: Request, data: PredictRequest = Form(...)):
    final_data = CustomData(
        carat=data.carat,
        depth=data.depth,
        table=data.table,
        x=data.x,
        y=data.y,
        z=data.z,
        cut=data.cut,
        color=data.color,
        clarity=data.clarity,
    ).get_data_as_dataframe()

    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(final_data)

    result = round(pred[0], 2)

    return templates.TemplateResponse(
        "result.html", {"request": request, "final_result": result}
    )

if __name__=="__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port=8000)

