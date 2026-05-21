from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI()

# Pydantic-модель для входных данных
class DefectDataRequest(BaseModel):
    defect_data: List[List[float]]

# Pydantic-модель для ответа
class DefectDataResponse(BaseModel):
    average_value: int

@app.post("/analyze-defects", response_model=DefectDataResponse)
def analyze_defects(request: DefectDataRequest):
    try:
        data = np.array(request.defect_data)
        if data.size == 0:
            raise HTTPException(status_code=400, detail="Defect data is empty")
        avg = int(np.round(np.mean(data)))

        # --- Логика определения статуса ---
        status = "APPROVED" if avg > 50 else "REJECTED" # Пример логики
        # --- /Логика ---

        # Генерируем URL для изображения (пример)
        image_url = f"http://localhost:8000/images/scan_{avg}.png"

        # Отправляем данные в ui-service
        inspection_data = {
            "pipeId": 123, # Замените на реальный ID трубы
            "imageUrl": image_url,
            "status": status
        }
        try:
            requests.post(f"{UI_SERVICE_URL}", json=inspection_data)
        except Exception as e:
            print(f"Ошибка отправки данных в UI Service: {e}")

        return DefectDataResponse(average_value=avg)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}