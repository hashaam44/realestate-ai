from pydantic import BaseModel, Field


class PropertyInput(BaseModel):
    bed: int = Field(gt=0, le=100)
    bath: float = Field(gt=0, le=100)
    acre_lot: float = Field(ge=0)
    house_size: float = Field(gt=0)
    prev_sold_year: int = Field(ge=1800, le=2100)
    status: str
    state: str


class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str