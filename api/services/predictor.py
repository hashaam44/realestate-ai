#Request response model
# Import Pydantic tools for data validation
from pydantic import BaseModel, Field


# Define the input data required for prediction
class PropertyInput(BaseModel):

    # Number of bedrooms
    bed: int = Field(gt=0, le=100)

    # Number of bathrooms
    bath: float = Field(gt=0, le=100)

    # Lot size in acres
    acre_lot: float = Field(ge=0)

    # House size in square feet
    house_size: float = Field(gt=0)

    # Previous sale year
    prev_sold_year: int = Field(ge=1800, le=2100)

    # Property status
    status: str

    # State name
    state: str


# Define the response returned after prediction
class PredictionResponse(BaseModel):

    # Predicted property price
    predicted_price: float

    # Currency used for the prediction
    currency: str