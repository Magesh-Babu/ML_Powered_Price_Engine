from pydantic import BaseModel, Field
from datetime import date
from typing import Literal

# Schema V1.0 - Current stable version (Strict Validation)
class QuoteSchemaV1(BaseModel):
    """Input schema for generating a price quote based on product specifications."""
    Alloy: Literal['Iron', 'Aluminium', 'Copper', 'Nickel', 'Titanium', 'Zinc', 'Steel', 'Brass', 'Lead', 'Tin']
    Finish: Literal['Powder coated', 'Wet painted', 'Electroplated', 'Anodized', 'Polished', 'Brushed', 'Chrome plated',
                       'Plasma sprayed', 'Phosphated', 'Hot-dip galvanized', 'Blackening']
    Length_m: float = Field(..., ge=0.1, le=100.0)
    Weight_kg_m: float = Field(..., ge=0.01, le=100.0)
    Tolerances: float = Field(..., ge=0.05, le=0.2)
    GD_T: Literal['low', 'medium', 'high']
    Order_Quantity: int = Field(..., ge=1)
    LME_Price_EUR: float = Field(..., ge=0.1)
    Customer_Category: Literal['micro', 'small', 'medium', 'large']
    Lead_Time_weeks: int = Field(..., ge=1, le=52)
    Profile_Name: str

class QuoteResponse(BaseModel):
    """Response containing a predicted price quote and model version."""
    predicted_price: float
    PI_lower: float
    PI_upper: float
    model_version: int

class FeedbackRequest(QuoteSchemaV1):
    """Feedback submission including actual price of a quoted item."""
    Quote_Price_SEK: float = Field(..., gt=0, description="The actual confirmed price for the item")
    Quote_Date: date

class FeedbackResponse(BaseModel):
    """Response after feedback is processed, confirming model update."""
    message: str
    new_version: int