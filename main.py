from fastapi import FastAPI, HTTPException
from schemas import QuoteSchemaV1, QuoteResponse, FeedbackRequest, FeedbackResponse
from inference import predict_price
from continuous_learning import add_feedback
from utils import get_user_logger

app = FastAPI(title="Intelligent Pricing Engine")

@app.post("/users/{user_id}/quote", response_model=QuoteResponse)
def get_quote(user_id: str, request: QuoteSchemaV1):
    """
    Generate a price quote for a given user's product specification.
    """
    logger = get_user_logger(user_id)
    try:
        price, lower, upper, version = predict_price(user_id, request.model_dump())
    except ValueError as e:
        # Likely no data/model for user
        logger.error(f"Quote generation failed: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Internal error during quote generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during quote generation")
    # Return the quote result
    return QuoteResponse(predicted_price=price, PI_lower=lower, PI_upper=upper, model_version=version)

@app.post("/users/{user_id}/feedback", response_model=FeedbackResponse)
def post_feedback(user_id: str, feedback: FeedbackRequest):
    """
    Submit a validated quote to improve the model for the user.
    """
    logger = get_user_logger(user_id)
    try:
        new_version = add_feedback(user_id, feedback.model_dump())
    except Exception as e:
        logger.error(f"Internal error during feedback processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during feedback processing")
    # Return confirmation of model update
    return FeedbackResponse(message="Feedback added and model retrained", new_version=new_version)
