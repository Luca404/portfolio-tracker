"""General helper utilities."""

from fastapi import HTTPException


def validate_order_input(order):
    """
    Validate order input data.

    Args:
        order: Order object to validate

    Raises:
        HTTPException: If validation fails
    """
    if order.quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be positive")
    if order.price <= 0:
        raise HTTPException(status_code=400, detail="Price must be positive")
    if order.order_type not in {"buy", "sell"}:
        raise HTTPException(status_code=400, detail="order_type must be 'buy' or 'sell'")
    if order.commission < 0:
        raise HTTPException(status_code=400, detail="Commission cannot be negative")
    if order.instrument_type.lower() not in {"stock", "etf"}:
        raise HTTPException(status_code=400, detail="instrument_type must be one of: stock, etf")
