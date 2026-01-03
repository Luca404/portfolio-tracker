from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class SubcategoryBase(BaseModel):
    name: str
    icon: str = "📌"


class SubcategoryCreate(SubcategoryBase):
    pass


class SubcategoryUpdate(BaseModel):
    name: Optional[str] = None
    icon: Optional[str] = None


class SubcategoryResponse(SubcategoryBase):
    id: int
    category_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CategoryBase(BaseModel):
    name: str
    icon: str = "💰"
    category_type: Optional[str] = None  # 'expense', 'income', 'investment', 'transfer', or null


class CategoryCreate(CategoryBase):
    pass


class CategoryUpdate(BaseModel):
    name: Optional[str] = None
    icon: Optional[str] = None
    category_type: Optional[str] = None


class CategoryResponse(CategoryBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    subcategories: list[SubcategoryResponse] = []

    class Config:
        from_attributes = True


class CategoryWithStats(CategoryResponse):
    total_amount: float = 0
    transaction_count: int = 0
