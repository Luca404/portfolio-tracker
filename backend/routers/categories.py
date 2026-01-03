from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from typing import Optional
from datetime import datetime

from models import CategoryModel, SubcategoryModel, UserModel, TransactionModel, TransactionType
from schemas import (
    CategoryCreate,
    CategoryUpdate,
    CategoryResponse,
    CategoryWithStats,
    SubcategoryCreate,
    SubcategoryUpdate,
    SubcategoryResponse,
)
from utils import get_db, verify_token
from utils.default_categories import create_default_categories

router = APIRouter(prefix="/api/categories", tags=["categories"])


# CATEGORIES


@router.get("", response_model=list[CategoryWithStats])
def get_categories(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token),
):
    """Get all categories for the current user with optional stats"""
    # Crea categorie di default se l'utente non ne ha
    create_default_categories(db, user)

    categories = db.execute(
        select(CategoryModel).where(CategoryModel.user_id == user.id)
    ).scalars().all()

    # If date filters provided, calculate stats
    result = []
    for category in categories:
        category_data = CategoryWithStats.model_validate(category)

        if start_date or end_date:
            # Build query for transactions
            query = select(func.sum(TransactionModel.amount), func.count(TransactionModel.id)).where(
                TransactionModel.user_id == user.id,
                TransactionModel.category == category.name,
                TransactionModel.type == TransactionType.EXPENSE
            )

            if start_date:
                query = query.where(TransactionModel.date >= datetime.fromisoformat(start_date))
            if end_date:
                query = query.where(TransactionModel.date <= datetime.fromisoformat(end_date))

            total, count = db.execute(query).first()
            category_data.total_amount = total or 0
            category_data.transaction_count = count or 0

        result.append(category_data)

    return result


@router.post("", response_model=CategoryResponse)
def create_category(
    category: CategoryCreate,
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token),
):
    """Create a new category"""
    # Check if category with same name already exists
    existing = db.execute(
        select(CategoryModel).where(
            CategoryModel.user_id == user.id,
            CategoryModel.name == category.name
        )
    ).scalars().first()

    if existing:
        raise HTTPException(status_code=400, detail="Category with this name already exists")

    new_category = CategoryModel(
        user_id=user.id,
        name=category.name,
        icon=category.icon,
    )
    db.add(new_category)
    db.commit()
    db.refresh(new_category)
    return new_category


@router.put("/{category_id}", response_model=CategoryResponse)
def update_category(
    category_id: int,
    category_update: CategoryUpdate,
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token),
):
    """Update a category"""
    category = db.execute(
        select(CategoryModel).where(
            CategoryModel.id == category_id,
            CategoryModel.user_id == user.id
        )
    ).scalars().first()

    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    if category_update.name is not None:
        # Check if new name conflicts with existing category
        existing = db.execute(
            select(CategoryModel).where(
                CategoryModel.user_id == user.id,
                CategoryModel.name == category_update.name,
                CategoryModel.id != category_id
            )
        ).scalars().first()

        if existing:
            raise HTTPException(status_code=400, detail="Category with this name already exists")

        category.name = category_update.name

    if category_update.icon is not None:
        category.icon = category_update.icon

    category.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(category)
    return category


@router.delete("/{category_id}")
def delete_category(
    category_id: int,
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token),
):
    """Delete a category"""
    category = db.execute(
        select(CategoryModel).where(
            CategoryModel.id == category_id,
            CategoryModel.user_id == user.id
        )
    ).scalars().first()

    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    db.delete(category)
    db.commit()
    return {"message": "Category deleted successfully"}


# SUBCATEGORIES


@router.get("/{category_id}/subcategories", response_model=list[SubcategoryResponse])
def get_subcategories(
    category_id: int,
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token),
):
    """Get all subcategories for a category"""
    # Verify category belongs to user
    category = db.execute(
        select(CategoryModel).where(
            CategoryModel.id == category_id,
            CategoryModel.user_id == user.id
        )
    ).scalars().first()

    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    return category.subcategories


@router.post("/{category_id}/subcategories", response_model=SubcategoryResponse)
def create_subcategory(
    category_id: int,
    subcategory: SubcategoryCreate,
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token),
):
    """Create a new subcategory"""
    # Verify category belongs to user
    category = db.execute(
        select(CategoryModel).where(
            CategoryModel.id == category_id,
            CategoryModel.user_id == user.id
        )
    ).scalars().first()

    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    # Check if subcategory with same name already exists
    existing = db.execute(
        select(SubcategoryModel).where(
            SubcategoryModel.category_id == category_id,
            SubcategoryModel.name == subcategory.name
        )
    ).scalars().first()

    if existing:
        raise HTTPException(status_code=400, detail="Subcategory with this name already exists")

    new_subcategory = SubcategoryModel(
        category_id=category_id,
        name=subcategory.name,
        icon=subcategory.icon,
    )
    db.add(new_subcategory)
    db.commit()
    db.refresh(new_subcategory)
    return new_subcategory


@router.put("/{category_id}/subcategories/{subcategory_id}", response_model=SubcategoryResponse)
def update_subcategory(
    category_id: int,
    subcategory_id: int,
    subcategory_update: SubcategoryUpdate,
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token),
):
    """Update a subcategory"""
    # Verify category belongs to user
    category = db.execute(
        select(CategoryModel).where(
            CategoryModel.id == category_id,
            CategoryModel.user_id == user.id
        )
    ).scalars().first()

    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    subcategory = db.execute(
        select(SubcategoryModel).where(
            SubcategoryModel.id == subcategory_id,
            SubcategoryModel.category_id == category_id
        )
    ).scalars().first()

    if not subcategory:
        raise HTTPException(status_code=404, detail="Subcategory not found")

    if subcategory_update.name is not None:
        # Check if new name conflicts with existing subcategory
        existing = db.execute(
            select(SubcategoryModel).where(
                SubcategoryModel.category_id == category_id,
                SubcategoryModel.name == subcategory_update.name,
                SubcategoryModel.id != subcategory_id
            )
        ).scalars().first()

        if existing:
            raise HTTPException(status_code=400, detail="Subcategory with this name already exists")

        subcategory.name = subcategory_update.name

    if subcategory_update.icon is not None:
        subcategory.icon = subcategory_update.icon

    subcategory.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(subcategory)
    return subcategory


@router.delete("/{category_id}/subcategories/{subcategory_id}")
def delete_subcategory(
    category_id: int,
    subcategory_id: int,
    db: Session = Depends(get_db),
    user: UserModel = Depends(verify_token),
):
    """Delete a subcategory"""
    # Verify category belongs to user
    category = db.execute(
        select(CategoryModel).where(
            CategoryModel.id == category_id,
            CategoryModel.user_id == user.id
        )
    ).scalars().first()

    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    subcategory = db.execute(
        select(SubcategoryModel).where(
            SubcategoryModel.id == subcategory_id,
            SubcategoryModel.category_id == category_id
        )
    ).scalars().first()

    if not subcategory:
        raise HTTPException(status_code=404, detail="Subcategory not found")

    db.delete(subcategory)
    db.commit()
    return {"message": "Subcategory deleted successfully"}
