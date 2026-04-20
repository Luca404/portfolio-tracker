from fastapi import APIRouter, Depends, HTTPException
from typing import Optional

from schemas import (
    CategoryCreate,
    CategoryUpdate,
    SubcategoryCreate,
    SubcategoryUpdate,
)
from utils import get_supabase, verify_token
from utils.default_categories import create_default_categories_if_needed

router = APIRouter(prefix="/api/categories", tags=["categories"])


# CATEGORIES

@router.get("")
def get_categories(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user_id: str = Depends(verify_token),
):
    sb = get_supabase()
    create_default_categories_if_needed(sb, user_id)

    categories = sb.table("categories").select("*, subcategories(*)").eq("user_id", user_id).execute().data

    result = []
    for cat in categories:
        total_amount = 0
        transaction_count = 0
        if start_date or end_date:
            query = sb.table("transactions").select("amount").eq("user_id", user_id).eq("category", cat["name"]).eq("type", "expense")
            if start_date:
                query = query.gte("date", start_date)
            if end_date:
                query = query.lte("date", end_date)
            txs = query.execute().data
            total_amount = sum(float(t["amount"]) for t in txs)
            transaction_count = len(txs)

        result.append({
            "id": cat["id"],
            "user_id": cat["user_id"],
            "name": cat["name"],
            "icon": cat.get("icon", "📌"),
            "category_type": cat.get("category_type"),
            "subcategories": cat.get("subcategories") or [],
            "total_amount": total_amount,
            "transaction_count": transaction_count,
            "created_at": cat.get("created_at"),
            "updated_at": cat.get("updated_at"),
        })

    return result


@router.post("")
def create_category(category: CategoryCreate, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    existing = sb.table("categories").select("id").eq("user_id", user_id).eq("name", category.name).execute().data
    if existing:
        raise HTTPException(status_code=400, detail="Category with this name already exists")

    result = sb.table("categories").insert({
        "user_id": user_id,
        "name": category.name,
        "icon": category.icon,
        "category_type": category.category_type,
    }).execute()
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create category")
    return result.data[0]


@router.put("/{category_id}")
def update_category(category_id: int, category_update: CategoryUpdate, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    existing = sb.table("categories").select("*").eq("id", category_id).eq("user_id", user_id).execute().data
    if not existing:
        raise HTTPException(status_code=404, detail="Category not found")

    update_data = {}
    if category_update.name is not None:
        conflict = sb.table("categories").select("id").eq("user_id", user_id).eq("name", category_update.name).neq("id", category_id).execute().data
        if conflict:
            raise HTTPException(status_code=400, detail="Category with this name already exists")
        update_data["name"] = category_update.name
    if category_update.icon is not None:
        update_data["icon"] = category_update.icon
    if category_update.category_type is not None:
        update_data["category_type"] = category_update.category_type

    result = sb.table("categories").update(update_data).eq("id", category_id).execute()
    return result.data[0] if result.data else existing[0]


@router.delete("/{category_id}")
def delete_category(category_id: int, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    existing = sb.table("categories").select("id").eq("id", category_id).eq("user_id", user_id).execute().data
    if not existing:
        raise HTTPException(status_code=404, detail="Category not found")
    sb.table("categories").delete().eq("id", category_id).execute()
    return {"message": "Category deleted successfully"}


# SUBCATEGORIES

@router.get("/{category_id}/subcategories")
def get_subcategories(category_id: int, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    cat = sb.table("categories").select("id").eq("id", category_id).eq("user_id", user_id).execute().data
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")
    return sb.table("subcategories").select("*").eq("category_id", category_id).execute().data


@router.post("/{category_id}/subcategories")
def create_subcategory(category_id: int, subcategory: SubcategoryCreate, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    cat = sb.table("categories").select("id").eq("id", category_id).eq("user_id", user_id).execute().data
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")

    existing = sb.table("subcategories").select("id").eq("category_id", category_id).eq("name", subcategory.name).execute().data
    if existing:
        raise HTTPException(status_code=400, detail="Subcategory with this name already exists")

    result = sb.table("subcategories").insert({"category_id": category_id, "name": subcategory.name, "icon": subcategory.icon}).execute()
    if not result.data:
        raise HTTPException(status_code=500, detail="Failed to create subcategory")
    return result.data[0]


@router.put("/{category_id}/subcategories/{subcategory_id}")
def update_subcategory(category_id: int, subcategory_id: int, subcategory_update: SubcategoryUpdate, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    cat = sb.table("categories").select("id").eq("id", category_id).eq("user_id", user_id).execute().data
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")

    existing = sb.table("subcategories").select("*").eq("id", subcategory_id).eq("category_id", category_id).execute().data
    if not existing:
        raise HTTPException(status_code=404, detail="Subcategory not found")

    update_data = {}
    if subcategory_update.name is not None:
        conflict = sb.table("subcategories").select("id").eq("category_id", category_id).eq("name", subcategory_update.name).neq("id", subcategory_id).execute().data
        if conflict:
            raise HTTPException(status_code=400, detail="Subcategory with this name already exists")
        update_data["name"] = subcategory_update.name
    if subcategory_update.icon is not None:
        update_data["icon"] = subcategory_update.icon

    result = sb.table("subcategories").update(update_data).eq("id", subcategory_id).execute()
    return result.data[0] if result.data else existing[0]


@router.delete("/{category_id}/subcategories/{subcategory_id}")
def delete_subcategory(category_id: int, subcategory_id: int, user_id: str = Depends(verify_token)):
    sb = get_supabase()
    cat = sb.table("categories").select("id").eq("id", category_id).eq("user_id", user_id).execute().data
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")
    existing = sb.table("subcategories").select("id").eq("id", subcategory_id).eq("category_id", category_id).execute().data
    if not existing:
        raise HTTPException(status_code=404, detail="Subcategory not found")
    sb.table("subcategories").delete().eq("id", subcategory_id).execute()
    return {"message": "Subcategory deleted successfully"}
