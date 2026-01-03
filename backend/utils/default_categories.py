"""
Utilità per creare categorie di default per nuovi utenti
"""
from sqlalchemy.orm import Session
from models import CategoryModel, UserModel


DEFAULT_CATEGORIES = [
    {"name": "Alimentari", "icon": "🍔", "category_type": "expense"},
    {"name": "Trasporti", "icon": "🚗", "category_type": "expense"},
    {"name": "Utenze", "icon": "⚡", "category_type": "expense"},
    {"name": "Svago", "icon": "🎮", "category_type": "expense"},
    {"name": "Salute", "icon": "🏥", "category_type": "expense"},
    {"name": "Shopping", "icon": "🛍️", "category_type": "expense"},
    {"name": "Investimento", "icon": "💰", "category_type": "investment"},
    {"name": "Stipendio", "icon": "💵", "category_type": "income"},
    {"name": "Bonus", "icon": "🎁", "category_type": "income"},
    {"name": "Trasferimento", "icon": "🔄", "category_type": "transfer"},
    {"name": "Altro", "icon": "📌", "category_type": None},  # Categoria generica per tutti i tipi
]

# IMPORTANTE: Questi nomi devono corrispondere esattamente a TransactionCategory enum nel frontend!


def create_default_categories(db: Session, user: UserModel) -> None:
    """
    Crea le categorie di default per un utente se non ne ha già.

    Args:
        db: Sessione database
        user: Modello utente per cui creare le categorie
    """
    # Verifica se l'utente ha già delle categorie
    existing_count = db.query(CategoryModel).filter(
        CategoryModel.user_id == user.id
    ).count()

    if existing_count > 0:
        return  # L'utente ha già delle categorie

    # Crea le categorie di default
    for cat_data in DEFAULT_CATEGORIES:
        category = CategoryModel(
            user_id=user.id,
            name=cat_data["name"],
            icon=cat_data["icon"],
            category_type=cat_data["category_type"],
        )
        db.add(category)

    db.commit()
