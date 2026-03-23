"""Categorie di default per nuovi utenti."""

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
    {"name": "Altro", "icon": "📌", "category_type": None},
]


def create_default_categories_if_needed(supabase, user_id: str) -> None:
    """Crea le categorie di default se l'utente non ne ha ancora."""
    result = supabase.table("categories").select("id").eq("user_id", user_id).limit(1).execute()
    if result.data:
        return

    rows = [{"user_id": user_id, **cat} for cat in DEFAULT_CATEGORIES]
    supabase.table("categories").insert(rows).execute()
