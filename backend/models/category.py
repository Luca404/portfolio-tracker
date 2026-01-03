from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from .base import Base


class CategoryModel(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    icon = Column(String, nullable=False, default="💰")
    category_type = Column(String, nullable=True)  # 'expense', 'income', 'investment', 'transfer', or null for all
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("UserModel", back_populates="categories")
    subcategories = relationship("SubcategoryModel", back_populates="category", cascade="all, delete-orphan")


class SubcategoryModel(Base):
    __tablename__ = "subcategories"

    id = Column(Integer, primary_key=True, index=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    name = Column(String, nullable=False)
    icon = Column(String, nullable=False, default="📌")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    category = relationship("CategoryModel", back_populates="subcategories")
