from sqlalchemy import Column, DateTime, Integer, String, text
from sqlalchemy.orm import relationship
from .base import Base


class UserModel(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))

    portfolios = relationship("PortfolioModel", back_populates="user", cascade="all, delete")
