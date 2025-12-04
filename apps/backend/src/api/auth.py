from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from src import schemas
from src.database import get_db
from src import crud
from src.utils.security import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=schemas.UserRead, status_code=status.HTTP_201_CREATED)
def register(user_in: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user. Returns the created user (without password).
    """
    existing = crud.get_user_by_email(db, user_in.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    user = crud.create_user(db, name=user_in.name, email=user_in.email, password=user_in.password)
    return user

@router.post("/login", response_model=schemas.Token)
def login(form: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    Login endpoint. Expects JSON with email and password (name is ignored for login).
    Returns a JWT access token.
    """
    user = crud.authenticate_user(db, email=form.email, password=form.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token({"sub": str(user.id)})
    return {"access_token": token, "token_type": "bearer"}
