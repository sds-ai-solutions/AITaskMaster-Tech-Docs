# **AITaskMaster Initial Development**
Great! Now that we have our development environment set up, let's begin the initial development of AITaskMaster. We'll focus on setting up the core components of your application.

## **1. Database Models**
Let's create our database models using SQLAlchemy. Create a new file _backend/app/models/models.py:_

``` { .yaml .copy }
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    tasks = relationship("Task", back_populates="owner")

class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String)
    due_date = Column(DateTime)
    priority = Column(Float)
    status = Column(String)
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="tasks")
```
  
!!! tip "AI Tip"
	The 'priority' field is a float to allow for precise AI-driven prioritization. Consider adding more fields that can help the AI better prioritize tasks, such as estimated time, complexity, or tags.

## **2. Database Connection**
Create a new file _backend/app/database.py_ to handle database connections:

``` { .yaml .copy }
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "postgresql://aitaskuser:your_secure_password@localhost/aitaskmaster"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```
  
!!! tip "AI Tip"
	In a production environment, you should use environment variables for the database URL to keep sensitive information secure.

## **3. API Routers**
Let's create some basic API endpoints. Create a new file _backend/app/routers/tasks.py:_

``` { .yaml .copy }
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from .. import models, schemas
from ..database import get_db

router = APIRouter()

@router.post("/tasks/", response_model=schemas.Task)
def create_task(task: schemas.TaskCreate, db: Session = Depends(get_db)):
    db_task = models.Task(**task.dict())
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

@router.get("/tasks/", response_model=List[schemas.Task])
def read_tasks(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    tasks = db.query(models.Task).offset(skip).limit(limit).all()
    return tasks

@router.get("/tasks/{task_id}", response_model=schemas.Task)
def read_task(task_id: int, db: Session = Depends(get_db)):
    db_task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return db_task

@router.put("/tasks/{task_id}", response_model=schemas.Task)
def update_task(task_id: int, task: schemas.TaskUpdate, db: Session = Depends(get_db)):
    db_task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    for var, value in vars(task).items():
        setattr(db_task, var, value) if value else None
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

@router.delete("/tasks/{task_id}", response_model=schemas.Task)
def delete_task(task_id: int, db: Session = Depends(get_db)):
    db_task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    db.delete(db_task)
    db.commit()
    return db_task
```
  
!!! tip "AI Tip"
	These are basic CRUD operations. As you develop your AI features, you might want to add endpoints for task prioritization or intelligent task suggestions.

## **4. Pydantic Schemas**
Create a new file _backend/app/schemas.py_ for Pydantic models:

``` { .yaml .copy }
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class TaskBase(BaseModel):
    title: str
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    priority: Optional[float] = None
    status: Optional[str] = None

class TaskCreate(TaskBase):
    pass

class TaskUpdate(TaskBase):
    title: Optional[str] = None

class Task(TaskBase):
    id: int
    owner_id: int

    class Config:
        orm_mode = True

class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    tasks: list[Task] = []

    class Config:
        orm_mode = True
```

## **5. Update Main FastAPI App**
Update your _backend/main.py_ file to include the new router:

``` { .yaml .copy }
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import tasks
from app.database import engine
from app import models

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tasks.router)

@app.get("/")
async def root():
    return {"message": "Welcome to AITaskMaster API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## **6. Frontend Development**
For the Vue.js frontend, let's create a basic component to list tasks. Create a new file _frontend/src/components/TaskList.vue:_

``` { .yaml .copy }






```
  
!!! tip "AI Tip"
	This is a basic component. You'll want to add more features like adding tasks, updating priorities, and visualizing AI-driven insights.

## **7. AI Integration**
Let's create a basic task prioritization service. Create a new file _backend/app/services/ai_prioritizer.py:_

``` { .yaml .copy }
import tensorflow as tf
import numpy as np

class AIPrioritizer:
    def __init__(self, model_path='ai/models/task_priority_model'):
        self.model = tf.keras.models.load_model(model_path)

    def prioritize_task(self, task_features):
        # Assuming task_features is a list of numerical features
        features = np.array([task_features])
        priority = self.model.predict(features)[0][0]
        return float(priority)

# Usage in your task creation/update endpoint:
# prioritizer = AIPrioritizer()
# task.priority = prioritizer.prioritize_task([feature1, feature2, feature3])
```
  
!!! tip "AI Tip"
	This is a simplistic implementation. In a real-world scenario, you'd need to carefully choose and preprocess your task features, and possibly use more advanced models like transformers for natural language understanding of task descriptions.

## **Next Steps**
1. Implement user authentication and authorization
2. Develop more comprehensive AI models for task prioritization
3. Create frontend components for task creation, updating, and deletion
4. Implement real-time updates using WebSockets
5. Add data visualization for task analytics
6. Set up unit and integration tests
7. Implement error handling and logging

This initial development sets up the core structure of your AITaskMaster application. You now have a basic backend API, database models, and a simple frontend component. The next phase will involve refining these components and adding more sophisticated AI-driven features.

[Proceed to Advanced Features](adv-feature.md){ .md-button }
