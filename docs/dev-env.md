# **AITaskMaster Development Environment Setup**
Great choices for your AITaskMaster project! Let's set up your development environment based on your selections:

- Server Type: Physical
- Operating System: Ubuntu 20.04 LTS
- Web Server: Nginx
- Database: PostgreSQL
- Backend: Python (FastAPI)
- Frontend: Vue.js
- AI Framework: TensorFlow

## **1. Server Setup**
Assuming you have Ubuntu 20.04 LTS installed on your physical server, let's start by updating the system and installing necessary dependencies:

``` { .yaml .copy }
sudo apt update && sudo apt upgrade -y
sudo apt install -y nginx postgresql postgresql-contrib python3 python3-pip nodejs npm
```
  
!!! tip "AI Tip"
	Regular system updates are crucial for security. Consider setting up automatic updates for non-critical packages.

## **2. Python Environment Setup**
Set up a virtual environment for your Python backend:

``` { .yaml .copy }
sudo apt install -y python3-venv
python3 -m venv ~/aitaskmaster-env
source ~/aitaskmaster-env/bin/activate
pip install fastapi uvicorn psycopg2-binary tensorflow
```
  
!!! tip "AI Tip"
	Virtual environments help manage dependencies for different projects effectively. Always activate the environment before working on your project.

## **3. PostgreSQL Setup**
Configure PostgreSQL for your project:

``` { .yaml .copy }
sudo -u postgres psql -c "CREATE DATABASE aitaskmaster;"
sudo -u postgres psql -c "CREATE USER aitaskuser WITH PASSWORD 'your_secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE aitaskmaster TO aitaskuser;"
```
  
!!! tip "AI Tip"
	Use a strong, unique password for your database. Consider using environment variables to store sensitive information like database credentials.

## **4. Nginx Configuration**
Create a basic Nginx configuration for your project:

``` { .yaml .copy }
sudo nano /etc/nginx/sites-available/aitaskmaster

# Add the following configuration:
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

sudo ln -s /etc/nginx/sites-available/aitaskmaster /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```
  
!!! tip "AI Tip"
	This basic configuration will proxy requests to your FastAPI application. You'll need to adjust the domain and possibly add SSL configuration for production use.

## **5. Vue.js Setup**
Set up your Vue.js frontend:

``` { .yaml .copy }
npm install -g @vue/cli
vue create aitaskmaster-frontend
cd aitaskmaster-frontend
npm install axios vuex
```
  
!!! tip "AI Tip"
	Consider using Vuex for state management in your Vue.js application, especially for handling AI-processed data and user tasks.

## **6. Project Structure**
Create the following project structure:

``` { .yaml .copy }
mkdir -p AITaskMaster/{backend,frontend,ai}
cd AITaskMaster/backend
pip freeze > requirements.txt
touch main.py
mkdir -p app/{models,routers,services}
cd ../frontend
# Move your Vue.js project here
cd ../ai
mkdir -p {models,data}
touch train.py
```
  
## **7. Initial Backend Code (main.py)**
Create a basic FastAPI application:

``` { .yaml .copy }
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to AITaskMaster API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
  
!!! tip "AI Tip"
	This basic setup includes CORS middleware to allow requests from your Vue.js frontend during development. Adjust the allowed origins for production.

## **8. TensorFlow Integration**
Create a simple TensorFlow model for task prioritization (ai/train.py):

``` { .yaml .copy }
import tensorflow as tf
import numpy as np

# Dummy data for task prioritization
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)  # Task features
y = np.array([1, 2, 3], dtype=float)  # Priority scores

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)

# Save the model
model.save('ai/models/task_priority_model')

print("Basic TensorFlow model created and saved.")
```
  
!!! tip "AI Tip"
	This is a very basic model. You'll need to develop more sophisticated models based on real task data and features relevant to prioritization.

## **Next Steps**
1. Implement database models and migrations
2. Develop API endpoints for task management
3. Create Vue.js components for the frontend
4. Integrate the TensorFlow model with your backend for task prioritization
5. Implement user authentication and authorization
6. Set up continuous integration and deployment pipelines

Your development environment is now set up! You can start building your AITaskMaster application. Remember to commit your code regularly to version control.

[Start Initial Development](init-dev.md){ .md-button }
