from fastapi import FastAPI
from validators import validator
from services import video2BlogService
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks

app = FastAPI()
origins = [
    "http://localhost:3000",  # React frontend running locally
    "https://your-frontend-domain.com",  # Example production domain
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,  # Allow cookies to be sent cross-origin
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
async def root():
    return {"result": "Hello World"}


def processBlogTask(url: str, email: str):
    video2BlogService.processBlog(url, email)

@app.get("/blogs/")
async def getBlog(url: str = 'https://www.youtube.com/watch?v=kCc8FmEb1nY',
                  email: str = 'default@email.com',
                  background_tasks: BackgroundTasks = None):
    print(url)
    try:
        validator.validateYoutubeLink(url)

    except Exception as e:
        print(e)
        return {"result": "Invalid link."}
    try:
        validator.validateEmail(email)

    except Exception as e:
        print(e)
        return {"result": "Invalid email format."}



    # video2BlogService.processBlog(url, email)
    background_tasks.add_task(processBlogTask, url, email)

    print("Returning Response")
    return {"result": "Blog is processing"}
