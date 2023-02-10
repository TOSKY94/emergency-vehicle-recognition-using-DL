# emergency-vehicle-recognition-using-DL
A Deep Learning model for emergency vehicle recognition.
A custom architure was used first then a pretrained one (vgg16) and performance compared. An API is then built using python FastAPI framework which accepts image as load and return prediction as response.


## Requirements
tensorflow
opencv-python
fastAPI

## API Testing
run > uvicorn main:app --reload
