import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps
import numpy as np
import tf_keras as keras
from scipy.spatial import distance
import io

model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"

IMAGE_SHAPE = (224, 224)

model = keras.Sequential([
    hub.KerasLayer(model_url, input_shape=IMAGE_SHAPE+(3,)),
])

# model = keras.layers.TFSMLayer('saved_model', call_endpoint='serving_default')

# model.export('saved_model')

def extract(file, is_from_user):
  if (is_from_user == "yes"):
    file = Image.open(io.BytesIO(file)).convert('L').resize(IMAGE_SHAPE)
  else:
     file = Image.open(file).convert('L').resize(IMAGE_SHAPE)

#   file = np.array(file)    
  file = np.stack((file,)*3, axis=-1)
  file = np.array(file)/255.0

  embedding = model.predict(file[np.newaxis, ...])
  print(embedding)
  vgg16_feature_np = np.array(embedding)
  flattended_feature = vgg16_feature_np.flatten()

  return flattended_feature

cat_image = Image.open('./2560px-A-Cat.jpg')
cat1 = extract('./2560px-A-Cat.jpg', "no")

# dc0 = distance.cdist([cat1], [cat1], metric = 'cosine')[0]

# print(dc0)
# print(f'The similiarity of cat and cat is: {format(dc0)}')
# print(type(dc0))

# print("hello")

from fastapi import FastAPI, File, UploadFile
import uvicorn
import io

app = FastAPI()


@app.get("/")
def hello():
    return {
        'message': 'Hello'
    }

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
    request_image = file.file.read()
    # image_file_pil = Image.open(io.BytesIO(request_image))
    extract_img = extract(request_image, "yes")

    result = distance.cdist([extract_img], [cat1], metric = 'cosine')[0]
    
    return {
        "status": "ok",
        "similiarity_score": result.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)