
import util
from flask import Flask, render_template, request
import base64
from PIL import Image
import io

app = Flask(__name__)

@app.route('/classify_image', methods = ["GET"])

def hello_world():
    return render_template('index.html')


@app.route('/classify_image', methods = ["POST"])

# def base64ToString(a):
#      return base64.b64decode(a).decode('utf-8')

def predict():
    imagefile = request.files['imagefile'] #

    # we are receiving the file as an image fileobject, this needs to be converted to base64 string so we could pass to util, but debugger says only bytes are allowed. So need to find a way around

    #image_read = imagefile.read()

    #imageb64 = base64.encodestring(image_read)

    #image_bytes = Image.open(io.BytesIO(imagefile))


    #image_path = "./" + imagefile.filename
    #image_path = "./" + imagefile.filename

    image_b64 = base64.b64encode(imagefile.read())



    #base64_img_bytes = (imagefile.read()).encode('utf-8')

    #image_string = image_b64.decode('utf-8')


    #my_string = base64.b64encode(img_file.read())

    #print(my_string)

    #imagefile.save(image_path)

    util.load_saved_artifacts()


    # typ1 = type(image_b64)
    # typ = type(image_string)


    predicted = util.classify_image(image_b64, None)
    #predicted = image_b64


    return render_template("index.html", prediction = predicted)

if __name__ == "__main__":

    app.run(port = 3000, debug = True)