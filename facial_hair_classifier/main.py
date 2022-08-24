
import util
from flask import Flask, render_template, request
import base64

app = Flask(__name__)

@app.route('/classify_image', methods = ["GET"])

def hello_world():
    return render_template('index.html')


@app.route('/classify_image', methods = ["POST"])

# def base64ToString(a):
#      return base64.b64decode(a).decode('utf-8')

def predict():
    imagefile = request.files['imagefile'] #
    image_path = "./test_images/" + imagefile.filename
    #image_b64 = base64.b64encode(imagefile.read())
    #image_string = image_b64.decode('utf-8')

    imagefile.save(image_path)

    util.load_saved_artifacts()


    # typ1 = type(image_b64)
    # typ = type(image_string)


    predicted = util.classify_image(None, image_path)
    #predicted = image_b64


    return render_template("index.html", prediction = predicted)

if __name__ == "__main__":

    app.run(port = 3000, debug = True)