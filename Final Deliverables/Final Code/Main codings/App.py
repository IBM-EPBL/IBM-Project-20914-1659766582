
from flask import Flask, render_template, flash, request, session
from flask import render_template, redirect, url_for, request
import cv2
import numpy as np

from keras.models import load_model

model = load_model('mnist.h5')
import smtplib



app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

app.config['DEBUG']


@app.route("/")
def homepage():
    return render_template('index.html')
@app.route("/UserLogin")
def UserLogin():
    return render_template('UserLogin.html')


@app.route("/start", methods=['GET', 'POST'])
def start():
    error = None
    if request.method == 'POST':

        file = request.files['fileupload']
        file.save('static/Out/Test.jpg')

        #image = cv2.imread('static/Out/Test.jpg')
        #print(import_file_path)
        filename = 'Output/Out/Test.jpg'
        #cv2.imwrite(filename, image)
        #print("After saving image:")



        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # cv2.imshow('contours', contours)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # make a rectangle box around each curve
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

            # Cropping out the digit from the image corresponding to the current contours in the for loop
            digit = th[y:y + h, x:x + w]

            # Resizing that digit to (18, 18)
            resized_digit = cv2.resize(digit, (18, 18))
            # cv2.imshow('resized_digit', resized_digit)

            # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
            padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

            digit = padded_digit.reshape(1, 28, 28, 1)
            digit = digit / 255.0

            pred = model.predict([digit])[0]
            final_pred = np.argmax(pred)
            # print(final_pred)

            data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 1
            cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)

        cv2.imshow('Predictions', image)
        cv2.waitKey(0)




    return render_template('UserLogin.html')













if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
