from flask import Flask, render_template, request, json, flash, redirect, url_for, session, logging
from flask_mysqldb import MySQL
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
from functools import wraps
import MySQLdb


app = Flask(__name__)
mysql = MySQL(app)
conn = MySQLdb.connect(host="localhost", user="root", password="TishuPaper", db="uRECO")
mysql.init_app(app)



@app.route('/')
def home():
    return render_template("home.html")


@app.route('/home')
def start_page():
    return render_template("home.html")





@app.route('/show_sign_up')
def showSignUp():
    return render_template('sign_up.html')


class RegisterForm(Form):
    name = StringField('name', [validators.Length(min=1, max=50)])
    username = StringField('username', [validators.Length(min=4, max=25)])
    email = StringField('email', [validators.Length(min=6, max=50)])
    password = PasswordField('password')
    c_password = PasswordField('c_password')


@app.route('/sign_up', methods=['GET', 'POST'])
def sign_up():
    form = RegisterForm(request.form)

    name = form.name.data
    username = form.username.data
    email = form.email.data
    password = form.password.data
    c_password = form.c_password.data

    if request.method == 'POST' and form.validate() and password == c_password:
        name = form.name.data
        username = form.username.data
        email = form.email.data
        password = form.password.data
        # password = sha256_crypt.encrypt(str(form.password.data))

        # Create cursor
        cur = conn.cursor()
        cur.execute("SELECT user_username FROM user WHERE user_username ='" + username + "'")
        data = cur.fetchall()

        print(len(data))

        if len(data) is 0:
            cur.execute("INSERT INTO user(user_name, user_email, user_username, user_password) VALUES(%s, %s, %s, %s)",
                        (name, email, username, password))

            # Commit to DB
            conn.commit()

            # Close connection
            cur.close()
            flash('You are now registered and can log in', 'success')

            return render_template("home.html")

    return render_template('sign_up.html', form=form)







@app.route('/show_sign_in')
def showSignIn():
    return render_template('sign_in.html')


@app.route('/sign_in', methods=['POST'])
def sign_in():
    username = request.form["user"]
    password = request.form["password"]

    cursor = conn.cursor()
    cursor.execute(
        "SELECT user_username FROM user WHERE user_username ='" + username + "' and user_password ='" + password + "'")
    temp = cursor.fetchone()

    if temp is None:
        return render_template('sign_in.html')
    else:
        return render_template("logged_in.html")







# @author SAIF MAHMUD
@app.route('/facial_expression_recognition')
def feature1():
    return facial_expression_recognition()
def facial_expression_recognition():
    from keras.models import model_from_json
    from keras.optimizers import SGD
    import numpy as np
    from time import sleep

    model = model_from_json(open(
        '/home/amit-roy/PycharmProjects/uRECO/FacialExpressionClassifer/models/Face_model_architecture.json').read())

    # model.load_weights('_model_weights.h5')
    model.load_weights('/home/amit-roy/PycharmProjects/uRECO/FacialExpressionClassifer/models/Face_model_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    def extract_face_features(gray, detected_face, offset_coefficients):
        (x, y, w, h) = detected_face
        # print x , y, w ,h
        horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
        vertical_offset = np.int(np.floor(offset_coefficients[1] * h))

        extracted_face = gray[y + vertical_offset:y + h,
                         x + horizontal_offset:x - horizontal_offset + w]

        # print extracted_face.shape
        new_extracted_face = zoom(extracted_face, (48. / extracted_face.shape[0],
                                                   48. / extracted_face.shape[1]))
        new_extracted_face = new_extracted_face.astype(np.float32)
        new_extracted_face /= float(new_extracted_face.max())
        return new_extracted_face

    from scipy.ndimage import zoom

    def detect_face(frame):
        cascPath = "/home/amit-roy/PycharmProjects/uRECO/FacialExpressionClassifer/models/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(48, 48),
            flags=cv2.FONT_HERSHEY_COMPLEX
        )
        return gray, detected_faces

    import cv2

    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        # sleep(0.8)
        ret, frame = video_capture.read()

        # detect faces
        gray, detected_faces = detect_face(frame)

        face_index = 0

        # predict output
        for face in detected_faces:
            (x, y, w, h) = face
            if w > 100:
                # draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # extract features
                extracted_face = extract_face_features(gray, face, (0.075, 0.05))  # (0.075, 0.05)

                # predict smile
                prediction_result = model.predict_classes(extracted_face.reshape(1, 48, 48, 1))

                # draw extracted face in the top right corner
                frame[face_index * 48: (face_index + 1) * 48, -49:-1, :] = cv2.cvtColor(extracted_face * 255,
                                                                                        cv2.COLOR_GRAY2RGB)

                # annotate main image with a label
                if prediction_result == 3:
                    cv2.putText(frame, "Happy!!", (x, y), cv2.FONT_ITALIC, 2, (255, 255, 255), 5)
                elif prediction_result == 0:
                    cv2.putText(frame, "Angry", (x, y), cv2.FONT_ITALIC, 2, (255, 255, 255), 5)
                elif prediction_result == 1:
                    cv2.putText(frame, "Disgust", (x, y), cv2.FONT_ITALIC, 2, (255, 255, 255), 5)
                elif prediction_result == 2:
                    cv2.putText(frame, "Fear", (x, y), cv2.FONT_ITALIC, 2, (255, 255, 255), 5)
                elif prediction_result == 4:
                    cv2.putText(frame, "Sad", (x, y), cv2.FONT_ITALIC, 2, (255, 255, 255), 5)
                elif prediction_result == 5:
                    cv2.putText(frame, "Surprise", (x, y), cv2.FONT_ITALIC, 2, (255, 255, 255), 5)
                else:
                    cv2.putText(frame, "Neutral", (x, y), cv2.FONT_ITALIC, 2, (255, 255, 255), 5)

                # increment counter
                face_index += 1

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Press 'Q' to Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    return render_template("logged_in.html")






# @author TAUHID TANJIM
@app.route('/sign_language_detection')
def feature2():
    return sign_language_detection()
def sign_language_detection():
    import cv2, pickle
    import numpy as np
    import tensorflow as tf
    from cnn_tf import cnn_model_fn
    import os
    import sqlite3, pyttsx3
    from keras.models import load_model
    from threading import Thread

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    model = load_model('/home/amit-roy/PycharmProjects/uRECO/ASL/cnn_model_keras2.h5')

    def get_hand_hist():
        with open("/home/amit-roy/PycharmProjects/uRECO/ASL/hist", "rb") as f:
            hist = pickle.load(f)
        return hist

    def get_image_size():
        img = cv2.imread('/home/amit-roy/PycharmProjects/uRECO/ASL/gestures/0/100.jpg', 0)
        return img.shape

    image_x, image_y = get_image_size()

    def keras_process_image(img):
        img = cv2.resize(img, (image_x, image_y))
        img = np.array(img, dtype=np.float32)
        img = np.reshape(img, (1, image_x, image_y, 1))
        return img

    def keras_predict(model, image):
        processed = keras_process_image(image)
        pred_probab = model.predict(processed)[0]
        pred_class = list(pred_probab).index(max(pred_probab))
        return max(pred_probab), pred_class

    def get_pred_text_from_db(pred_class):
        conn = sqlite3.connect("/home/amit-roy/PycharmProjects/uRECO/ASL/gesture_db.db")
        cmd = "SELECT g_name FROM gesture WHERE g_id=" + str(pred_class)
        cursor = conn.execute(cmd)
        for row in cursor:
            return row[0]

    def get_pred_from_contour(contour, thresh):
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        save_img = thresh[y1:y1 + h1, x1:x1 + w1]
        text = ""
        if w1 > h1:
            save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0, cv2.BORDER_CONSTANT,
                                          (0, 0, 0))
        elif h1 > w1:
            save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2), cv2.BORDER_CONSTANT,
                                          (0, 0, 0))
        pred_probab, pred_class = keras_predict(model, save_img)
        if pred_probab * 100 > 70:
            text = get_pred_text_from_db(pred_class)
        return text

    def get_operator(pred_text):
        try:
            pred_text = int(pred_text)
        except:
            return ""
        operator = ""
        if pred_text == 1:
            operator = "+"
        elif pred_text == 2:
            operator = "-"
        elif pred_text == 3:
            operator = "*"
        elif pred_text == 4:
            operator = "/"
        elif pred_text == 5:
            operator = "%"
        elif pred_text == 6:
            operator = "**"
        elif pred_text == 7:
            operator = ">>"
        elif pred_text == 8:
            operator = "<<"
        elif pred_text == 9:
            operator = "&"
        elif pred_text == 0:
            operator = "|"
        return operator

    hist = get_hand_hist()
    x, y, w, h = 300, 100, 300, 300
    is_voice_on = True

    def get_img_contour_thresh(img):
        img = cv2.flip(img, 1)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cv2.filter2D(dst, -1, disc, dst)
        blur = cv2.GaussianBlur(dst, (11, 11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        thresh = cv2.merge((thresh, thresh, thresh))
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        thresh = thresh[y:y + h, x:x + w]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
        return img, contours, thresh

    def say_text(text):
        if not is_voice_on:
            return
        while engine._inLoop:
            pass
        engine.say(text)
        engine.runAndWait()

    def calculator_mode(cam):
        global is_voice_on
        flag = {"first": False, "operator": False, "second": False, "clear": False}
        count_same_frames = 0
        first, operator, second = "", "", ""
        pred_text = ""
        calc_text = ""
        info = "Enter first number"
        Thread(target=say_text, args=(info,)).start()
        count_clear_frames = 0
        while True:
            img = cam.read()[1]
            img, contours, thresh = get_img_contour_thresh(img)
            old_pred_text = pred_text
            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(contour) > 10000:
                    pred_text = get_pred_from_contour(contour, thresh)
                    if old_pred_text == pred_text:
                        count_same_frames += 1
                    else:
                        count_same_frames = 0

                    if pred_text == "C":
                        if count_same_frames > 5:
                            count_same_frames = 0
                            first, second, operator, pred_text, calc_text = '', '', '', '', ''
                            flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
                            info = "Enter first number"
                            Thread(target=say_text, args=(info,)).start()

                    elif pred_text == "Best of Luck " and count_same_frames > 15:
                        count_same_frames = 0
                        if flag['clear']:
                            first, second, operator, pred_text, calc_text = '', '', '', '', ''
                            flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
                            info = "Enter first number"
                            Thread(target=say_text, args=(info,)).start()
                        elif second != '':
                            flag['second'] = True
                            info = "Clear screen"
                            # Thread(target=say_text, args=(info,)).start()
                            second = ''
                            flag['clear'] = True
                            calc_text += "= " + str(eval(calc_text))
                            if is_voice_on:
                                speech = calc_text
                                speech = speech.replace('-', ' minus ')
                                speech = speech.replace('/', ' divided by ')
                                speech = speech.replace('**', ' raised to the power ')
                                speech = speech.replace('*', ' multiplied by ')
                                speech = speech.replace('%', ' mod ')
                                speech = speech.replace('>>', ' bitwise right shift ')
                                speech = speech.replace('<<', ' bitwise leftt shift ')
                                speech = speech.replace('&', ' bitwise and ')
                                speech = speech.replace('|', ' bitwise or ')
                                Thread(target=say_text, args=(speech,)).start()
                        elif first != '':
                            flag['first'] = True
                            info = "Enter operator"
                            Thread(target=say_text, args=(info,)).start()
                            first = ''

                    elif pred_text != "Best of Luck " and pred_text.isnumeric():
                        if flag['first'] == False:
                            if count_same_frames > 15:
                                count_same_frames = 0
                                Thread(target=say_text, args=(pred_text,)).start()
                                first += pred_text
                                calc_text += pred_text
                        elif flag['operator'] == False:
                            operator = get_operator(pred_text)
                            if count_same_frames > 15:
                                count_same_frames = 0
                                flag['operator'] = True
                                calc_text += operator
                                info = "Enter second number"
                                Thread(target=say_text, args=(info,)).start()
                                operator = ''
                        elif flag['second'] == False:
                            if count_same_frames > 15:
                                Thread(target=say_text, args=(pred_text,)).start()
                                second += pred_text
                                calc_text += pred_text
                                count_same_frames = 0

            if count_clear_frames == 30:
                first, second, operator, pred_text, calc_text = '', '', '', '', ''
                flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
                info = "Enter first number"
                Thread(target=say_text, args=(info,)).start()
                count_clear_frames = 0

            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blackboard, "Calculator Mode", (100, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0))
            cv2.putText(blackboard, "Predicted text- " + pred_text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1,
                        (255, 255, 0))
            cv2.putText(blackboard, "Operator " + operator, (30, 140), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 127))
            cv2.putText(blackboard, calc_text, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
            cv2.putText(blackboard, info, (30, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255))
            if is_voice_on:
                cv2.putText(blackboard, "Voice on", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
            else:
                cv2.putText(blackboard, "Voice off", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            res = np.hstack((img, blackboard))
            cv2.imshow("Recognizing gesture", res)
            cv2.imshow("thresh", thresh)
            keypress = cv2.waitKey(1)
            if keypress == ord('q') or keypress == ord('t'):
                break
            if keypress == ord('v') and is_voice_on:
                is_voice_on = False
            elif keypress == ord('v') and not is_voice_on:
                is_voice_on = True

        if keypress == ord('t'):
            return 1
        else:
            return 0

    def text_mode(cam):
        global is_voice_on
        is_voice_on = True
        text = ""
        word = ""
        count_same_frame = 0
        while True:
            img = cam.read()[1]
            img, contours, thresh = get_img_contour_thresh(img)
            old_text = text
            if len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(contour) > 10000:
                    text = get_pred_from_contour(contour, thresh)
                    if old_text == text:
                        count_same_frame += 1
                    else:
                        count_same_frame = 0

                    if count_same_frame > 20:
                        if len(text) == 1:
                            Thread(target=say_text, args=(text,)).start()
                        word = word + text
                        if word.startswith('I/Me '):
                            word = word.replace('I/Me ', 'I ')
                        elif word.endswith('I/Me '):
                            word = word.replace('I/Me ', 'me ')
                        count_same_frame = 0

                elif cv2.contourArea(contour) < 1000:
                    if word != '':
                        # print('yolo')
                        # say_text(text)
                        Thread(target=say_text, args=(word,)).start()
                    text = ""
                    word = ""
            else:
                if word != '':
                    # print('yolo1')
                    # say_text(text)
                    Thread(target=say_text, args=(word,)).start()
                text = ""
                word = ""
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blackboard, "Text Mode", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0))
            cv2.putText(blackboard, "Predicted text- " + text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
            cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
            if is_voice_on:
                cv2.putText(blackboard, "Voice on", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
            else:
                cv2.putText(blackboard, "Voice off", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            res = np.hstack((img, blackboard))
            cv2.imshow("Recognizing gesture", res)
            cv2.imshow("thresh", thresh)
            keypress = cv2.waitKey(1)
            if keypress == ord('q') or keypress == ord('c'):
                break
            if keypress == ord('v') and is_voice_on:
                is_voice_on = False
            elif keypress == ord('v') and not is_voice_on:
                is_voice_on = True

        if keypress == ord('c'):
            return 2
        else:
            return 0

    def recognize():
        cam = cv2.VideoCapture(1)
        if cam.read()[0] == False:
            cam = cv2.VideoCapture(0)
        text = ""
        word = ""
        count_same_frame = 0
        keypress = 1
        while True:
            if keypress == 1:
                keypress = text_mode(cam)
            elif keypress == 2:
                keypress = calculator_mode(cam)
            else:
                break

    keras_predict(model, np.zeros((50, 50), dtype=np.uint8))
    recognize()
    return render_template("logged_in.html")





# @author AMIT ROY
@app.route('/virtual_keyboard')
def feature3():
    return virtual_keyboard()
def virtual_keyboard():
    # return keyboard.main()
    import cv2
    import pickle
    import numpy as np
    import pyautogui as gui

    with open("/home/amit-roy/PycharmProjects/uRECO/VirtualKeyboard/range.pickle",
              "rb") as f:  # range.pickle is generated by range-detector.py
        t = pickle.load(f)
    cam = cv2.VideoCapture(1)

    if cam.read()[0] == False:
        cam = cv2.VideoCapture(0)

    hsv_lower = np.array([t[0], t[1], t[2]])
    hsv_upper = np.array([t[3], t[4], t[5]])

    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)  # width of video captured by the webcam
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)  # height of the video captured by the webcam
    max_keys_in_a_row = 10  # max number of keys in any row is 10 i.e the first row which contains qwertyuiop
    key_width = int(
        width / max_keys_in_a_row)  # width of one key. width is divided by 10 as the max number of keys in a single row is 10.

    def get_keys():
        row1_key_width = key_width * 10  # width of first row of keys
        row2_key_width = key_width * 9  # width of second row
        row3_key_width = key_width * 7  # width of third row
        row4_key_width = key_width * 5  # width of spacebar
        row_keys = []  # stores the keys along with its 2 corner coordinates and the center coordinate

        # for the first row
        x1, y1 = 0, int((
                                height - key_width * 4) / 2)  # 4 is due to the fact that we will have 4 rows. y1 is set such that the whole keyboard has equal margin on both top and bottom
        x2, y2 = key_width + x1, key_width + y1
        c1, c2 = x1, y1  # copying x1, y1
        c = 0
        keys = "QWERTYUIOP"
        for i in range(len(keys)):
            row_keys.append([keys[c], (x1, y1), (x2, y2), (int((x2 + x1) / 2) - 5, int((y2 + y1) / 2) + 10)])
            x1 += key_width
            x2 += key_width
            c += 1
        x1, y1 = c1, c2  # copying back from c1, c2

        # for second row
        x1, y1 = int((
                             row1_key_width - row2_key_width) / 2) + x1, y1 + key_width  # x1 is set such that it leaves equal margin on both left and right side
        x2, y2 = key_width + x1, key_width + y1
        c1, c2 = x1, y1
        c = 0
        keys = "ASDFGHJKL"
        for i in range(len(keys)):
            row_keys.append([keys[c], (x1, y1), (x2, y2), (int((x2 + x1) / 2) - 5, int((y2 + y1) / 2) + 10)])
            x1 += key_width
            x2 += key_width
            c += 1
        x1, y1 = c1, c2

        # for third row
        x1, y1 = int((row2_key_width - row3_key_width) / 2) + x1, y1 + key_width
        x2, y2 = key_width + x1, key_width + y1
        c1, c2 = x1, y1
        c = 0
        keys = "ZXCVBNM"
        for i in range(len(keys)):
            row_keys.append([keys[c], (x1, y1), (x2, y2), (int((x2 + x1) / 2) - 5, int((y2 + y1) / 2) + 10)])
            x1 += key_width
            x2 += key_width
            c += 1
        x1, y1 = c1, c2

        # for the space bar
        x1, y1 = int((row3_key_width - row4_key_width) / 2) + x1, y1 + key_width
        x2, y2 = 5 * key_width + x1, key_width + y1
        c1, c2 = x1, y1
        c = 0
        keys = " "
        for i in range(len(keys)):
            row_keys.append([keys[c], (x1, y1), (x2, y2), (int((x2 + x1) / 2) - 5, int((y2 + y1) / 2) + 10)])
            x1 += key_width
            x2 += key_width
            c += 1
        x1, y1 = c1, c2

        return row_keys

    def do_keypress(img, center, row_keys_points):
        # this fuction presses a key and marks the pressed key with blue color
        for row in row_keys_points:
            arr1 = list(np.int0(np.array(center) >= np.array(
                row[1])))  # center of the contour has greater value than the top left corner point of a key
            arr2 = list(np.int0(np.array(center) <= np.array(
                row[2])))  # center of the contour has less value than the bottom right corner point of a key
            if arr1 == [1, 1] and arr2 == [1, 1]:
                gui.press(row[0])
                cv2.fillConvexPoly(img, np.array([np.array(row[1]), \
                                                  np.array([row[1][0], row[2][1]]), \
                                                  np.array(row[2]), \
                                                  np.array([row[2][0], row[1][1]])]), \
                                   (255, 0, 0))
        return img

    def v_keyboard_main():
        row_keys_points = get_keys()
        new_area, old_area = 0, 0
        c, c2 = 0, 0  # c stores the number of iterations for calculating the difference b/w present area and previous area
        # c2 stores the number of iterations for calculating the difference b/w present center and previous center
        flag_keypress = False  # if a key is pressed then this flag is True
        while True:
            img = cam.read()[1]
            img = cv2.flip(img, 1)
            imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(imgHSV, hsv_lower, hsv_upper)
            blur = cv2.medianBlur(mask, 15)
            blur = cv2.GaussianBlur(blur, (5, 5), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

            if len(contours) > 0:
                cnt = max(contours, key=cv2.contourArea)

                if cv2.contourArea(cnt) > 350:
                    # draw a rectangle and a center
                    rect = cv2.minAreaRect(cnt)
                    center = list(rect[0])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.circle(img, tuple(np.int0(center)), 2, (0, 255, 0), 2)
                    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

                    # calculation of difference of area and center
                    new_area = cv2.contourArea(cnt)
                    new_center = np.int0(center)
                    if c == 0:
                        old_area = new_area
                    c += 1
                    diff_area = 0
                    if c > 3:  # after every 3rd iteration difference of area is calculated
                        diff_area = new_area - old_area
                        c = 0
                    if c2 == 0:
                        old_center = new_center
                    c2 += 1
                    diff_center = np.array([0, 0])
                    if c2 > 5:  # after every 5th iteration difference of center is claculated
                        diff_center = new_center - old_center
                        c2 = 0

                    # setting some thresholds
                    center_threshold = 10
                    area_threshold = 200
                    if abs(diff_center[0]) < center_threshold or abs(diff_center[1]) < center_threshold:
                        print(diff_area)
                        if diff_area > area_threshold and flag_keypress == False:
                            img = do_keypress(img, new_center, row_keys_points)
                            flag_keypress = True
                        elif diff_area < -(area_threshold) and flag_keypress == True:
                            flag_keypress = False
                else:
                    flag_keypress = False
            else:
                flag_keypress = False

            # displaying the keyboard
            for key in row_keys_points:
                cv2.putText(img, key[0], key[3], cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0))
                cv2.rectangle(img, key[1], key[2], (255, 0, 0), thickness=2)

            cv2.imshow("img", img)

            if cv2.waitKey(1) == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

    v_keyboard_main()
    return render_template("logged_in.html")





if __name__ == '__main__':
    app.secret_key = 'TishuPaper'
    app.run()