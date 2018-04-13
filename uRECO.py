from flask import Flask, render_template, request, json, flash, redirect, url_for, session, logging
from flask_mysqldb import MySQL
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
from functools import wraps
import MySQLdb

#from VirtualKeyboard import virtual_keyboard as keyboard



app = Flask(__name__)
mysql = MySQL(app)

conn = MySQLdb.connect(host="localhost",user="root",password="TishuPaper",db="uRECO")


#create cursor

# init MYSQL
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


# Register Form Class
class RegisterForm(Form):
    name = StringField('name', [validators.Length(min=1, max=50)])
    username = StringField('username', [validators.Length(min=4, max=25)])
    email = StringField('email', [validators.Length(min=6, max=50)])
    password = PasswordField('password')
    c_password = PasswordField('c_password')




@app.route('/sign_up', methods=['GET','POST'])
def sign_up():

        form = RegisterForm(request.form)

        name = form.name.data
        username = form.username.data
        email = form.email.data
        password = form.password.data
        c_password = form.c_password.data

        if request.method =='POST' and form.validate() and password == c_password:
            name = form.name.data
            username = form.username.data
            email = form.email.data
            password = form.password.data
            #password = sha256_crypt.encrypt(str(form.password.data))

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



        return render_template('sign_up.html',form=form)

@app.route('/show_sign_in')
def showSignIn():
    return render_template('sign_in.html')


@app.route('/sign_in', methods=['POST'])
def sign_in():
    username = request.form["user"]
    password = request.form["password"]

    cursor = conn.cursor()
    cursor.execute("SELECT user_username FROM user WHERE user_username ='" + username + "' and user_password ='"+password+"'")
    temp = cursor.fetchone()

    if temp is None:
        return render_template('sign_in.html')
    else:
        return render_template("logged_in.html")


#author SAIF MAHMUD
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





@app.route('/sign_language_detection')
def feature2():
    return sign_language_detection()






#@author AMIT ROY
@app.route('/virtual_keyboard')
def feature3():
    return virtual_keyboard()

def virtual_keyboard():
    #return keyboard.main()
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





def sign_language_detection():
    return "Not Yet Merged"


if __name__ == '__main__':
    app.secret_key = 'TishuPaper'
    app.run()