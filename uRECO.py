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


@app.route('/facial_expression_recognition')
def feature1():
    return facial_expression_recognition()


@app.route('/sign_language_detection')
def feature2():
    return sign_language_detection()


@app.route('/virtual_keyboard')
def feature3():
    return virtual_keyboard()








#from VirtualKeyboard import virtual_keyboard as keyboard
def virtual_keyboard():
    #return keyboard.main()
    return "Not Yet Merged"

#from FacialExpressionClassifer import DetectEmotion as emodetect
def facial_expression_recognition():
    #return emodetect
    return "Not Yet Merged"


def sign_language_detection():
    return "Not Yet Merged"


if __name__ == '__main__':
    app.secret_key = 'TishuPaper'
    app.run()