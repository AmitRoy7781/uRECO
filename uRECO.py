from flask import Flask,render_template,flash,redirect,url_for,session,logging
from flask_mysqldb import MySQL
from wtforms import Form,StringField,TextAreaField,PasswordField,validators
#from passlib.hash import sha256_cryptf

app = Flask(__name__)


@app.route('/')
def main():
    return render_template("index.html")

@app.route('/sign_up')
def sign_up():
    return render_template("sign_up.html")

@app.route('/sign_in')
def sign_in():
    return render_template("sign_in.html")


if __name__ == '__main__':
    app.run()
