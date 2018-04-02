from flask import Flask, render_template, request, json, flash, redirect, url_for, session, logging
from flask_mysqldb import MySQL
import MySQLdb


mysql = MySQL()
app = Flask(__name__)

# MySQL configurations
conn = MySQLdb.connect(host="localhost",user="root",password="tauhid 123",db="uRECO")

mysql.init_app(app)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/show_sign_up')
def showSignUp():
    return render_template('sign_up.html')


@app.route('/sign_up', methods=['POST'])
def sign_up():


        name = request.form["name"]
        email = request.form["email"]
        password = request.form["pass"]



        cursor = conn.cursor()
        cursor.execute("INSERT INTO user (user_username,user_password,user_email)VALUES(%s,%s,%s)", (name, password, email))
        conn.commit()
        return render_template("home.html")


@app.route('/show_sign_in')
def showSignIn():
    return render_template('sign_in.html')


@app.route('/sign_in')
def sign_in():
    return render_template("sign_in.html")


if __name__ == '__main__':
    app.run()
