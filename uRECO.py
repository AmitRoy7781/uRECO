from flask import Flask, render_template, request, json, flash, redirect, url_for, session, logging
from flask_mysqldb import MySQL

mysql = MySQL()
app = Flask(__name__)

# MySQL configurations
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'TishuPaper'
app.config['MYSQL_DATABASE_DB'] = 'uRECO'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/show_sign_up')
def showSignUp():
    return render_template('sign_up.html')


@app.route('/sign_up', methods=['POST', 'GET'])
def sign_up():
    try:
        name = request.form['name']
        email = request.form['email']
        password = request.form['pass']

        if name and email and password:

            conn = mysql.connect()
            cursor = conn.cursor()
            # _hashed_password = generate_password_hash(password)
            cursor.callproc('sp_createUser', (name, email, password))
            data = cursor.fetchall()

            if len(data) is 0:
                conn.commit()
                return json.dumps({'message': 'User created successfully !'})
            else:
                return json.dumps({'error': str(data[0])})
        else:
            return json.dumps({'html': '<span>Enter the required fields</span>'})

    except Exception as e:
        return json.dumps({'error': str(e)})

    finally:
        cursor.close()
        conn.close()
    render_template("sign_up.html")

@app.route('/sign_in')
def sign_in():
    return render_template("sign_in.html")


if __name__ == '__main__':
    app.run()
