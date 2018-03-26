from flask import Flask,render_template

app = Flask(__name__)


@app.route('/')
def main():
    return render_template("index.html")

@app.route('/sign_up')
def sign_up():
    return 'Not Yet Implemented'

@app.route('/sign_in')
def sign_in():
    return 'Not Yet Implemented'


if __name__ == '__main__':
    app.run()
