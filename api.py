from flask import Flask, render_template, request, url_for, flash
from flask_mail import Mail, Message
import os
import set_envariables
import alberto.botCore

app = Flask(__name__)

set_envariables.setVariables()

SECRET_KEY = os.environ.get("SECRET_KEY")
MAIL_USERNAME = os.environ.get("MAIL_USERNAME")
WEBMAIL_USERNAME = os.environ.get("WEBMAIL_USERNAME")
WEBMAIL_PASSWORD = os.environ.get("WEBMAIL_PASSWORD")

mail_settings = {
    "MAIL_SERVER": 'smtp.gmail.com',
    "MAIL_PORT": 465,
    "MAIL_USE_TLS": False,
    "MAIL_USE_SSL": True,
    "MAIL_USERNAME": WEBMAIL_USERNAME,
    "MAIL_PASSWORD": WEBMAIL_PASSWORD
}

app.config.update(mail_settings)
app.secret_key = SECRET_KEY

mail = Mail(app)

@app.route('/', methods = ['POST', 'GET'])
def home():
    if request.method == 'POST':
        result = request.form

        bodymsg = "Name:" + str(request.form['name']) + "\nEmail:" + str(request.form['email']) + '\nBody:' + str(request.form['message'])

        with app.app_context():
            msg = Message(subject="Webfolio Automatic Message",
                          sender=app.config.get("MAIL_USERNAME"),
                          recipients=[MAIL_USERNAME], 
                          body=bodymsg)
            mail.send(msg)

        flash("Message sent!")

        return render_template("index.html", result=result)
    return render_template("index.html")

@app.route('/LuisCV')
def showCV():
    return render_template("curriculum.html")

@app.route('/Alberto')
def renderBot():
    return render_template("chatbot.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')


    return str(alberto.botCore.response(userText))

if __name__ == '__main__':
    app.run(debug=True)
