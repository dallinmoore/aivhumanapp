from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("app.html")

@app.route("/upload", methods=["POST"])
def upload():
    # This is a placeholder for your upload functionality
    return "File upload functionality will go here"

if __name__ == "__main__":
    app.run(debug=True)



