from flask import Flask, request, render_template
from lfunction import yahooData

app = Flask(__name__)


@app.route('/')
def form():
    return render_template('form.html')


@app.route('/data', methods=['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form
        return render_template('data.html', form_data=form_data)


app.run(host='localhost', port=5000)


if __name__ == "__main__":
    app.run(debug=True)
    A = yahooData('META')
    A.calibKernelDensity()
    f = A.getUdlgtgtPrice(0.50)
    print(f)



