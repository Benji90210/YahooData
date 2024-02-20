from flask import Flask, request, render_template
import matplotlib.pyplot as plt
from lfunction import yahooData


import os
import numpy as np

app = Flask(__name__)
proba = [0.1, 0.25, 0.5, 0.75, 0.9]


def create_figure(period, kernel, field, udlg):
    A = yahooData(udlg, period, kernel, field)
    A.calibKernelDensity()
    ys = A.getUdlgtgtPrice(proba)
    return ys


@app.route('/', methods=['GET', 'POST'])
def form():
    period = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    kernel = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
    field = ['Open', 'High', 'Low', 'Close', 'Volume']
    return render_template('form.html', period=period, kernel=kernel, field=field)


@app.route('/data', methods=['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':

        form_data = request.form
        period = request.form['period']
        kernel = request.form['kernel']
        field = request.form['field']
        udlg = request.form['Ticker']

        ys = create_figure(period, kernel, field, udlg)
        print(ys)
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        ax.plot(proba, ys)
        fig.savefig('./static/img.jpg')
        #plt.close(fig)
        #return render_template('data.html', form_data=form_data, url='./static/images/img.png')
        return render_template('data.html', form_data=form_data)


app.run(host='localhost', port=5000)


if __name__ == "__main__":
    '''
    # Generate sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    # Plot the data
    plt.plot(x, y)
    # Set the title and labels
    plt.title("Sine Wave")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    # Save the plot as a PNG, JPEG and PDF files
    plt.savefig('./static/img.jpg')
    #plt.close(fig)
    print(os.path.exists('./static/img.jpg'))
    '''
    app.run(debug=True)



