from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def result():

    weight = float(request.form['weight'])
    Category = float(request.form['Category'])
    Subcategory = float(request.form['Sub category'])
    PRODUCTBRANDS = float(request.form['PRODUCT BRANDS'])
    MRP = float(request.form['MRP'])
    ItemName = float(request.form['Item Name'])
    ItemDisc  = float(request.form['Item Disc %'])
    Itemdiscamount = float(request.form['Item disc amount'])
    Profitpercentage  = float(request.form['Profit %'])
    ItemNetAmount = float(request.form['Item Net Amount'])
    profit  = float(request.form['profit'])
    Netcost	  = float(request.form['Net cost'])
    quantity  = float(request.form['quantity'])

    X = np.array([[ weight, Category,Subcategory,PRODUCTBRANDS,MRP,ItemName,ItemDisc,Itemdiscamount,Profitpercentage, ItemNetAmount,profit,Netcost,quantity]])


    model_path = r'C:\Users\Tms\Dropbox\PC\Desktop\sales ml\model\sales ml_XG_boost_model.sav'

    model = joblib.load(model_path)

    Y_pred = model.predict(X)
    predicted_sales = float(Y_pred)
    return render_template('result.html', result=predicted_sales)
    
    #return jsonify({'Sales Prediction of the product is = ': float(Y_pred)})


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
