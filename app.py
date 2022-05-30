from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder="template")

reg = pickle.load(open("model log reg.pkl", "rb"))


@app.route("/")
def hello_worl():
    print("Test1")
    return render_template("main.html")


@app.route("/about")
def hello_wor():
    print("Test2")
    return render_template("about.html")


@app.route("/choose")
def hello_wo():
    print("Test3")
    return render_template("choose.html")


@app.route("/remedy")
def hello_wr():
    print("Test4")
    return render_template("sol.html")


@app.route("/test")
def hello1():
    print("Test5")
    return render_template("test.html")


@app.route("/predict", methods=["POST"])
def home():
    print("Test6")
    data1 = float(request.form["a"])
    data2 = float(request.form["b"])
    data3 = float(request.form["c"])
    data4 = float(request.form["d"])
    d5 = float(request.form["e"])
    d6 = float(request.form["f"])
    d7 = float(request.form["g"])
    d8 = float(request.form["h"])
    d9 = float(request.form["i"])
    d10 = float(request.form["j"])
    d11 = float(request.form["k"])
    d12 = float(request.form["l"])
    d13 = float(request.form["m"])
    d14 = float(request.form["n"])
    d15 = float(request.form["o"])
    d16 = float(request.form["p"])
    d17 = float(request.form["q"])
    d18 = float(request.form["r"])
    d19 = float(request.form["s"])


    arr = np.array(
        [
            [
                data1,
                data2,
                data3,
                data4,
                d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16,
                d17, d18, d19
            ]
        ]
    )
    pred = reg.predict(arr)
    print(pred)
    return render_template("index.html", data=pred)


reg1 = pickle.load(open("model log_reg_prof.pkl", "rb"))


@app.route("/test")
def hello2():
    print("Test7")
    return render_template("test.html")

@app.route("/test2")
def hello():
    print("Test7a")
    return render_template("test2.html")

@app.route("/Result", methods=["POST"])
def hom():
    print("Test8")
    data1 = float(request.form["a"])
    data2 = float(request.form["b"])
    data3 = float(request.form["c"])
    data4 = float(request.form["d"])
    d5 = float(request.form["e"])
    d6 = float(request.form["f"])
    d7 = float(request.form["g"])
    d8 = float(request.form["h"])
    d9 = float(request.form["i"])
    d10 = float(request.form["j"])
    d11 = float(request.form["k"])
    d12 = float(request.form["l"])
    d13 = float(request.form["m"])
    d14 = float(request.form["n"])
    d15 = float(request.form["o"])



    arr = np.array(
        [
            [
                data1,
                data2,
                data3,
                data4,
                d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15
            ]
        ]
    )
    print(arr)
    pred1 = reg1.predict(arr)
    print(pred1)
    return render_template("index.html", data=pred1)

reg2 = pickle.load(open("model.pkl", "rb"))

@app.route("/test3")
def hello21():
    print("Test7")
    return render_template("test3.html")

@app.route("/Result1", methods=["POST"])
def hom1():
    print("Test8")
    data1 = float(request.form["a"])
    data2 = float(request.form["b"])
    data3 = float(request.form["c"])
    data4 = float(request.form["d"])
    d5 = float(request.form["e"])
    d6 = float(request.form["f"])
    d7 = float(request.form["g"])
    d8 = float(request.form["h"])
    d9 = float(request.form["i"])
    d10 = float(request.form["j"])
    d11 = float(request.form["k"])




    arr = np.array(
        [
            [
                data1,
                data2,
                data3,
                data4,
                d5, d6, d7, d8, d9, d10, d11
            ]
        ]
    )
    print(arr)
    pred2 = reg2.predict(arr)
    print(pred2)
    return render_template("index1.html", data=pred2)


if __name__ == "__main__":
    app.run(debug=True)