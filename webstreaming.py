'''
实现从网页打开视频并进行检测
去除了复杂的终端　运行，默认打开本地端口8000
python webstreaming.py
改为使用opencv打开摄像头
'''

from GazeTracking.FindEye import findeye
from flask import Response
from flask import Flask, request, redirect, url_for
from flask import render_template
from werkzeug.utils import secure_filename
import threading
import argparse
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import datetime

basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = '/home/hichens/Datasets/xieshi/'
SAVE_FOLDER = os.path.join(basedir,  'static/video/')
ALLOWED_EXTENSIONS = set(['mp4', 'flv'])

sns.set()
warnings.filterwarnings("ignore")



outputFrame = None
lock = threading.Lock()
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = datetime.timedelta(seconds=1) # file refresh time
app.config['MAX_CONTENT_LENGTH'] = 160 * 1024 * 1024 # maximum file  <= 160MB
flag = False # control the camera
filename = ''

# 统计变量
Xrelative = []
Yrelative = []
cosValue = []

# result standard data
inx  = 0
RiskDegree = [r'无斜视风险.',
              r'有斜视风险，建议去医院进一步检查!',
              r'斜视风险非常高，建议去医院进一步检查！',
]

Standard_max_Xrelative = 0.73
Standard_min_Xrelative = 0.67
Standard_avg_Xrelative = 0.7

Standard_max_Yrelative = 0.06
Standard_min_Yrelative = 0.016
Standard_avg_Yrelative = 0.03

Standard_S_x = 0.0155
Standard_S_y = 0.085
Standard_S_xy = 0.0189

Standard_max_cosValue = 1
Standard_min_cosValue = 0.5
Standard_avg_cosValue = 0.65


# "/home/hichens/Datasets/xieshi/lj.mp4"
# "http://admin:admin@192.168.43.1:8081/"   #此处@后的ipv4 地址需要修改为自己的地址
# 0
source = 0 # filename, phone camera, computer camera
video= source



@app.route("/index")
@app.route("/")
def index():
    # return the rendered template
    global flag
    flag = False
    return render_template("index.html")


@app.route("/main")
def main():
    # return the rendered template
    global flag
    flag = True
    return render_template("main.html")

def NormProcess(X):
    sigma = np.std(X)
    mean = np.mean(X)
    newX = []
    for x in X:
       if mean - 2 * sigma <= x <= mean + 2 * sigma:
           newX.append(x)
    return newX


@app.route("/result")
def result():
    global flag, Xrelative, Yrelative, cosValue
    flag = False
    basedir = os.path.abspath(os.path.dirname(__file__))
    path = basedir + "/static/images/result.png"
    fig = plt.gcf()
    fig.set_size_inches(11.5, 6.5) # output size

    try:
        Xrelative = NormProcess(Xrelative)
        Yrelative = NormProcess(Yrelative)
        cosValue = NormProcess(cosValue)

        '''figure 1 show the x relative position, incluce min, max, avg '''
        plt.subplot(221)
        plt.title('figure 1')
        N = len(Xrelative)
        XX = range(N)
        max_Xrelative, min_Xrelative, avg_Xrelative = max(Xrelative), min(Xrelative), np.mean(Xrelative)
        max_Yrelative, min_Yrelative, avg_Yrelative = max(Yrelative), min(Yrelative), np.mean(Yrelative)
        plt.plot(XX, Xrelative, label='X relative')
        plt.plot(XX, [avg_Xrelative] * N, '-', label='X relative average')
        plt.plot(XX, [Standard_avg_Xrelative]*N, '-', label="reference average")
        plt.plot(XX, [Standard_max_Xrelative] * N, '--')
        plt.plot(XX, [Standard_min_Xrelative] * N, '--')
        plt.legend()

        '''figure 2 show the y relative position, incluce min, max,   avg '''
        plt.subplot(222)
        plt.title('figure 2')
        N = len(Yrelative)
        XX = range(N)
        plt.plot(XX, Yrelative, label='Y relative')
        plt.plot(XX, [Standard_max_Yrelative] * N, '--')
        plt.plot(XX, [Standard_min_Yrelative] * N, '--')
        plt.plot(XX, [avg_Yrelative] * N, '-', label='Y relative average')
        plt.plot(XX, [Standard_avg_Yrelative]*N, '-', label="reference average")
        plt.legend()

        '''figure 3 show the S value'''
        plt.subplot(223)
        plt.title('figure 3')
        S_x = np.std(Xrelative)
        S_y = np.std(Yrelative)
        S_xy = np.sqrt(S_x * S_x + S_y * S_y)
        S = [S_x, S_y, S_xy]
        Standad_S = [Standard_S_x, Standard_S_y, Standard_S_xy]
        XX = np.arange(3)
        width = 0.4
        plt.bar(XX, S, width=width, label='S')
        plt.bar(XX + width, Standad_S, width=0.4, label='Standard')
        plt.legend()
        plt.xticks(range(3), ['S_x', 'S_y', 'S_xy'])

        '''figure 4 show the cosine relative value '''
        plt.subplot(224)
        plt.title('figure 4')
        min_cos = min(cosValue)
        max_cos = max(cosValue)
        avg_cos = np.mean(cosValue)
        N = len(cosValue)
        XX = range(N)
        plt.plot(XX, cosValue, '-o', label='cosine relative value')
        plt.plot(XX, [Standard_max_cosValue] * N, '--')
        plt.plot(XX, [Standard_min_cosValue] * N, '--')
        plt.plot(XX, [avg_cos] * N, '-', label='cosine average value')
        plt.plot(XX, [Standard_avg_cosValue]*N, '-', label="reference average")
        plt.legend()

        plt.show()
        fig.savefig(path, dpi=100)
        print("here")

        global inx
        inx = 0
        if max_Xrelative > Standard_max_Xrelative:
            inx += 1
        if min_Xrelative < Standard_min_Xrelative:
            inx += 1
        if avg_Xrelative  < Standard_avg_Xrelative - S_x  or  avg_Xrelative  > Standard_avg_Xrelative + S_x:
            inx += 3
        if max_Yrelative > Standard_max_Yrelative:
            inx += 1
        if min_Yrelative < Standard_min_Yrelative:
            inx += 1
        if avg_Yrelative  < Standard_avg_Yrelative - S_x  or  avg_Yrelative  > Standard_avg_Yrelative + S_y:
            inx += 3
        if avg_cos < Standard_min_cosValue:
            inx += 10

        inx //= 7
        # inx = min(inx, 2)

    except:
        return redirect(url_for('ProcessError'))
    else:
        print(inx)
        print(RiskDegree[inx])
        data = {
            '水平相对移动最大移动值': [round(max_Xrelative, 2), Standard_max_Xrelative],
            '水平相对移动最小移动值': [round(min_Xrelative, 2), Standard_min_Xrelative],
            '水平相对移动平均移动值': [round(avg_Xrelative, 2), Standard_avg_Xrelative],
            '竖直相对移动最大移动值': [round(max_Yrelative, 2), Standard_max_Yrelative],
            '竖直相对移动最小移动值': [round(min_Yrelative, 2), Standard_min_Yrelative],
            '竖直相对移动平均移动值': [round(avg_Yrelative, 2), Standard_avg_Yrelative],

            '水平相对移动方差值': [round(S_x, 4), Standard_S_x],
            '竖直相对移动方差值': [round(S_y, 4), Standard_S_y],
            '综合相对移动方差值': [round(S_xy, 4), Standard_S_xy],

            '最大余弦相似值': [round(max_cos, 4), Standard_max_cosValue],
            '最小余弦相似值': [round(min_cos, 4), Standard_min_cosValue],
            '平均余弦相似值': [round(avg_cos, 4), Standard_avg_cosValue]

        }
        Result = [RiskDegree[inx], inx]
        return render_template("result.html", data=data, Result=Result)


def detect():
    global outputFrame, lock, Xrelative, Yrelative
    cap = cv2.VideoCapture(video)
    Xrelative.clear()
    Yrelative.clear()
    cosValue.clear()
    while True:
        if flag == False:
            break
        ret, frame = cap.read()
        if ret:
            frame, xr, yr, cosXY = findeye(frame)
            if xr and yr and cosXY:
                Xrelative.append(xr)
                Yrelative.append(yr)
                cosValue.append(cosXY)

            with lock:
                scale_percent = 700  # percent of original size
                width = scale_percent
                height = scale_percent * frame.shape[0] // frame.shape[1]
                frame = cv2.resize(frame, (width, height))
                outputFrame = frame.copy()
        else:
            pass

@app.route("/PorcessError")
def ProcessError():
    return render_template("ProcessError.html")

def generate():

    global outputFrame, lock
    # time.sleep(1.0)

    t = threading.Thread(target=detect)
    t.daemon = True
    t.start()
    while True:

        with lock:
            if outputFrame is None:
                continue
            (temp, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not temp:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/videoprocess', methods=['GET', 'POST'])
def upload_file():
    global filename, video

    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            file.save(SAVE_FOLDER+filename)
            video = SAVE_FOLDER+filename
            return redirect(url_for('main'))
        else:
            return redirect(url_for('index')+"#2")

        #return redirect(url_for('uploadpage'))


@app.route('/cameraprocess', methods=['GET', 'POST'])
def camerapage():
    global video
    if request.method == 'POST':
        video = source
        return redirect(url_for('main'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False,
            threaded=True, use_reloader=False)

