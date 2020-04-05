import os
from flask import Flask, render_template, Response
from video_capture import video_capture
from gan_capture import gan_capture

app = Flask(__name__)

#ps: this is a simplified version of camera streaming,
#    more details and better camera is in: https://blog.miguelgrinberg.com/post/video-streaming-with-flask

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def cv2_frame_generator():
    """Frame generator function."""
    camera = video_capture()

    report_every = 20
    counter = 0
    while True:
        frame, fps = camera.get_frame()
        counter += 1
        if counter > report_every:
               print("fps:",fps) # around 25-30 fps
               counter = 0
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gan_frame_generator():
    path = "/media/ekmek/Vitek/Vitek2020/PythonProjects/GAN_handler/models/"
    pro_gan_path = path+"aerials/aerials512vectors1024px_snapshot-010200.pkl" # 5 fps
    pro_gan_path = path+"aerials/aerials128vectors256px_-snapshot-007440_GsOnly.pkl" # 17 fps demo

    gan_model_handler = gan_capture(pro_gan_path)

    report_every = 1
    counter = 0
    while True:
        frame, fps = gan_model_handler.get_frame()

        counter += 1
        if counter > report_every:
               print("fps:",fps) # around 5 fps
               counter = 0
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    
    #gen = cv2_frame_generator()
    gen = gan_frame_generator()
    return Response(gen, mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
