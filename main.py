import os
import threading

from flask import Flask
from flask import Response, render_template

from drawer.Drawer import *
from modules.tf_object_detector.object_detector import object_detector as tf_object_detector
from modules.hardhat_classifier.Xception_hardhat_classifier import XceptionHardhatClassifier

PATH_TO_PB = 'modules/hardhat_classifier/Xception_hardhat_with_hats.pb'
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
video = 'video/' + os.listdir('video')[0]


class VideoRunner():
    def __init__(self):

        self.obj_det = tf_object_detector()

        self.hardhat_classifier = XceptionHardhatClassifier()

        self.input_video = video

        self.cap = cv2.VideoCapture(self.input_video)

        _, self.frame = self.cap.read()

    def process(self):

        while self.cap.isOpened():
            _, self.frame = self.cap.read()
            if _:
                draw_script, ms, fps = self.obj_det.process(self.frame)
                drawer = Drawer()
                self.frame = drawer.process(self.frame, draw_script)
                cv2.putText(self.frame, "INFERENCE TIME/FPS: {} ms/{} fps".format(ms, fps), (2, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.FILLED)

                (flag, encodedImage) = cv2.imencode(".jpg", self.frame)
                # ensure the frame was successfully encoded
                if not flag:
                    continue
                # yield the output frame in the byte format
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encodedImage) + b'\r\n')
            else:
                self.cap = cv2.VideoCapture(self.input_video)

        self.cap.release()
        # cv2.destroyAllWindows()
        self.obj_det.close()


# render HTML templates
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(VideoRunner.process(self=vr),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    vr = VideoRunner()
    vr.process()
    print('Done')
    # start the flask app
    app.run(host='0.0.0.0', port=8000, debug=True,
            threaded=True, use_reloader=False)
