import os
import cv2
from wurlitzer import pipes

def generate_video(path, score, threshold):

    print ('[*] Concatenating KeyFrame ...')

    capture = cv2.VideoCapture(path)
    
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    dataset = path.split('/')[0]

    filename = path.split('/')[-1].split('.')[0] + '.mkv'

    writer = cv2.VideoWriter(os.path.join('summary', dataset, filename), fourcc, 24.0, size)

    frames = None

    nframe = score.shape[1]

    k = 0
    
    with pipes() as (out, err):
    
        for f in range(nframe - 1):

            ret, frame = capture.read()

            if ret is None or frame is None:

                break
            
            if score[0][f] > threshold:

                k += 1

                writer.write(frame)

    capture.release()
    
    writer.release()

    print ('[*] Ratio KeyFrame / Frames => {} %'.format(int(float(k) / nframe * 100)))

    print ('[*] Summary is save to => {}'.format(os.path.join('summary', dataset, filename)))
    
    cv2.destroyAllWindows()

