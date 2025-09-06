from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2 as cv
from PIL import Image
cam = cv.VideoCapture(0)
mtcnn=MTCNN()
resnet = InceptionResnetV1(pretrained='casia-webface').eval()
img=Image.open("image.png")
while True:
    try:
        ret, frame = cam.read()
        boxes,_=mtcnn.detect(frame)
        if not ret:
            raise Exception("Can't receive frame (stream end?). Exiting ...")
        if boxes is not None:
            for box in boxes:
                frame=cv.rectangle(frame,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),color=(255,0,0))
            faces1, _ = mtcnn.detect(img)
            if faces1 is not None and boxes is not None and frame is not None:
                aligned1 = mtcnn(img)
                aligned2 = mtcnn(frame)
                if aligned1 is not None and aligned2 is not None:
                    embeddings1 = resnet(aligned1.unsqueeze(0)).detach()
                    embeddings2 = resnet(aligned2.unsqueeze(0)).detach()
                    distance = (embeddings1 - embeddings2).norm().item()
                    if distance < 1.0:
                        cv.putText(frame,"Eh o Gustavo",org=(50,50),fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.8, color = (0, 255, 0), thickness = 2)
                        print("Eh o Gustavo")
                    else:
                        cv.putText(frame,"Nao eh o Gustavo",org=(50,50),fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.8, color = (0, 0, 255), thickness = 2)
                        print("Nao eh o Gustavo")
        else:
            cv.putText(frame,"Ninguem na imagem",org=(50,50),fontFace = cv.FONT_HERSHEY_SIMPLEX, fontScale = 0.8, color = (0, 0, 255), thickness = 2)
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
    except Exception as e:
        print(e)
        break
cam.release()
cv.destroyAllWindows()