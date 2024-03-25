import torch
import insightface
import cv2

import base64
import numpy as np
import io

img = 'img/1.jpg'
model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0, det_thresh=0.45)
face_img = cv2.imread(img)


# res = model.get(face_img)

def encode_np(array):
    bio = io.BytesIO()
    np.save(bio, array)
    return base64.standard_b64encode(bio.getvalue())


def round(value, factor):
    return int(value.astype(float) * factor) / factor

rgb_small_frame = face_img[:, :, ::-1]
faces = model.get(rgb_small_frame)

result = {"numberOfFaces": len(faces),
            "faces": [
                {
                    "score": round(face.det_score, 1000),
                    "boundingBox": {
                        "x1": round(face.bbox[0], 100),
                        "y1": round(face.bbox[1], 100),
                        "x2": round(face.bbox[2], 100),
                        "y2": round(face.bbox[3], 100),
                    },
                    "keyPoints": [
                        {"x": round(xy[0], 100), "y": round(xy[1], 100)}
                        for xy in face.kps
                    ],
                    "landmarks3d68": [
                        {
                            "x": round(xyz[0], 100),
                            "y": round(xyz[1], 100),
                            "z": round(xyz[2], 100),
                        }
                        for xyz in face.landmark_3d_68
                    ],
                    "landmarks2d106": [
                        {
                            "x": round(xy[0], 100),
                            "y": round(xy[1], 100),
                        }
                        for xy in face.landmark_2d_106
                    ],
                    "attributes": {"sex": face.sex, "age": face.age},
                    "embedding": encode_np(face.embedding),
                }
                for face in faces
            ]
        }

print(result)
# print('人脸数量：', len(res))
# print("#"*100)
# print('res keys: ', res[0].keys())  # 结果包括 ['bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'gender', 'age', 'embedding']
# print("#"*100)
# print('age: ', res[0]['age'])
# print("#"*100)
# print('gender: ', res[0]['gender'])
# print("#"*100)
# print('embedding: ', res[0]['embedding'])
# print("#"*100)
# print('landmark_2d_106: ', res[0]['landmark_2d_106'])
# print("#"*100)
# print('landmark_3d_68: ', res[0]['landmark_3d_68'])