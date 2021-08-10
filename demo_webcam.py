import cv2
import numpy as np
import tensorflow as tf

from common import normalize_image

if __name__ == '__main__':
    # model path
    checkpoint_path = "./weights/pretrained/MobileFaceNet_TF.ckpt"
    meta_graph_path = "./weights/pretrained/MobileFaceNet_TF.ckpt.meta"

    # make session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    graph = tf.get_default_graph()
    session = tf.Session(graph=graph, config=config)

    # restore graph
    saver = tf.train.import_meta_graph(meta_graph_path)
    saver.restore(session, checkpoint_path)

    # get i/o
    inputs = graph.get_tensor_by_name("input:0")
    outputs = graph.get_tensor_by_name("embeddings:0")

    # prepare camera
    video_capture = cv2.VideoCapture(0)

    # prepare registered feature
    registered_feafures = np.zeros((1, 192))

    while True:
        _, frame = video_capture.read()
        input_image = cv2.resize(frame, (112, 112))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        normalized_image = normalize_image(input_image)

        input_batch = normalized_image.reshape((1, 112, 112, 3))
        feed_dict = {inputs: input_batch}
        features = session.run(outputs, feed_dict=feed_dict)

        diff = np.subtract(features, registered_feafures)
        distance = np.sum(np.square(diff), 1)

        threshold = 0.5
        text1 = "authed" if distance < threshold else "not authed"
        text2 = "ditstance:" + str(distance)
        output_frame = frame
        output_frame = cv2.putText(output_frame, text1, (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (180, 180, 80), 2)
        output_frame = cv2.putText(output_frame, text2, (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, (180, 180, 180), 2)
        cv2.imshow('frame', output_frame)

        key_pressed = cv2.waitKey(1)
        if key_pressed & 0xFF == ord('r'):
            registered_feafures = features
            print("registered!!")
