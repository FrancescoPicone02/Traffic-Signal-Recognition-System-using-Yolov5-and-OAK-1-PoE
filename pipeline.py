import depthai as dai
import cv2

 
# Crea pipeline
pipeline = dai.Pipeline()

# Nodo: fotocamera RGB
cam = pipeline.create(dai.node.ColorCamera)
#cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P) #%%%%%%%%%%%%%%%%%
cam.setPreviewSize(640, 640)
cam.setInterleaved(False)
cam.setFps(60)

cam_control = pipeline.create(dai.node.XLinkIn)  #%%%%%%%%%%%%%%%%%
cam_control.setStreamName("control")   #%%%%%%%%%%%%%%%%%
cam_control.out.link(cam.inputControl)    #%%%%%%%%%%%%%%%%%

# Nodo: rete YOLO
nn = pipeline.create(dai.node.YoloDetectionNetwork)
nn.setBlobPath(r"C:\Users\Lab_elettronica\dai_env\Lib\Francesco_Picone_ultimo\best1_openvino_2022.1_6shave.blob")
nn.setConfidenceThreshold(0.5)
nn.setNumClasses(15)
nn.setCoordinateSize(4)
nn.setIouThreshold(0.5)

# Set anchors (copiati dal file JSON)
nn.setAnchors([
    10.0, 13.0, 16.0, 30.0, 33.0, 23.0,
    30.0, 61.0, 62.0, 45.0, 59.0, 119.0,
    116.0, 90.0, 156.0, 198.0, 373.0, 326.0
])

# Set anchor masks (copiati dal file JSON)
nn.setAnchorMasks({
    "side80": [0, 1, 2],
    "side40": [3, 4, 5],
    "side20": [6, 7, 8]
})

nn.setNumInferenceThreads(3)


# Linking camera -> rete
cam.preview.link(nn.input)

# Output video
xout_cam = pipeline.create(dai.node.XLinkOut)
xout_cam.setStreamName("cam")
cam.preview.link(xout_cam.input)

# Output detections
xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
nn.out.link(xout_nn.input)

with dai.Device(pipeline) as device:
    q_cam = device.getOutputQueue(name="cam", maxSize=4, blocking=False)
    q_nn  = device.getOutputQueue(name="nn",  maxSize=4, blocking=False)

    #control_queue = device.getInputQueue("control")
    #control = dai.CameraControl()
    #control.setManualFocus(30)
    #control_queue.send(control)

    labels = [  # <-- questa riga è ora correttamente indentata
        "Green Light", "Red Light", "Speed Limit 10", "Speed Limit 100", "Speed Limit 110",
        "Speed Limit 120", "Speed Limit 20", "Speed Limit 30", "Speed Limit 40",
        "Speed Limit 50", "Speed Limit 60", "Speed Limit 70", "Speed Limit 80",
        "Speed Limit 90", "Stop"
    ]

    while True:
        in_cam = q_cam.get()
        in_nn = q_nn.get()

        frame = in_cam.getCvFrame()
        detections = in_nn.detections

        for det in detections:
            x1 = int(det.xmin * frame.shape[1])
            y1 = int(det.ymin * frame.shape[0])
            x2 = int(det.xmax * frame.shape[1])
            y2 = int(det.ymax * frame.shape[0])
            cls = int(det.label)
            conf = det.confidence

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{labels[cls]} {conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow("OAK-D Pro PoE", frame)
        if cv2.waitKey(1) == 27:
            break
