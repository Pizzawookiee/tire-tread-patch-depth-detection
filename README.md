# tire-tread-patch-depth-detection
Very fast YOLO-based tire tread patch depth prediction.

Basically, YOLO segmentation masks out the tire, the tire is split into n patches (default is 4), each patch is independently re-classified by YOLO, and for each patch a manual heuristic takes the prediction and makes a very rough estimate of the tire depth.
<img width="640" height="640" alt="image" src="https://github.com/user-attachments/assets/d2582a9c-836a-4f20-a29a-f7315d657683" />

<img width="640" height="640" alt="image" src="https://github.com/user-attachments/assets/18281c0d-48f5-40a9-bcb7-6c0fdd9a593d" />


Trained on this public domain dataset on Roboflow: https://universe.roboflow.com/mark-aft7n/tire-tread/dataset/5

Find checkpoint here: https://drive.google.com/file/d/1pCzzURLajh_I1ulydtH-12gQBr9HAGcF/view?usp=drive_link
