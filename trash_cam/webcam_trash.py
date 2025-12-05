import cv2
import numpy as np
import tensorflow as tf
import json

# ====== CONFIG ======
MODEL_PATH = 'best_model.h5'
CLASS_INDEX_PATH = 'class_indices.json'  # make sure the file is really this name
IMG_SIZE = (128, 128)  # must match what you used in flow_from_directory
CONFIDENCE_THRESHOLD = 0.5
# ====================

print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("[INFO] Loading class indices...")
with open(CLASS_INDEX_PATH, 'r') as f:
    index_to_class = json.load(f)

# Keys in JSON might be strings, so normalize to int
index_to_class = {int(k): v for k, v in index_to_class.items()}

print("[INFO] Classes:", index_to_class)

def preprocess_frame(frame):
    """
    Takes a BGR frame (from OpenCV), resizes, normalizes it,
    and returns a batch of shape (1, h, w, 3).
    """
    # Resize to model input size
    img = cv2.resize(frame, IMG_SIZE)

    # Convert BGR (OpenCV) to RGB (TensorFlow)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Scale to [0, 1]
    img = img.astype('float32') / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def predict_class(frame):
    """
    Run model prediction on a frame and return (label, probability)
    """
    batch = preprocess_frame(frame)
    preds = model.predict(batch, verbose=0)[0]  # 1D array of probs

    class_id = int(np.argmax(preds))
    prob = float(preds[class_id])

    label = index_to_class.get(class_id, f"class_{class_id}")
    return label, prob

def main():
    print("[INFO] Starting webcam...")
    cap = cv2.VideoCapture(0)  # 0 = default camera

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # Make a copy to display
        display_frame = frame.copy()

        # Get prediction
        label, prob = predict_class(frame)

        # Only show label if confidence > threshold
        if prob >= CONFIDENCE_THRESHOLD:
            text = f"{label} ({prob*100:.1f}%)"
        else:
            text = f"Unsure ({prob*100:.1f}%)"

        # Put text on the frame
        cv2.putText(display_frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow("Trash Classifier - press 'q' to quit", display_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam closed.")

if __name__ == "__main__":
    main()
