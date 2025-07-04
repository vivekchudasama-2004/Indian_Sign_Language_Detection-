import cv2
import numpy as np
import pandas as pd
import os
import mediapipe as mp
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import time
from collections import deque, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import tempfile
import shutil
from urllib.parse import urlparse
import requests
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class CISLRDataProcessor :
    def __init__(self, cache_dir=None) :
        self.cache_dir = cache_dir or tempfile.mkdtemp()
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.dataset = None

    def load_cislr_dataset(self) :
        try :
            self.dataset = load_dataset("Exploration-Lab/CISLR")
            return self.dataset
        except Exception as e :
            print(f"Error loading CISLR dataset: {e}")
            return None

    def save_video_from_bytes(self, video_bytes, filename) :
        try :
            filepath = os.path.join(self.cache_dir, filename)
            with open(filepath, 'wb') as f :
                f.write(video_bytes)
            return filepath
        except Exception as e :
            print(f"Error saving video {filename}: {e}")
            return None

    def extract_frames_from_video(self, video_path, max_frames=30) :
        try :
            if not os.path.exists(video_path) :
                return None

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened() :
                return None

            frames = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0 :
                cap.release()
                return None

            skip = max(1, total_frames // max_frames)

            while len(frames) < max_frames and cap.isOpened() :
                ret, frame = cap.read()
                if not ret :
                    break

                if frame_count % skip == 0 :
                    frames.append(frame)

                frame_count += 1

            cap.release()

            if len(frames) == 0 :
                return None

            return frames

        except Exception as e :
            return None

    def extract_landmarks_from_frame(self, frame) :
        try :
            if frame is None :
                return None

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(frame_rgb)
            pose_results = self.pose.process(frame_rgb)

            landmarks = []

            hand_landmarks_flat = []
            if hand_results.multi_hand_landmarks :
                for hand_landmarks in hand_results.multi_hand_landmarks :
                    for landmark in hand_landmarks.landmark :
                        hand_landmarks_flat.extend([landmark.x, landmark.y, landmark.z])

            if len(hand_landmarks_flat) < 126 :
                hand_landmarks_flat.extend([0.0] * (126 - len(hand_landmarks_flat)))
            elif len(hand_landmarks_flat) > 126 :
                hand_landmarks_flat = hand_landmarks_flat[:126]

            landmarks.extend(hand_landmarks_flat)

            if pose_results.pose_landmarks :
                for i in range(11) :
                    landmark = pose_results.pose_landmarks.landmark[i]
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            else :
                landmarks.extend([0.0] * 33)

            if len(landmarks) != 159 :
                if len(landmarks) < 159 :
                    landmarks.extend([0.0] * (159 - len(landmarks)))
                else :
                    landmarks = landmarks[:159]

            return np.array(landmarks, dtype=np.float32)

        except Exception as e :
            return None

    def process_video_sequence(self, video_data, max_frames=30) :
        try :
            video_path = None

            if hasattr(video_data, 'path') :
                video_path = video_data.path
            elif hasattr(video_data, 'read') :
                video_bytes = video_data.read()
                temp_path = os.path.join(self.cache_dir, f"temp_video_{time.time():.0f}.mp4")
                video_path = self.save_video_from_bytes(video_bytes, os.path.basename(temp_path))
            elif isinstance(video_data, dict) :
                if 'path' in video_data :
                    video_path = video_data['path']
                elif 'bytes' in video_data :
                    temp_path = os.path.join(self.cache_dir, f"temp_video_{time.time():.0f}.mp4")
                    video_path = self.save_video_from_bytes(video_data['bytes'], os.path.basename(temp_path))
            elif isinstance(video_data, bytes) :
                temp_path = os.path.join(self.cache_dir, f"temp_video_{time.time():.0f}.mp4")
                video_path = self.save_video_from_bytes(video_data, os.path.basename(temp_path))
            else :
                video_path = str(video_data)

            if not video_path :
                return None

            frames = self.extract_frames_from_video(video_path, max_frames)
            if not frames :
                return None

            sequence_features = []
            for frame in frames :
                landmarks = self.extract_landmarks_from_frame(frame)
                if landmarks is not None :
                    sequence_features.append(landmarks)

            if len(sequence_features) == 0 :
                return None

            while len(sequence_features) < max_frames :
                sequence_features.append(np.zeros(159, dtype=np.float32))

            sequence_features = sequence_features[:max_frames]

            if video_path and 'temp_video' in video_path :
                try :
                    os.remove(video_path)
                except :
                    pass

            return np.array(sequence_features, dtype=np.float32)

        except Exception as e :
            return None

    def prepare_cislr_dataset(self, split='train', max_samples=None, max_frames=30) :
        if self.dataset is None :
            return None, None, None, None

        if split not in self.dataset :
            return None, None, None, None

        data = self.dataset[split]
        if max_samples :
            data = data.select(range(min(max_samples, len(data))))

        X = []
        y = []

        successful_count = 0
        failed_count = 0

        for i, sample in enumerate(tqdm(data, desc="Processing samples")) :
            try :
                label = None
                possible_label_keys = ['label', 'word', 'sign', 'class', 'category']
                for key in possible_label_keys :
                    if key in sample :
                        label = sample[key]
                        break

                if label is None :
                    for key in sample.keys() :
                        if any(term in key.lower() for term in ['label', 'word', 'sign', 'class']) :
                            label = sample[key]
                            break

                if label is None :
                    failed_count += 1
                    continue

                video_data = None
                possible_video_keys = ['video', 'mp4', 'mov', 'avi', 'clip', 'file']
                for key in possible_video_keys :
                    if key in sample :
                        video_data = sample[key]
                        break

                if video_data is None :
                    for key in sample.keys() :
                        if any(term in key.lower() for term in ['video', 'mp4', 'mov', 'clip', 'file']) :
                            video_data = sample[key]
                            break

                if video_data is None :
                    failed_count += 1
                    continue

                sequence = self.process_video_sequence(video_data, max_frames)
                if sequence is not None :
                    X.append(sequence)
                    y.append(label)
                    successful_count += 1
                else :
                    failed_count += 1

            except Exception as e :
                failed_count += 1
                continue

        if len(X) == 0 :
            return None, None, None, None

        X = np.array(X)
        y = np.array(y)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        return X, y_encoded, label_encoder, y


class ISLModel :
    def __init__(self, input_shape, num_classes) :
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self) :
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=self.input_shape),
            BatchNormalization(),
            Dropout(0.3),

            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),

            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),

            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),

            Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50) :
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint('best_cislr_model.h5', save_best_only=True)
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def save_model(self, filepath) :
        self.model.save(filepath)

    def load_model(self, filepath) :
        self.model = load_model(filepath)

    def plot_training_history(self, history) :
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('cislr_training_history.png')
        plt.show()


class RealTimeISLRecognizer :
    def __init__(self, model_path, label_encoder_path, sequence_length=30) :
        self.model = load_model(model_path)
        with open(label_encoder_path, 'rb') as f :
            self.label_encoder = pickle.load(f)

        self.sequence_length = sequence_length
        self.sequence_buffer = deque(maxlen=sequence_length)

        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.prediction_buffer = deque(maxlen=5)
        self.last_prediction = ""
        self.prediction_confidence = 0.0

    def extract_landmarks_realtime(self, frame) :
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)

        landmarks = []

        hand_landmarks_flat = []
        if hand_results.multi_hand_landmarks :
            for hand_landmarks in hand_results.multi_hand_landmarks :
                for landmark in hand_landmarks.landmark :
                    hand_landmarks_flat.extend([landmark.x, landmark.y, landmark.z])

                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        if len(hand_landmarks_flat) < 126 :
            hand_landmarks_flat.extend([0.0] * (126 - len(hand_landmarks_flat)))
        elif len(hand_landmarks_flat) > 126 :
            hand_landmarks_flat = hand_landmarks_flat[:126]

        landmarks.extend(hand_landmarks_flat)

        if pose_results.pose_landmarks :
            for i in range(11) :
                landmark = pose_results.pose_landmarks.landmark[i]
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        else :
            landmarks.extend([0.0] * 33)

        if len(landmarks) != 159 :
            if len(landmarks) < 159 :
                landmarks.extend([0.0] * (159 - len(landmarks)))
            else :
                landmarks = landmarks[:159]

        return np.array(landmarks, dtype=np.float32), frame

    def predict_sign(self) :
        if len(self.sequence_buffer) < self.sequence_length :
            return None, 0.0

        sequence = np.array(list(self.sequence_buffer))
        sequence = sequence.reshape(1, self.sequence_length, -1)

        prediction = self.model.predict(sequence, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])

        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]

        return predicted_label, confidence

    def smooth_predictions(self, prediction, confidence, threshold=0.7) :
        if confidence > threshold :
            self.prediction_buffer.append(prediction)

        if len(self.prediction_buffer) >= 3 :
            most_common = Counter(self.prediction_buffer).most_common(1)[0][0]
            return most_common

        return self.last_prediction

    def run_realtime_recognition(self) :
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while True :
            ret, frame = cap.read()
            if not ret :
                break

            frame = cv2.flip(frame, 1)

            landmarks, annotated_frame = self.extract_landmarks_realtime(frame)

            self.sequence_buffer.append(landmarks)

            if len(self.sequence_buffer) == self.sequence_length :
                prediction, confidence = self.predict_sign()

                if prediction :
                    smoothed_prediction = self.smooth_predictions(prediction, confidence)

                    if smoothed_prediction != self.last_prediction and confidence > 0.7 :
                        self.last_prediction = smoothed_prediction
                        self.prediction_confidence = confidence

            cv2.putText(annotated_frame, f"Sign: {self.last_prediction}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Confidence: {self.prediction_confidence:.2f}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Buffer: {len(self.sequence_buffer)}/{self.sequence_length}",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.imshow('CISLR Real-time Recognition', annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') :
                break
            elif key == ord('r') :
                self.sequence_buffer.clear()

        cap.release()
        cv2.destroyAllWindows()


def filter_classes_by_sample_count(X, y, label_encoder, min_samples=2) :
    unique_classes, counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[counts >= min_samples]

    if len(valid_classes) == 0 :
        return None, None, None

    mask = np.isin(y, valid_classes)
    X_filtered = X[mask]
    y_filtered = y[mask]

    original_labels = label_encoder.inverse_transform(y_filtered)
    new_label_encoder = LabelEncoder()
    y_filtered_encoded = new_label_encoder.fit_transform(original_labels)

    return X_filtered, y_filtered_encoded, new_label_encoder


def evaluate_model(model_path, label_encoder_path, X_test, y_test) :
    model = load_model(model_path)
    with open(label_encoder_path, 'rb') as f :
        label_encoder = pickle.load(f)

    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('cislr_confusion_matrix.png')
    plt.show()


def create_synthetic_data(num_samples=100, num_classes=10, sequence_length=30, feature_dim=159) :
    X = np.random.rand(num_samples, sequence_length, feature_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples)

    label_encoder = LabelEncoder()
    class_names = [f"sign_{i}" for i in range(num_classes)]
    label_encoder.fit(class_names)

    y_labels = [class_names[i] for i in y]

    return X, y, label_encoder, y_labels


def main() :
    print("CISLR Real-time Recognition System")
    print("=" * 50)

    print("Choose an option:")
    print("1. Load and explore CISLR dataset")
    print("2. Train new model")
    print("3. Run real-time recognition")
    print("4. Evaluate existing model")
    print("5. Train with synthetic data")
    choice = input("Enter choice (1-5): ")

    if choice == "1" :
        processor = CISLRDataProcessor()
        dataset = processor.load_cislr_dataset()

        if dataset :
            print("\nDataset loaded successfully!")
            print("Available splits:", list(dataset.keys()))

            if 'train' in dataset :
                sample = dataset['train'][0]
                print("\nSample data structure:")
                for key, value in sample.items() :
                    print(f"  {key}: {type(value)}")
                    if isinstance(value, (str, int, float)) :
                        print(f"    Value: {value}")
            elif 'test' in dataset :
                sample = dataset['test'][0]
                print("\nSample data structure:")
                for key, value in sample.items() :
                    print(f"  {key}: {type(value)}")
                    if isinstance(value, (str, int, float)) :
                        print(f"    Value: {value}")

    elif choice == "2" :
        processor = CISLRDataProcessor()
        dataset = processor.load_cislr_dataset()

        if dataset is None :
            print("Failed to load dataset. Using synthetic data.")
            choice = "5"

        if choice == "2" :
            available_splits = list(dataset.keys())
            print(f"Available splits: {available_splits}")

            if len(available_splits) == 1 :
                split_to_use = available_splits[0]
            else :
                print("Choose split for training:")
                for i, split in enumerate(available_splits) :
                    print(f"{i + 1}. {split}")
                choice_split = input("Enter choice: ")
                try :
                    split_to_use = available_splits[int(choice_split) - 1]
                except (ValueError, IndexError) :
                    split_to_use = available_splits[0]

            max_samples = input("Enter max samples (default 100): ")
            max_samples = int(max_samples) if max_samples else 100

            max_frames = input("Enter max frames per video (default 30): ")
            max_frames = int(max_frames) if max_frames else 30

            X, y, label_encoder, original_labels = processor.prepare_cislr_dataset(
                split=split_to_use, max_samples=max_samples, max_frames=max_frames
            )

            if X is None or len(X) == 0 :
                print("Failed to process real data. Using synthetic data.")
                choice = "5"

    if choice == "5" :
        num_samples = int(input("Enter number of samples (default 200): ") or "200")
        num_classes = int(input("Enter number of classes (default 10): ") or "10")

        X, y, label_encoder, original_labels = create_synthetic_data(
            num_samples=num_samples, num_classes=num_classes
        )

    if choice in ["2", "5"] :
        if X is not None and len(X) > 0 :
            print(f"Dataset shape: {X.shape}")
            print(f"Number of classes: {len(np.unique(y))}")

            X_filtered, y_filtered, new_label_encoder = filter_classes_by_sample_count(
                X, y, label_encoder, min_samples=2
            )

            if X_filtered is None :
                print("Not enough samples for training.")
                return

            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )

            print(f"Training set: {X_train.shape}")
            print(f"Validation set: {X_val.shape}")
            print(f"Test set: {X_test.shape}")

            model = ISLModel(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                num_classes=len(np.unique(y_filtered))
            )

            model.build_model()
            print("Model built successfully!")

            epochs = int(input("Enter number of epochs (default 50): ") or "50")

            history = model.train(X_train, y_train, X_val, y_val, epochs=epochs)

            model.save_model('cislr_model.h5')

            with open('cislr_label_encoder.pkl', 'wb') as f :
                pickle.dump(new_label_encoder, f)

            print("Model and label encoder saved!")

            model.plot_training_history(history)

            evaluate_model('cislr_model.h5', 'cislr_label_encoder.pkl', X_test, y_test)

    elif choice == "3" :
        model_path = input("Enter model path (default: cislr_model.h5): ") or "cislr_model.h5"
        encoder_path = input(
            "Enter label encoder path (default: cislr_label_encoder.pkl): ") or "cislr_label_encoder.pkl"

        if not os.path.exists(model_path) or not os.path.exists(encoder_path) :
            print("Model files not found. Please train a model first.")
            return

        recognizer = RealTimeISLRecognizer(model_path, encoder_path)
        recognizer.run_realtime_recognition()

    elif choice == "4" :
        model_path = input("Enter model path: ")
        encoder_path = input("Enter label encoder path: ")

        if not os.path.exists(model_path) or not os.path.exists(encoder_path) :
            print("Model files not found.")
            return

        processor = CISLRDataProcessor()
        dataset = processor.load_cislr_dataset()

        if dataset is None :
            print("Using synthetic test data.")
            X_test, y_test, _, _ = create_synthetic_data(num_samples=50, num_classes=10)
        else :
            X_test, y_test, _, _ = processor.prepare_cislr_dataset(split='test', max_samples=100)

        if X_test is not None :
            evaluate_model(model_path, encoder_path, X_test, y_test)
        else :
            print("Could not load test data.")


if __name__ == "__main__" :
    main()