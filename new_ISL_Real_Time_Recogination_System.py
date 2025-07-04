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
        """Load CISLR dataset from Hugging Face"""
        try :
            print("Loading CISLR dataset from Hugging Face...")
            # Load the dataset
            self.dataset = load_dataset("Exploration-Lab/CISLR")
            print(f"Dataset loaded successfully!")

            # Print dataset info
            print("\nDataset splits:")
            for split in self.dataset.keys() :
                print(f"  {split}: {len(self.dataset[split])} samples")

            # Show sample data structure
            if 'train' in self.dataset :
                sample = self.dataset['train'][0]
                print(f"\nSample data keys: {list(sample.keys())}")

            return self.dataset

        except Exception as e :
            print(f"Error loading CISLR dataset: {e}")
            print("Make sure you have logged in with: huggingface-cli login")
            return None

    def download_video(self, video_url, filename) :
        """Download video from URL"""
        try :
            response = requests.get(video_url, stream=True)
            if response.status_code == 200 :
                filepath = os.path.join(self.cache_dir, filename)
                with open(filepath, 'wb') as f :
                    for chunk in response.iter_content(chunk_size=8192) :
                        f.write(chunk)
                return filepath
            else :
                return None
        except Exception as e :
            print(f"Error downloading video {filename}: {e}")
            return None

    def extract_frames_from_video(self, video_path, max_frames=30) :
        """Extract frames from video file"""
        try :
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened() :
                return None

            frames = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate frame skip to get max_frames evenly distributed
            skip = max(1, total_frames // max_frames)

            while len(frames) < max_frames and cap.isOpened() :
                ret, frame = cap.read()
                if not ret :
                    break

                if frame_count % skip == 0 :
                    frames.append(frame)

                frame_count += 1

            cap.release()
            return frames

        except Exception as e :
            print(f"Error extracting frames: {e}")
            return None

    def extract_landmarks_from_frame(self, frame) :
        """Extract hand and pose landmarks from a single frame"""
        try :
            if frame is None :
                return None

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(frame_rgb)
            pose_results = self.pose.process(frame_rgb)

            landmarks = []

            # Extract hand landmarks (126 features: 2 hands * 21 landmarks * 3 coordinates)
            hand_landmarks_flat = []
            if hand_results.multi_hand_landmarks :
                for hand_landmarks in hand_results.multi_hand_landmarks :
                    for landmark in hand_landmarks.landmark :
                        hand_landmarks_flat.extend([landmark.x, landmark.y, landmark.z])

            # Pad or truncate to exactly 126 features
            if len(hand_landmarks_flat) < 126 :
                hand_landmarks_flat.extend([0.0] * (126 - len(hand_landmarks_flat)))
            elif len(hand_landmarks_flat) > 126 :
                hand_landmarks_flat = hand_landmarks_flat[:126]

            landmarks.extend(hand_landmarks_flat)

            # Extract pose landmarks (33 features: 11 landmarks * 3 coordinates)
            if pose_results.pose_landmarks :
                for i in range(11) :  # Upper body only
                    landmark = pose_results.pose_landmarks.landmark[i]
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            else :
                landmarks.extend([0.0] * 33)

            # Ensure exactly 159 features
            if len(landmarks) != 159 :
                if len(landmarks) < 159 :
                    landmarks.extend([0.0] * (159 - len(landmarks)))
                else :
                    landmarks = landmarks[:159]

            return np.array(landmarks, dtype=np.float32)

        except Exception as e :
            print(f"Error extracting landmarks: {e}")
            return None

    def process_video_sequence(self, video_data, max_frames=30) :
        """Process a video to extract landmark sequences"""
        try :
            # Handle different video data formats
            if hasattr(video_data, 'path') :
                # If it's a file path
                video_path = video_data.path
            elif isinstance(video_data, dict) and 'bytes' in video_data :
                # If it's video bytes, save to temporary file
                temp_path = os.path.join(self.cache_dir, f"temp_video_{time.time()}.mp4")
                with open(temp_path, 'wb') as f :
                    f.write(video_data['bytes'])
                video_path = temp_path
            else :
                # Assume it's a direct path or URL
                video_path = str(video_data)

            # Extract frames from video
            frames = self.extract_frames_from_video(video_path, max_frames)
            if not frames :
                return None

            # Extract landmarks from each frame
            sequence_features = []
            for frame in frames :
                landmarks = self.extract_landmarks_from_frame(frame)
                if landmarks is not None :
                    sequence_features.append(landmarks)

            if len(sequence_features) == 0 :
                return None

            # Pad or truncate to max_frames
            while len(sequence_features) < max_frames :
                sequence_features.append(np.zeros(159, dtype=np.float32))

            sequence_features = sequence_features[:max_frames]

            # Clean up temporary file if created
            if 'temp_video' in video_path :
                try :
                    os.remove(video_path)
                except :
                    pass

            return np.array(sequence_features, dtype=np.float32)

        except Exception as e :
            print(f"Error processing video sequence: {e}")
            return None

    def prepare_cislr_dataset(self, split='train', max_samples=None, max_frames=30) :
        """Prepare CISLR dataset for training"""
        if self.dataset is None :
            print("Dataset not loaded. Please run load_cislr_dataset() first.")
            return None, None, None, None

        if split not in self.dataset :
            print(f"Split '{split}' not found in dataset")
            return None, None, None, None

        data = self.dataset[split]
        if max_samples :
            data = data.select(range(min(max_samples, len(data))))

        X = []
        y = []

        print(f"Processing {len(data)} samples from {split} split...")

        for i, sample in enumerate(tqdm(data)) :
            try :
                # Get label (word/sign)
                if 'label' in sample :
                    label = sample['label']
                elif 'word' in sample :
                    label = sample['word']
                elif 'sign' in sample :
                    label = sample['sign']
                else :
                    # Try to find label in other possible fields
                    label = None
                    for key in sample.keys() :
                        if 'label' in key.lower() or 'word' in key.lower() or 'sign' in key.lower() :
                            label = sample[key]
                            break
                    if label is None :
                        continue

                # Get video data
                video_data = None
                if 'video' in sample :
                    video_data = sample['video']
                elif 'mp4' in sample :
                    video_data = sample['mp4']
                else :
                    # Try to find video in other possible fields
                    for key in sample.keys() :
                        if 'video' in key.lower() or 'mp4' in key.lower() :
                            video_data = sample[key]
                            break

                if video_data is None :
                    continue

                # Process video sequence
                sequence = self.process_video_sequence(video_data, max_frames)
                if sequence is not None :
                    X.append(sequence)
                    y.append(label)

                # Progress update
                if (i + 1) % 50 == 0 :
                    print(f"Processed {i + 1}/{len(data)} samples, {len(X)} successful")

            except Exception as e :
                print(f"Error processing sample {i}: {e}")
                continue

        if len(X) == 0 :
            print("No samples were successfully processed!")
            return None, None, None, None

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        print(f"Dataset prepared successfully:")
        print(f"  Sequences: {X.shape}")
        print(f"  Labels: {len(np.unique(y_encoded))} unique classes")
        print(f"  Class distribution: {Counter(y).most_common(10)}")

        return X, y_encoded, label_encoder, y


class ISLModel :
    def __init__(self, input_shape, num_classes) :
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self) :
        """Build LSTM-based model for sequence classification"""
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
        """Train the model"""
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
        """Save the trained model"""
        self.model.save(filepath)

    def load_model(self, filepath) :
        """Load a trained model"""
        self.model = load_model(filepath)

    def plot_training_history(self, history) :
        """Plot training history"""
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

        # MediaPipe setup
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

        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=5)
        self.last_prediction = ""
        self.prediction_confidence = 0.0

    def extract_landmarks_realtime(self, frame) :
        """Extract landmarks from a single frame in real-time"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(frame_rgb)
        pose_results = self.pose.process(frame_rgb)

        landmarks = []

        # Extract hand landmarks
        hand_landmarks_flat = []
        if hand_results.multi_hand_landmarks :
            for hand_landmarks in hand_results.multi_hand_landmarks :
                for landmark in hand_landmarks.landmark :
                    hand_landmarks_flat.extend([landmark.x, landmark.y, landmark.z])

                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Pad or truncate to exactly 126 features
        if len(hand_landmarks_flat) < 126 :
            hand_landmarks_flat.extend([0.0] * (126 - len(hand_landmarks_flat)))
        elif len(hand_landmarks_flat) > 126 :
            hand_landmarks_flat = hand_landmarks_flat[:126]

        landmarks.extend(hand_landmarks_flat)

        # Extract pose landmarks
        if pose_results.pose_landmarks :
            for i in range(11) :  # Upper body only
                landmark = pose_results.pose_landmarks.landmark[i]
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        else :
            landmarks.extend([0.0] * 33)

        # Ensure exactly 159 features
        if len(landmarks) != 159 :
            if len(landmarks) < 159 :
                landmarks.extend([0.0] * (159 - len(landmarks)))
            else :
                landmarks = landmarks[:159]

        return np.array(landmarks, dtype=np.float32), frame

    def predict_sign(self) :
        """Predict sign from current sequence buffer"""
        if len(self.sequence_buffer) < self.sequence_length :
            return None, 0.0

        # Prepare sequence for prediction
        sequence = np.array(list(self.sequence_buffer))
        sequence = sequence.reshape(1, self.sequence_length, -1)

        # Make prediction
        prediction = self.model.predict(sequence, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])

        # Decode prediction
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]

        return predicted_label, confidence

    def smooth_predictions(self, prediction, confidence, threshold=0.7) :
        """Smooth predictions to reduce noise"""
        if confidence > threshold :
            self.prediction_buffer.append(prediction)

        if len(self.prediction_buffer) >= 3 :
            # Return most common prediction
            most_common = Counter(self.prediction_buffer).most_common(1)[0][0]
            return most_common

        return self.last_prediction

    def run_realtime_recognition(self) :
        """Run real-time ISL recognition"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Starting real-time ISL recognition...")
        print("Press 'q' to quit, 'r' to reset buffer")

        while True :
            ret, frame = cap.read()
            if not ret :
                break

            # Flip frame horizontally for selfie-view
            frame = cv2.flip(frame, 1)

            # Extract landmarks
            landmarks, annotated_frame = self.extract_landmarks_realtime(frame)

            # Add to sequence buffer
            self.sequence_buffer.append(landmarks)

            # Make prediction if buffer is full
            if len(self.sequence_buffer) == self.sequence_length :
                prediction, confidence = self.predict_sign()

                if prediction :
                    smoothed_prediction = self.smooth_predictions(prediction, confidence)

                    if smoothed_prediction != self.last_prediction and confidence > 0.7 :
                        self.last_prediction = smoothed_prediction
                        self.prediction_confidence = confidence

            # Display results on frame
            cv2.putText(annotated_frame, f"Sign: {self.last_prediction}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Confidence: {self.prediction_confidence:.2f}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Buffer: {len(self.sequence_buffer)}/{self.sequence_length}",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Show frame
            cv2.imshow('CISLR Real-time Recognition', annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') :
                break
            elif key == ord('r') :
                self.sequence_buffer.clear()

        cap.release()
        cv2.destroyAllWindows()


def filter_classes_by_sample_count(X, y, label_encoder, min_samples=2) :
    """Filter out classes with fewer than min_samples"""
    unique_classes, counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[counts >= min_samples]

    if len(valid_classes) == 0 :
        print("No classes have enough samples!")
        return None, None, None

    # Filter data
    mask = np.isin(y, valid_classes)
    X_filtered = X[mask]
    y_filtered = y[mask]

    # Create new label encoder for filtered classes
    original_labels = label_encoder.inverse_transform(y_filtered)
    new_label_encoder = LabelEncoder()
    y_filtered_encoded = new_label_encoder.fit_transform(original_labels)

    return X_filtered, y_filtered_encoded, new_label_encoder


def evaluate_model(model_path, label_encoder_path, X_test, y_test) :
    """Evaluate the trained model"""
    # Load model and label encoder
    model = load_model(model_path)
    with open(label_encoder_path, 'rb') as f :
        label_encoder = pickle.load(f)

    # Make predictions
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Confusion matrix
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


def main() :
    print("CISLR Real-time Recognition System")
    print("=" * 50)

    print("Choose an option:")
    print("1. Load and explore CISLR dataset")
    print("2. Train new model")
    print("3. Run real-time recognition")
    print("4. Evaluate existing model")
    choice = input("Enter choice (1, 2, 3, or 4): ")

    if choice == "1" :
        # Load and explore dataset
        processor = CISLRDataProcessor()
        dataset = processor.load_cislr_dataset()

        if dataset :
            print("\nDataset loaded successfully!")
            print("Available splits:", list(dataset.keys()))

            # Show sample data
            if 'train' in dataset :
                sample = dataset['train'][0]
                print("\nSample data structure:")
                for key, value in sample.items() :
                    print(f"  {key}: {type(value)}")
                    if isinstance(value, (str, int, float)) :
                        print(f"    Value: {value}")


    # Fix for the main() function - replace the training section (choice == "2")

    elif choice == "2" :

        # Training mode

        processor = CISLRDataProcessor()

        # Load dataset

        dataset = processor.load_cislr_dataset()

        if dataset is None :
            print("Failed to load dataset")

            return

        # Check available splits and let user choose

        available_splits = list(dataset.keys())

        print(f"Available splits: {available_splits}")

        if len(available_splits) == 1 :

            # Only one split available, use it for training

            split_to_use = available_splits[0]

            print(f"Using '{split_to_use}' split for training")

        else :

            # Multiple splits available, let user choose

            print("Choose split for training:")

            for i, split in enumerate(available_splits) :
                print(f"{i + 1}. {split}")

            choice_split = input("Enter choice: ")

            try :

                split_to_use = available_splits[int(choice_split) - 1]

            except (ValueError, IndexError) :

                split_to_use = available_splits[0]

                print(f"Invalid choice, using '{split_to_use}'")

        # Ask for parameters

        max_samples = input("Enter max samples to process (default: 1000): ")

        max_samples = int(max_samples) if max_samples else 1000

        max_frames = input("Enter max frames per video (default: 30): ")

        max_frames = int(max_frames) if max_frames else 30

        print(f"Processing dataset with max_samples={max_samples}, max_frames={max_frames}")

        # Prepare dataset using the available split

        X, y, label_encoder, original_labels = processor.prepare_cislr_dataset(

            split=split_to_use, max_samples=max_samples, max_frames=max_frames

        )

        if X is not None and len(X) > 0 :

            print(f"Initial dataset shape: {X.shape}")

            print(f"Initial number of classes: {len(np.unique(y))}")

            # Filter classes with insufficient samples

            X_filtered, y_filtered, new_label_encoder = filter_classes_by_sample_count(

                X, y, label_encoder, min_samples=2

            )

            if X_filtered is None :
                print("Cannot proceed with training - no classes have enough samples")

                return

            # Check if we have enough classes after filtering

            if len(np.unique(y_filtered)) < 2 :
                print("Error: Need at least 2 classes for training")

                return

            try :

                # Split the data into train and validation sets

                # Since we only have one split, we'll create our own train/test split

                X_train, X_val, y_train, y_val = train_test_split(

                    X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered

                )

                print(f"Training data shape: {X_train.shape}")

                print(f"Validation data shape: {X_val.shape}")

                print(f"Number of classes: {len(np.unique(y_train))}")

                # Build and train model

                model = ISLModel(input_shape=(X_train.shape[1], X_train.shape[2]),

                                 num_classes=len(np.unique(y_train)))

                model.build_model()

                print("Starting training...")

                epochs = input("Enter number of epochs (default: 50): ")

                epochs = int(epochs) if epochs else 50

                history = model.train(X_train, y_train, X_val, y_val, epochs=epochs)

                # Save model and label encoder

                model.save_model('final_cislr_model.h5')

                with open('cislr_label_encoder.pkl', 'wb') as f :

                    pickle.dump(new_label_encoder, f)

                print("Model training completed and saved!")

                # Plot training history

                model.plot_training_history(history)

                # Evaluate on validation set

                print("\nValidation Results:")

                evaluate_model('final_cislr_model.h5', 'cislr_label_encoder.pkl', X_val, y_val)


            except Exception as e :

                print(f"Error during training: {e}")

        else :

            print("No data could be processed for training")

    elif choice == "3" :
        # Real-time recognition mode
        model_path = input("Enter model path (default: 'final_cislr_model.h5'): ") or 'final_cislr_model.h5'
        label_encoder_path = input(
            "Enter label encoder path (default: 'cislr_label_encoder.pkl'): ") or 'cislr_label_encoder.pkl'

        if not os.path.exists(model_path) or not os.path.exists(label_encoder_path) :
            print("Model or label encoder file not found!")
            return

        try :
            recognizer = RealTimeISLRecognizer(model_path, label_encoder_path)
            recognizer.run_realtime_recognition()
        except Exception as e :
            print(f"Error in real-time recognition: {e}")

    elif choice == "4" :
        # Evaluation mode
        model_path = input("Enter model path (default: 'final_cislr_model.h5'): ") or 'final_cislr_model.h5'
        label_encoder_path = input(
            "Enter label encoder path (default: 'cislr_label_encoder.pkl'): ") or 'cislr_label_encoder.pkl'

        if not os.path.exists(model_path) or not os.path.exists(label_encoder_path) :
            print("Model or label encoder file not found!")
            return

        print("Load test data...")
        processor = CISLRDataProcessor()

        # Load dataset
        dataset = processor.load_cislr_dataset()
        if dataset is None :
            print("Failed to load dataset")
            return

        # Use test split if available, otherwise use a portion of train
        test_split = 'test' if 'test' in dataset else 'train'
        max_samples = input("Enter max test samples (default: 200): ")
        max_samples = int(max_samples) if max_samples else 200

        X, y, label_encoder, original_labels = processor.prepare_cislr_dataset(
            split=test_split, max_samples=max_samples, max_frames=30
        )

        if X is not None and len(X) > 0 :
            X_filtered, y_filtered, new_label_encoder = filter_classes_by_sample_count(
                X, y, label_encoder, min_samples=2
            )

            if X_filtered is not None :
                evaluate_model(model_path, label_encoder_path, X_filtered, y_filtered)
            else :
                print("Could not prepare test data")
        else :
            print("No test data could be processed")

    else :
        print("Invalid choice")


if __name__ == "__main__" :
    main()