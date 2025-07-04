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


class ISLDataProcessor :
    def __init__(self, csv_path, frames_base_path) :
        self.csv_path = csv_path
        self.frames_base_path = frames_base_path
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

    def load_data(self) :
        """Load and process the CSV/Excel data"""
        try :
            file_extension = os.path.splitext(self.csv_path)[1].lower()

            if file_extension == '.csv' :
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                df = None
                for encoding in encodings :
                    try :
                        df = pd.read_csv(self.csv_path, encoding=encoding)
                        break
                    except UnicodeDecodeError :
                        continue
                if df is None :
                    raise ValueError("Could not load CSV with any encoding")
            elif file_extension in ['.xlsx', '.xls'] :
                df = pd.read_excel(self.csv_path)
            else :
                raise ValueError(f"Unsupported file format: {file_extension}")

            return df
        except Exception as e :
            print(f"Error loading file: {e}")
            return None

    def extract_landmarks(self, image_path) :
        """Extract hand and pose landmarks from image"""
        try :
            image = cv2.imread(image_path)
            if image is None :
                return None

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(image_rgb)
            pose_results = self.pose.process(image_rgb)

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
            return None

    def process_sentence_sequence(self, sentence_paths, max_frames=30) :
        """Process a sequence of frames for a sentence"""
        sequence_features = []
        expected_feature_size = 159

        for path in sentence_paths[:max_frames] :
            if isinstance(path, str) :
                if 'ISL_CSLRT_Corpus\\' in path :
                    clean_path = path.replace('ISL_CSLRT_Corpus\\', '').replace('\\', os.sep)
                else :
                    clean_path = path.replace('\\', os.sep)

                full_path = os.path.join(self.frames_base_path, clean_path)

                if not os.path.exists(full_path) :
                    full_path = os.path.join(self.frames_base_path, path)

                if not os.path.exists(full_path) :
                    filename = os.path.basename(path)
                    for root, dirs, files in os.walk(self.frames_base_path) :
                        if filename in files :
                            full_path = os.path.join(root, filename)
                            break

                landmarks = self.extract_landmarks(full_path)
                if landmarks is not None and len(landmarks) == expected_feature_size :
                    sequence_features.append(landmarks)

        if len(sequence_features) == 0 :
            return None

        sequence_features = [np.array(seq, dtype=np.float32) for seq in sequence_features]

        # Pad or truncate to max_frames
        while len(sequence_features) < max_frames :
            sequence_features.append(np.zeros(expected_feature_size, dtype=np.float32))

        try :
            result = np.array(sequence_features[:max_frames], dtype=np.float32)
            return result
        except Exception as e :
            return None

    def prepare_dataset(self, max_frames=30) :
        """Prepare the complete dataset"""
        df = self.load_data()
        if df is None :
            return None, None, None, None

        # Find correct column names
        column_names = [col.lower() for col in df.columns]
        sentence_col = frames_col = None

        for col, col_lower in zip(df.columns, column_names) :
            if col_lower in ['word', 'sentence', 'sentences', 'sign glosses', 'label'] :
                sentence_col = col
                break

        for col, col_lower in zip(df.columns, column_names) :
            if 'path' in col_lower or 'location' in col_lower or 'file' in col_lower :
                frames_col = col
                break

        if sentence_col is None or frames_col is None :
            print("Error: Could not find required columns")
            return None, None, None, None

        # Group by sentence/word
        grouped = df.groupby(sentence_col)
        X = []
        y = []

        for label, group in grouped :
            paths = group[frames_col].tolist()

            if paths and isinstance(paths[0], str) and paths[0].lower().endswith('.mp4') :
                continue

            sequence = self.process_sentence_sequence(paths, max_frames)
            if sequence is not None and len(sequence) > 0 :
                X.append(sequence)
                y.append(label)

        if len(X) == 0 :
            print("No sequences were successfully processed!")
            return None, None, None, None

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

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
            ModelCheckpoint('best_isl_model.h5', save_best_only=True)
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
        plt.savefig('training_history.png')
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
            cv2.imshow('ISL Real-time Recognition', annotated_frame)

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


def inspect_file(file_path) :
    """Helper function to inspect CSV/Excel file structure"""
    try :
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.csv' :
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            for encoding in encodings :
                try :
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError :
                    continue
            if df is None :
                print("Could not load CSV with any encoding")
                return None
        elif file_extension in ['.xlsx', '.xls'] :
            df = pd.read_excel(file_path)
        else :
            print(f"Unsupported file format: {file_extension}")
            return None

        print(f"File: {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("\nFirst 5 rows:")
        print(df.head())

        # Check for path columns
        for col in df.columns :
            if 'path' in col.lower() or 'location' in col.lower() or 'file' in col.lower() :
                print(f"\n{col} samples:")
                sample_paths = df[col].dropna().head(3).tolist()
                for path in sample_paths :
                    print(f"  {path}")

        return df
    except Exception as e :
        print(f"Error inspecting file: {e}")
        return None


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
    plt.savefig('confusion_matrix.png')
    plt.show()


def main() :
    print("ISL Real-time Recognition System")
    print("=" * 50)

    # Configuration - Update these paths according to your setup
    csv_options = {
        "1" : {
            "path" : r"C:\Users\Vivek\PycharmProjects\ISL\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\corpus_csv_files\ISL_CSLRT_Corpus_word_details.xlsx",
            "description" : "Word-level frames (Word + Frames path)"
        },
        "2" : {
            "path" : r"C:\Users\Vivek\PycharmProjects\ISL\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\corpus_csv_files\ISL_CSLRT_Corpus_frame_details.xlsx",
            "description" : "Sentence-level frames (Sentence + Frames path)"
        },
        "3" : {
            "path" : r"C:\Users\Vivek\PycharmProjects\ISL\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\corpus_csv_files\ISL_CSLRT_Corpus_details.xlsx",
            "description" : "Sentence-level videos (Sentences + File location)"
        },
        "4" : {
            "path" : r"C:\Users\Vivek\PycharmProjects\ISL\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\corpus_csv_files\ISL_Corpus_sign_glosses.csv",
            "description" : "Sign glosses only (Sentence + SIGN GLOSSES)"
        }
    }

    frames_base_path = r"C:\Users\Vivek\PycharmProjects\ISL\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus"

    print("Available files:")
    for key, info in csv_options.items() :
        print(f"{key}. {info['description']}")

    print("\nChoose an option:")
    print("0. Inspect file")
    print("1. Train new model")
    print("2. Run real-time recognition")
    print("3. Evaluate existing model")
    choice = input("Enter choice (0, 1, 2, or 3): ")

    if choice == "0" :
        # Inspect file
        print("\nWhich file do you want to inspect?")
        for key, info in csv_options.items() :
            print(f"{key}. {info['description']}")

        file_choice = input("Enter file choice (1-4): ")
        if file_choice in csv_options :
            inspect_file(csv_options[file_choice]["path"])
        else :
            print("Invalid choice")

    elif choice == "1" :
        # Training mode
        print("\nWhich file do you want to use for training?")
        for key, info in csv_options.items() :
            if key != "4" :  # Skip glosses-only file
                print(f"{key}. {info['description']}")

        file_choice = input("Enter file choice (1-3): ")
        if file_choice in csv_options and file_choice != "4" :
            file_path = csv_options[file_choice]["path"]
            print(f"Using file: {file_path}")
            print("Loading and processing data...")

            processor = ISLDataProcessor(file_path, frames_base_path)
            X, y, label_encoder, original_labels = processor.prepare_dataset(max_frames=30)

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
                    # Split data with stratification
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
                    history = model.train(X_train, y_train, X_val, y_val, epochs=50)

                    # Save model and label encoder
                    model.save_model('final_isl_model.h5')
                    with open('label_encoder.pkl', 'wb') as f :
                        pickle.dump(new_label_encoder, f)

                    print("Model training completed and saved!")

                    # Plot training history
                    model.plot_training_history(history)

                    # Evaluate on validation set
                    print("\nValidation Results:")
                    evaluate_model('final_isl_model.h5', 'label_encoder.pkl', X_val, y_val)

                except Exception as e :
                    print(f"Error during training: {e}")
            else :
                print("No data could be processed for training")

    elif choice == "2" :
        # Real-time recognition mode
        model_path = input("Enter model path (default: 'final_isl_model.h5'): ") or 'final_isl_model.h5'
        label_encoder_path = input("Enter label encoder path (default: 'label_encoder.pkl'): ") or 'label_encoder.pkl'

        if not os.path.exists(model_path) or not os.path.exists(label_encoder_path) :
            print("Model or label encoder file not found!")
            return

        try :
            recognizer = RealTimeISLRecognizer(model_path, label_encoder_path)
            recognizer.run_realtime_recognition()
        except Exception as e :
            print(f"Error in real-time recognition: {e}")

    elif choice == "3" :
        # Evaluation mode
        model_path = input("Enter model path (default: 'final_isl_model.h5'): ") or 'final_isl_model.h5'
        label_encoder_path = input("Enter label encoder path (default: 'label_encoder.pkl'): ") or 'label_encoder.pkl'

        if not os.path.exists(model_path) or not os.path.exists(label_encoder_path) :
            print("Model or label encoder file not found!")
            return

        print("Load test data first...")
        # You would need to load test data here
        # For now, we'll use the same data loading process
        file_choice = input("Enter file choice for test data (1-3): ")
        if file_choice in csv_options and file_choice != "4" :
            file_path = csv_options[file_choice]["path"]
            processor = ISLDataProcessor(file_path, frames_base_path)
            X, y, label_encoder, original_labels = processor.prepare_dataset(max_frames=30)

            if X is not None and len(X) > 0 :
                X_filtered, y_filtered, new_label_encoder = filter_classes_by_sample_count(
                    X, y, label_encoder, min_samples=2
                )

                if X_filtered is not None :
                    # Split for testing
                    _, X_test, _, y_test = train_test_split(
                        X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
                    )

                    evaluate_model(model_path, label_encoder_path, X_test, y_test)
                else :
                    print("Could not prepare test data")
            else :
                print("No test data could be processed")

    else :
        print("Invalid choice")


if __name__ == "__main__" :
    main()