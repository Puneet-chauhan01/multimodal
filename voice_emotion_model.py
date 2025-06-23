
# import os
# import librosa
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# import joblib  # For saving scaler and encoder

# # Define the path to your dataset
# data_path = r"D:\mmd\multimodal-emotion-recognition\data\Audio_Speech_Actors_01-24"

# def extract_features(audio_path):
#     try:
#         # Load audio with fixed sample rate and duration
#         y, sr = librosa.load(audio_path, sr=22050, duration=3)
        
#         # Extract MFCC features with padding if needed
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        
#         # Pad/Cut to ensure consistent shape
#         if mfcc.shape[1] < 130:
#             mfcc = np.pad(mfcc, ((0, 0), (0, 130 - mfcc.shape[1])), mode='constant')
#         else:
#             mfcc = mfcc[:, :130]
            
#         return np.mean(mfcc, axis=1)
#     except Exception as e:
#         print(f"Error processing {audio_path}: {e}")
#         return None

# def load_ravdess_data(data_path):
#     features = []
#     labels = []
    
#     # RAVDESS emotion mapping (simplified to 5 classes)
#     emotion_map = {
#         '03': 'happy',
#         '04': 'sad',
#         '05': 'angry',
#         '06': 'fear',
#         '08': 'surprise',
#         # Combine neutral and calm into neutral
#         '01': 'neutral',
#         '02': 'neutral',
#         # Exclude disgust (limited samples)
#         '07': None  
#     }
    
#     for actor_folder in os.listdir(data_path):
#         actor_path = os.path.join(data_path, actor_folder)
        
#         if os.path.isdir(actor_path):
#             for audio_file in os.listdir(actor_path):
#                 if audio_file.endswith('.wav'):
#                     parts = audio_file.split('-')
#                     if len(parts) >= 3:
#                         emotion_code = parts[2]
#                         mapped_emotion = emotion_map.get(emotion_code)
                        
#                         if mapped_emotion is not None:
#                             audio_path = os.path.join(actor_path, audio_file)
#                             features_extracted = extract_features(audio_path)
                            
#                             if features_extracted is not None:
#                                 features.append(features_extracted)
#                                 labels.append(mapped_emotion)
    
#     return np.array(features), np.array(labels)

# # Load and preprocess data
# X, y = load_ravdess_data(data_path)

# # Filter out None values (if any)
# valid_indices = [i for i, label in enumerate(y) if label is not None]
# X = X[valid_indices]
# y = y[valid_indices]

# # Encode labels
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# # Split data with stratification
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_encoded, 
#     test_size=0.2, 
#     random_state=42,
#     stratify=y_encoded
# )

# # Standardize features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Save preprocessing objects
# joblib.dump(scaler, 'voice_scaler.pkl')
# joblib.dump(label_encoder, 'voice_label_encoder.pkl')

# from sklearn.metrics import classification_report


# model = tf.keras.models.load_model('voice_emotion_model_W.h5')

# # Load your test features and labels (assuming you saved them or re-extract)
# # For example, if you saved X_test and y_test as numpy files, load them:
# # X_test = np.load('X_test.npy')
# # y_test = np.load('y_test.npy')

# # If you don't have saved test data, you must run the same preprocessing pipeline again to get X_test and y_test
# # Assuming you have them already:

# # Scale your test features (if not already scaled)
# # X_test_scaled = scaler.transform(X_test)

# # Predict probabilities and get predicted classes



# # Build improved model
# model = tf.keras.Sequential([
#     tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
#     tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer='l2'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.5),
    
#     tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer='l2'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.4),
    
#     tf.keras.layers.Dense(256, activation='relu', kernel_regularizer='l2'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.3),
    
#     tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
# ])

# # Compile with adjusted learning rate
# from tensorflow.keras.optimizers import AdamW

# optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)

# # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# model.compile(
#     optimizer=optimizer,
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# # Add callbacks
# early_stop = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=15,
#     restore_best_weights=True
# )

# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.2,
#     patience=5,
#     min_lr=1e-8
# )

# # Train model
# history = model.fit(
#     X_train, y_train,
#     epochs=1000,
#     batch_size=64,
#     validation_data=(X_test, y_test),
#     callbacks=[early_stop, reduce_lr],
#     verbose=1
# )

# y_pred_probs = model.predict(X_test)
# y_pred = np.argmax(y_pred_probs, axis=1)

# # Decode encoded labels back to original class names
# y_true_labels = label_encoder.inverse_transform(y_test)
# y_pred_labels = label_encoder.inverse_transform(y_pred)

# # Print classification report
# print(classification_report(y_true_labels, y_pred_labels))
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"\nFinal Test Accuracy: {test_accuracy:.2%}")

# # Save model
# model.save('voice_emotion_model_W.h5')
# print("Model saved successfully")





import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from imblearn.over_sampling import SMOTE

# Define the path to your dataset
data_path = r"D:\mmd\multimodal-emotion-recognition\data\Audio_Speech_Actors_01-24"

def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=3)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        mfcc = np.pad(mfcc, ((0, 0), (0, max(0, 130 - mfcc.shape[1]))), mode='constant')[:, :130]
        return np.mean(mfcc, axis=1)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def load_ravdess_data(data_path):
    features, labels = [], []
    emotion_map = {
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fear',
        '08': 'surprise',
        # Combine neutral and calm into neutral
        '01': 'neutral',
        '02': 'neutral',
        # Exclude disgust (limited samples)
        '07': None  
    }
    for actor_folder in os.listdir(data_path):
        actor_path = os.path.join(data_path, actor_folder)
        if os.path.isdir(actor_path):
            for audio_file in os.listdir(actor_path):
                if audio_file.endswith('.wav'):
                    emotion_code = audio_file.split('-')[2]
                    label = emotion_map.get(emotion_code)
                    if label:
                        features_extracted = extract_features(os.path.join(actor_path, audio_file))
                        if features_extracted is not None:
                            features.append(features_extracted)
                            labels.append(label)
    return np.array(features), np.array(labels)

# Load data
X, y = load_ravdess_data(data_path)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE before train-test split
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Save scaler and label encoder
joblib.dump(scaler, 'voice_scaler.pkl')
joblib.dump(label_encoder, 'voice_label_encoder.pkl')

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2048, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile
from tensorflow.keras.optimizers import AdamW
optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-8)

# Train
history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Evaluate
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Decode labels
y_true_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

# Report
print("\nClassification Report:\n")
print(classification_report(y_true_labels, y_pred_labels))

# Accuracy
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy: {test_accuracy:.2%}")

# Save model
model.save('voice_emotion_model_W1.h5')
print("Model saved successfully.")
