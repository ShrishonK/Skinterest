import os
import pandas as pd
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow as tf

def organize_dataset(base_path, images_path, metadata_path, output_path, val_split=0.2):
    """
    Organizes HAM10000 dataset into train and validation sets based on diagnosis categories.
    """
    # Define the mapping of dx codes to full names
    diagnosis_mapping = {
        'akiec': 'Actinic keratoses and intraepithelial carcinoma',
        'bcc': 'basal cell carcinoma',
        'bkl': 'benign keratosis-like lesions',
        'df': 'dermatofibroma',
        'mel': 'melanoma',
        'nv': 'melanocytic nevi',
        'vasc': 'vascular lesions'
    }
    
    # Read metadata
    df = pd.read_csv(metadata_path)
    
    # Create necessary directories
    for split in ['train', 'val']:
        for dx_code in diagnosis_mapping.keys():
            folder_path = os.path.join(output_path, split, dx_code)
            os.makedirs(folder_path, exist_ok=True)
            print(f"Created directory: {folder_path}")
    
    # Manual stratified split
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()
    
    # Split each class maintaining proportions
    for diagnosis in df['dx'].unique():
        # Get all samples for this diagnosis
        diagnosis_df = df[df['dx'] == diagnosis].copy()
        
        # Shuffle the dataframe
        diagnosis_df = diagnosis_df.sample(frac=1, random_state=42)
        
        # Calculate split point
        n_val = int(len(diagnosis_df) * val_split)
        
        # Split the data
        diagnosis_val = diagnosis_df[:n_val]
        diagnosis_train = diagnosis_df[n_val:]
        
        # Append to main dataframes
        train_df = pd.concat([train_df, diagnosis_train])
        val_df = pd.concat([val_df, diagnosis_val])
    
    # Shuffle the final datasets
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\nDataset split statistics:")
    print(f"Training samples: {len(train_df)} ({(1-val_split)*100}%)")
    print(f"Validation samples: {len(val_df)} ({val_split*100}%)")
    
    # Function to copy images
    def copy_images(dataframe, split):
        successful_copies = 0
        failed_copies = 0
        
        for idx, row in dataframe.iterrows():
            src_path = os.path.join(images_path, row['image_id'] + '.jpg')
            dst_path = os.path.join(output_path, split, row['dx'], row['image_id'] + '.jpg')
            
            try:
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    successful_copies += 1
                else:
                    print(f"Warning: Image not found - {row['image_id']}")
                    failed_copies += 1
            except Exception as e:
                print(f"Error copying {row['image_id']}: {str(e)}")
                failed_copies += 1
        
        return successful_copies, failed_copies
    
    # Copy images to respective folders
    print("\nCopying training images...")
    train_success, train_failed = copy_images(train_df, 'train')
    print(f"Training: {train_success} images copied, {train_failed} failed")
    
    print("\nCopying validation images...")
    val_success, val_failed = copy_images(val_df, 'val')
    print(f"Validation: {val_success} images copied, {val_failed} failed")
    
    # Print distribution of classes
    print("\nClass distribution:")
    print("\nTraining set:")
    print(train_df['dx'].value_counts())
    print("\nValidation set:")
    print(val_df['dx'].value_counts())
    
    return train_df, val_df

def augment_class(input_dir, target_count=1000):
    """
    Augments images in a directory until reaching target_count.
    """
    files = os.listdir(input_dir)
    current_count = len(files)
    
    if current_count >= target_count:
        return
    
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='constant'
    )
    
    num_to_generate = target_count - current_count
    files_to_augment = files.copy()
    
    print(f"Augmenting class {os.path.basename(input_dir)} from {current_count} to {target_count} images")
    
    i = 0
    while i < num_to_generate:
        for file in files_to_augment:
            if i >= num_to_generate:
                break
                
            img_path = os.path.join(input_dir, file)
            img = load_img(img_path, target_size=(300, 300))
            x = img_to_array(img)
            x = np.expand_dims(x, 0)
            
            for batch in datagen.flow(
                x, 
                batch_size=1,
                save_to_dir=input_dir,
                save_prefix=f'aug_{i}',
                save_format='jpg'
            ):
                i += 1
                break

def balance_dataset(base_path, images_path, metadata_path, output_path, val_split=0.2):
    """
    Organizes and balances the dataset through augmentation.
    """
    # First organize the dataset normally
    train_df, val_df = organize_dataset(base_path, images_path, metadata_path, output_path, val_split)
    
    # Augment training data for underrepresented classes
    train_path = os.path.join(output_path, 'train')
    for class_dir in os.listdir(train_path):
        class_path = os.path.join(train_path, class_dir)
        augment_class(class_path, target_count=1000)
    
    return train_df, val_df

def create_model():
    """
    Creates an improved CNN model for skin lesion classification.
    """
    model = tf.keras.Sequential([
        # First Convolutional Block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Second Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Third Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Dense Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation='softmax')  # 7 classes
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_generators(output_path, batch_size=32):
    """
    Creates training and validation data generators.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='constant'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(output_path, 'train'),
        target_size=(300, 300),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(output_path, 'val'),
        target_size=(300, 300),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, val_generator

def main():
    # Define paths
    base_path = './HAM10000'
    images_path = os.path.join(base_path, 'images')
    metadata_path = os.path.join(base_path, 'metadata.csv')
    output_path = os.path.join(base_path, 'organized_dataset')
    
    # Balance and organize dataset
    train_df, val_df = balance_dataset(base_path, images_path, metadata_path, output_path)
    
    # Create generators
    train_generator, val_generator = create_generators(output_path)
    
    # Create and train model
    model = create_model()
    print(model.summary())
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=100,  # Increased epochs since we have early stopping
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    # Save training history
    pd.DataFrame(history.history).to_csv('training_history.csv')

if __name__ == "__main__":
    main()