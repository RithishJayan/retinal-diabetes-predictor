def load_retina_data(csv_path, img_dir, img_size=224):
    import pandas as pd
    import numpy as np
    import os
    import cv2
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    images = []
    labels = []
    missing = 0

    for i, row in df.iterrows():
        image_id = row['id_code']
        label = row['diagnosis']
        img_path = os.path.join(img_dir, image_id + '.png')

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0
            images.append(img)
            labels.append(label)
        else:
            missing += 1

    print(f"✅ Loaded {len(images)} images. ❌ Skipped {missing} missing files.")
    
    images = np.array(images)
    labels = to_categorical(labels, num_classes=5)

    return train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)
