import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


CIFAR10_DIR =  'C:/Users/User/OneDrive/Υπολογιστής/CIFAR_KNN/cifar-10-batches-py'
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_pickle_batch(filename):
    """Load a single CIFAR-10 batch from a pickle file."""
    with open(filename, 'rb') as f:
        # Use 'latin1' encoding for Python 3 compatibility with Python 2 pickled data
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        # Αναμόρφωση δεδομένων εικόνας
        
        return X, np.array(Y)

def load_cifar10(root_dir):
    
    
    # Φόρτωση των 5 training batches
    X_train_batches, y_train_batches = [], []
    for i in range(1, 6):
        fpath = os.path.join(root_dir, 'data_batch_%d' % (i, ))
        X, Y = load_pickle_batch(fpath)
        X_train_batches.append(X)
        y_train_batches.append(Y)    
    
    # Concatenate all training batches
    X_train = np.concatenate(X_train_batches)
    y_train = np.concatenate(y_train_batches)
    
    # Φόρτωσε το test batch
    fpath = os.path.join(root_dir, 'test_batch')
    X_test, y_test = load_pickle_batch(fpath)
    
    # Data flattened (N, 3072) από pickle 
    # Κανονικοποίηση των pixel
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    return X_train, y_train, X_test, y_test

# Load the data using the custom function
print("Φόρτωση δεδομένων από pickle batches...")
X_train_flat, y_train, X_test_flat, y_test = load_cifar10(CIFAR10_DIR)

print(f"Moρφή των τrain data : {X_train_flat.shape}")
print(f"Μορφή των Test data : {X_test_flat.shape}")

def get_image_from_vector(vector):
    """
    Μετατρέπει ένα επίπεδο διάνυσμα (3072,) πίσω σε έναν πίνακα εικόνας (32, 32, 3)
    και διασφαλίζει τον σωστό τύπο uint8 για εμφάνιση.
    
    :param vector: Ένα διάνυσμα 3072 διαστάσεων.
    :return: Ένας πίνακας εικόνας 32x32x3 τύπου np.uint8.
    """
    # Επαναδιαμόρφωση του διανύσματος σε (3, 32, 32)
    image_temp = vector.reshape(3, 32, 32)
    
    # Μεταφορά στη μορφή (32, 32, 3) που απαιτείται από το Matplotlib (Ύψος, Πλάτος, Κανάλια)
    image = image_temp.transpose(1, 2, 0)
    
    # Διασφάλιση ότι τα δεδομένα είναι στο εύρος 0-255 και τύπου uint8
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Αν είναι κανονικοποιημένο (0.0 έως 1.0), το επαναφέρουμε
        image = (image * 255).astype(np.uint8)
    else:
        # Απλώς διασφαλίζουμε ότι ο τύπος είναι uint8
        image = image.astype(np.uint8)

    return image



class NearestCentroidFromScratch:
    
    def __init__(self):
        self.centroids = {}
        self.classes = None

    def fit(self, X, y):
        """Calculates the centroid (mean) for each class."""
        self.classes = np.unique(y)
        
        for c in self.classes:
            # Select all samples belonging to the current class 'c'
            X_c = X[y == c]
            # Compute the centroid: the mean of all samples in the class
            self.centroids[c] = np.mean(X_c, axis=0)
            
        print(f"Υπολογισμός των κεντρών για  {len(self.classes)} κλάσεις.")

    def predict(self, X):
        """Predicts the class label for each sample in X."""
        predictions = []
        
        # Iterate over each test sample
        for x_sample in X:
            distances = {}
            
            # Ευκλειδια απόσταση για κάθε centroid
            for c, centroid in self.centroids.items():
                distance_sq = np.sum((x_sample - centroid) ** 2)
                distances[c] = distance_sq
            
            # class with the minimum distance (the nearest centroid)
            nearest_class = min(distances, key=distances.get)
            predictions.append(nearest_class)
            
        return np.array(predictions)

# Initialize and train the classifier
nc_model = NearestCentroidFromScratch()
print("Training Nearest Centroid Classifier...")
nc_model.fit(X_train_flat, y_train)


 
def main():
    """
    Κύρια συνάρτηση για τον Nearest Centroid classification 
    """
    
    # 1. Load Data
    
    try:
        X_train_flat, y_train, X_test_flat, y_test = load_cifar10(CIFAR10_DIR)
        print(f"Φόρτωση δεδομένων: Train {X_train_flat.shape}, Test {X_test_flat.shape}")
    except FileNotFoundError:
        print(f"Δεν βρέθηκε το CIFAR-10 data directory στο path '{CIFAR10_DIR}'.")
        
        return # Exit the function if data isn't found

    # 2. Initialize and Train the Classifier
    nc_model = NearestCentroidFromScratch()
    
    nc_model.fit(X_train_flat, y_train)

    # 3. Predict and Evaluate
    
    y_pred = nc_model.predict(X_test_flat)

    # Calculate Accuracy
    correct_predictions = np.sum(y_pred == y_test)
    total_samples = len(y_test)
    accuracy = correct_predictions / total_samples

    # 4. Report Results
    print("\n" + "="*40)
    print(f"Τελικά αποτελέσματα για πλήθος test samples: {total_samples}")
    print(f"Σωστά ταξινομημένα: {correct_predictions}")
    print(f"Ακρίβεια (Accuracy): {accuracy*100:.2f}%")
    print("="*40)
    try:
        X_train, y_train, X_test, y_test = load_cifar10(CIFAR10_DIR)
        
        
        X = X_train
        y = y_train
        
    except FileNotFoundError as e:
        print(f"\n ERROR: {e}")
        print(f"Please check that your CIFAR10_DIR variable is set correctly: {CIFAR10_DIR}")
        return

    

    # --- Plotting Logic ---
    num_plots = 10
    fig, axes = plt.subplots(1, num_plots, figsize=(15, 2))
    fig.suptitle('Sample CIFAR-10 Images', fontsize=16)

    # 10 τυχαίοι δείκτες για απεικόνιση των εικόνων του dataset
    random_indices = np.random.choice(X.shape[0], num_plots, replace=False)

    for i, idx in enumerate(random_indices):
        # 1.  flattened vector
        img_vector = X[idx]
        label = y[idx]
        
        # 2. Reshape to image format (32, 32, 3)
        img_display = get_image_from_vector(img_vector)
        
        # 3. Plotting
        ax = axes[i]
        ax.imshow(img_display)
        ax.set_title(CLASS_NAMES[label], fontsize=8)
        ax.axis('off') # Hide axes ticks and labels

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    

   

# The standard entry point for running a Python script
if __name__ == '__main__':
    main()