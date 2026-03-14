
import numpy as np
import pickle
import os
import sys
import time
from collections import Counter


# --- ΡΥΘΜΙΣΕΙΣ ΥΠΟΣΥΝΟΛΩΝ ---
# Ορίζουμε το μέγεθος των υποσυνόλων για γρήγορη εκτέλεση.
num_training = 1000  # Αριθμός δειγμάτων εκπαίδευσης
num_test = 100       # Αριθμός δειγμάτων δοκιμής
current_metric='L2'

# Ορισμός διαδρομής (Βεβαιωθείτε ότι ο φάκελος CIFAR-10 υπάρχει εδώ)
data_dir = 'C:/Users/User/OneDrive/Υπολογιστής/CIFAR_KNN/cifar-10-batches-py'
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# --------------------------------------------------------------------------
# 1. Βοηθητικές Συναρτήσεις Φόρτωσης Δεδομένων
# --------------------------------------------------------------------------

def load_cifar_batch(filename):
    """Φορτώνει ένα batch του CIFAR-10 από αρχείο pickle."""
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        # Αναμόρφωση δεδομένων εικόνας: N x 3 x 32 x 32 -> N x 32 x 32 x 3
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_cifar10(data_dir):
    """Φορτώνει ΟΛΑ τα δεδομένα και τα επιστρέφει."""
    # Φόρτωση training data (5 batches)
    X_train_list = []
    Y_train_list = []
    for i in range(1, 6):
        f = os.path.join(data_dir, 'data_batch_%d' % (i,))
        X, Y = load_cifar_batch(f)
        X_train_list.append(X)
        Y_train_list.append(Y)
    X_train = np.concatenate(X_train_list)
    Y_train = np.concatenate(Y_train_list)

    # Φόρτωση test data
    X_test, Y_test = load_cifar_batch(os.path.join(data_dir, 'test_batch'))

    return X_train, Y_train, X_test, Y_test



# --------------------------------------------------------------------------
# 2. Υλοποίηση του k-NN
# --------------------------------------------------------------------------

class KNearestNeighbor(object):
    """ Ταξινομητής kNN """

    def __init__(self):
        pass

    def train(self, X, y):
        """ Απομνημόνευση των δεδομένων εκπαίδευσης (lazy learner). """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, metric='L2'):
        """ Προβλέπει τις ετικέτες για τα δεδομένα δοκιμής X (εδώ k=1). """
        num_test = X.shape[0]
        Y_pred=np.zeros(num_test,dtype=self.y_train.dtype)
        
        print(f"Υπολογισμός {num_test}x{self.X_train.shape[0]} αποστάσεων...")
        start_time = time.time()
        
        # Υπολογισμός αποστάσεων L2 
        dists = np.sqrt(np.sum(np.square(X[:, np.newaxis, :]  - self.X_train), axis=2))

        end_time = time.time()
        print(f"Ο υπολογισμός αποστάσεων ολοκληρώθηκε σε {end_time - start_time:.2f} δευτερόλεπτα.")


        # Εύρεση του πλησιέστερου γείτονα (k=1)
        # argmin βρίσκει τον δείκτη της ελάχιστης απόστασης σε κάθε σειρά
        #min_indices = np.argmin(dists, axis=1)
        
        # Η πρόβλεψη είναι η ετικέτα του πλησιέστερου δείγματος
        #Y_pred = self.y_train[min_indices]
        for i in range(num_test):
            k_closest_indices=np.argsort(dists[i,:])[:k]
            k_nearest_labels=self.y_train[k_closest_indices]
            Y_pred[i]=Counter(k_nearest_labels).most_common(1)[0][0]

        return Y_pred
    
   


# --------------------------------------------------------------------------
# 3. Κύριο Script Εκτέλεσης
# --------------------------------------------------------------------------

if __name__ == '__main__':
    print("--- Φόρτωση Δεδομένων CIFAR-10 ---")
    try:
        X_full_train, Y_full_train, X_full_test, Y_full_test = load_cifar10(data_dir)
    except FileNotFoundError:
        print(f"Σφάλμα: Βεβαιωθείτε ότι ο φάκελος '{data_dir}' υπάρχει και περιέχει τα αρχεία.")
        sys.exit()

    # --- Εφαρμογή Υποσυνόλων ---
    # Επιλέγουμε τα πρώτα N δείγματα από το πλήρες σύνολο δεδομένων
    X_train = X_full_train[:num_training]
    Y_train = Y_full_train[:num_training]

    X_test = X_full_test[:num_test]
    Y_test = Y_full_test[:num_test]

    print(f"Χρησιμοποιούμε: {num_training} δείγματα εκπαίδευσης και {num_test} δοκιμής.")

    # Αναμόρφωση (Flatten) των εικόνων σε διανύσματα (N, 3072)
    X_train_flat = X_train.reshape(num_training, -1)
    X_test_flat = X_test.reshape(num_test, -1)
    
    print(f"Σχήμα X_train (Flatten): {X_train_flat.shape}")

    # Δημιουργία και εκπαίδευση (απομνημόνευση) του ταξινομητή
    classifier = KNearestNeighbor()
    classifier.train(X_train_flat, Y_train)

    # Πρόβλεψη με k και Ευκλείδια απόσταση
    k_value = 1
    print(f"\n--- Έναρξη Ταξινόμησης k-NN με k={k_value}")
    
    Y_pred = classifier.predict(X_test_flat, k=k_value, metric='L2')

    # Υπολογισμός ακρίβειας
    num_correct = np.sum(Y_pred == Y_test)
    accuracy = float(num_correct) / num_test

    print("\n==============================================")
    print(f"Τελικά Αποτελέσματα για {num_training}/{num_test} δείγματα:")
    print(f"  Σωστά ταξινομημένα: {num_correct} / {num_test}")
    print(f"  Ακρίβεια (Accuracy): {accuracy * 100:.2f} %")
    print("==============================================")
    
    