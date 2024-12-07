from google.cloud import firestore

# Inisialisasi Firestore
def init_firestore():
    return firestore.Client()
