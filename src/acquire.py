import os

# Write supporting functions here
def download_data():
    os.system("kaggle competitions download nlp-getting-started -f test.csv -p data/raw/")
    os.system("kaggle competitions download nlp-getting-started -f train.csv -p data/raw/")

def data_exists():
    return os.path.isfile("data/raw/test.csv") and os.path.isfile("data/raw/train.csv")

def run():
    print("Acquire: downloading raw data files...")

    if data_exists():
        print("Data already exists")
    else:
        download_data()
        print("Data acquired")

    print("Acquire: Completed!")
