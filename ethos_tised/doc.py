import csv
import os

df_knn = []
df_minutal = []


def data_knn():
    global df_knn
    data_path = os.path.join(os.path.dirname(__file__), "data", "input_knn.csv")
    with open(data_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        df_knn = [row for row in reader]


def data_minutal():
    global df_minutal
    data_path = os.path.join(os.path.dirname(__file__), "data", "minutal_new.csv")
    with open(data_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        df_minutal = [row for row in reader]
