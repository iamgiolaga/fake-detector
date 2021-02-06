import pickle
from termcolor import colored

''' DESCRIPTION '''
''' This is the file where models are loaded and reused '''

fullname = "selected_models/02.12.2020_11.49/model_"
models = []

for i in range(0, 5):
    print(colored("Downloading model " + str(i) + "...", "red"))
    file = open(fullname + "(" + str(i) + ").pickle", "rb")
    models.append(pickle.load(file))

print(colored("Download completed!\n", "green"))

for i in range(0, len(models)):
    print("MODEL " + str(i) + ":")
    print(models[i])
