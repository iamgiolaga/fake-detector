import pickle

fullname = "preprocessed_datasets/text/28.12.2020_18.48/dataset_fake.pickle"
file = open(fullname, "rb")
dataset_fake = pickle.load(file)

fullname = "preprocessed_datasets/text/28.12.2020_18.48/dataset_true.pickle"
file = open(fullname, "rb")
dataset_true = pickle.load(file)

print(len(dataset_fake.aggregated))
print(len(dataset_true.aggregated))