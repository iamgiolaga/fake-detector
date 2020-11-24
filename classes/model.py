import pandas as pd
import os
import pickle

from datetime import datetime

class Model:

    def __init__(self, fuzzy_inductor):
        self.model = fuzzy_inductor
        self.date = datetime.now().strftime('%d.%m.%Y')
        self.time = datetime.now().strftime('%H.%M')

    def write_model(self):
        params = self.extract_params()
        # 1. read to see if any dataframe is already available
        # 2. if yes, read it and update it
        # 3. if not, create it and fill it

        outdir = "selected_models/"
        outname = "models.csv"

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        fullname = os.path.join(outdir, outname)

        try:
            models = pd.read_csv("selected_models/models.csv")
        except:
            column_names = ["Date", "Time", "Model"]
            models = pd.DataFrame(columns=column_names)

        models = models.append({
            "Date": self.date, "Time": self.time, "Model": params
        }, ignore_index=True)

        models.to_csv(fullname, index=False)

        outdir = "selected_models/"
        outname = "model_" + self.date + "_" + self.time


        if not os.path.exists(outdir):
            os.makedirs(outdir)

        fullname = os.path.join(outdir, outname)

        file = open(fullname, "wb")
        pickle.dump(self, file)
        file.close()

    def extract_params(self):
        return self.model.get_params()