import pandas as pd

class Model:

    def __init__(self, fuzzy_inductor):
        self.model = fuzzy_inductor

    def write_model(self):
        params = self.extract_params()
        # 1. read to see if any dataframe is already available
        # 2. if yes, read it and update it
        # 3. if not, create it and fill it
        try:
            models = pd.read_csv("selected_models/models.csv")
        except:
            column_names = ["Model"]
            models = pd.DataFrame(columns=column_names)

        models = models.append({"Model": params}, ignore_index=True)

        models.to_csv("selected_models/models.csv", index=False)

    def extract_params(self):
        return self.model.get_params()