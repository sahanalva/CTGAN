import pandas as pd
import numpy as np
import torch
import synthesizer

INPUT_FILE_PATH = "~/Downloads/toy_data.csv"
SAVE_MODEL_PATH = "/Users/sahanalva/Counterfactual Research/toy_data_full_200.pt"

df = pd.read_csv(INPUT_FILE_PATH)
required_cols = ['loan_type_name','loan_purpose_name','agency_abbr','property_type_name', \
                 'state_name','preapproval_name', 'purchaser_type_name','loan_amount_000s', \
                 'tract_to_msamd_income', 'population','number_of_1_to_4_family_units','minority_population',\
                 'applicant_income_000s','hud_median_family_income','number_of_owner_occupied_units']
conditional_cols = None

df = df[required_cols]
nunique = df.apply(pd.Series.nunique)
discrete_columns = nunique[nunique < 100].index
numeric_columns = df.select_dtypes(include=['number']).columns
object_columns = df.select_dtypes(include=['object']).columns
df[numeric_columns] = df[numeric_columns].fillna(0)
df[object_columns] = df[object_columns].fillna("NA")

model = synthesizer.CTGANSynthesizer(embedding_dim = 128, batch_size=500)
model.fit(df, discrete_columns, conditional_cols, 200)
torch.save(model, SAVE_MODEL_PATH)