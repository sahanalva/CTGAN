import pandas as pd
import numpy as np
import torch


df = pd.read_csv("~/ctgan_all/ctgan_data/toy_data.csv")
required_cols = ['applicant_sex', 'applicant_income_000s', 'applicant_sex_name','agency_abbr']
df = df[required_cols]
nunique = df.apply(pd.Series.nunique)
discrete_columns = nunique[nunique < 100].index
numeric_columns = df.select_dtypes(include=['number']).columns
object_columns = df.select_dtypes(include=['object']).columns
df[numeric_columns] = df[numeric_columns].fillna(0)
df[object_columns] = df[object_columns].fillna("NA")
conditional_cols = ['applicant_sex','agency_abbr']

data = df.loc[0:100000,:]
model = synthesizer.CTGANSynthesizer(embedding_dim = 128, batch_size=250)
model.fit(data, discrete_columns, conditional_cols, 100,log_frequency= False)
torch.save(model, "~/ctgan_all/ctgan_models/temp.pt")