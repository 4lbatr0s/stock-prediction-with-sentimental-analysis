from typing import final
from keras.models import load_model
import pandas as pd
import numpy as np
import algoritma.final_model as final_model
model = load_model("mstf.h5")

merged_data = pd.read_excel("DB_MSFT.xlsx")
all_close_values = merged_data['Close'].values

# last_close_values = all_close_values[-5:]
# last_close_values =
prediction = model.predict(final_model.last_count_days)
print(prediction)

# test datasını kullanabilirim.
# trainin kullanmama gerek yok zaten modelde train ettim.

# modeli kaydet
# feature scaler ile test datasını .mat formatında kaydet.

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html
