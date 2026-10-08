# ------------------------------------------------------------------
# Paste this as the LAST cell of Pred_Maint.ipynb and run it after
# cell 23 (permutation importance). It writes the three small files
# the Streamlit app reads, then zips them for download from Colab.
# Variable names match the notebook: test, p_te, fail, imp.
# ------------------------------------------------------------------
import os

os.makedirs("app/data", exist_ok=True)

app = test[["machineID", "datetime", "volt", "rotate", "pressure", "vibration", "y"]].copy()
app["risk"] = p_te
app = app.astype({"machineID": "int16", "volt": "float32", "rotate": "float32",
                  "pressure": "float32", "vibration": "float32", "risk": "float32", "y": "int8"})
app.sort_values(["machineID", "datetime"]).to_parquet("app/data/app_data.parquet", index=False)

fail[fail.datetime >= "2015-09-02"].to_csv("app/data/failures_test.csv", index=False)
imp.head(15).rename("importance").to_csv("app/data/feature_importance.csv")

print(len(app), "rows exported")   # expect 291100
!zip -r app_data.zip app/data
