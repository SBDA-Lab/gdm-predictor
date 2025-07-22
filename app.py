import streamlit as st
import pandas as pd
import torch
import joblib
import numpy as np
from torch_geometric.nn import SAGEConv
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix
from io import BytesIO

# === Load model and metadata ===
model_path = "graphsage_cpu_model.pth"
scaler = joblib.load("scaler.joblib")
feature_names = joblib.load("feature_names.joblib")

# === Column labels and default values ===
column_labels = {
    "cg10139015": "CpG Marker 1",
    "cg23507676": "CpG Marker 2",
    "cg11001216": "CpG Marker 3",
    "cg04539775": "CpG Marker 4",
    "cg04985016": "CpG Marker 5",
    "Age_Yon": "Age (years)",
    "V1_Height_Waist_ratio": "Height-to-Waist Ratio (V1)",
    "V2_family_his_diab_Yon": "Family History of Diabetes (V2)",
    "V2_pt_of_hba1c_Yon": "Aware of HbA1c (V2)",
    "V2_BP_rding1_sys_Yon": "Systolic BP (V2)",
    "V2_BP_rding1_dia_Yon": "Diastolic BP (V2)",
    "V1_Gesti_age_by_LMP_weeks_Yon": "Gestational Age by LMP (Weeks)",
    "V1_BMI_calc_Yon": "BMI (V1)",
    "Soci_class_code_Yon": "Socioeconomic Class Code",
    "V1_soci_class_imp_Yon": "Socioeconomic Class (Imputed)",
    "V2_venous_hba1c_Yon": "Venous HbA1c (V2)",
    "V1_Height_Yon": "Height (cm)",
    "V1_Weight_Yon": "Weight (kg)",
    "V2_waist_r1_Yon": "Waist Circumference (V2)",
    "V1_parity_Yon": "Parity (V1)",
    "Pre_preg_wt": "Pre-pregnancy Weight",
    "V3_Ven_Fastin": "Fasting Glucose (V3)",
    "V3_Ven_60_min": "60-min Glucose (V3)",
    "V3_Ven_120_min": "120-min Glucose (V3)",
    "V3ven_HbA1c": "Venous HbA1c (V3)"
}

default_values = {
    "Age_Yon": 27, "V1_BMI_calc_Yon": 24, "V2_BP_rding1_sys_Yon": 115,
    "V2_BP_rding1_dia_Yon": 75, "V2_family_his_diab_Yon": 0,
    "V2_pt_of_hba1c_Yon": 1, "Soci_class_code_Yon": 1,
}

# === GraphSAGE Model ===
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GraphSAGE(len(feature_names), 32, 2)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# === GDM Predictor Function ===
def preprocess(df):
    # Infer and encode socio class
    if "Soci_class_code_Yon" in df.columns:
        df["V1_soci_class_imp_Yon"] = df["Soci_class_code_Yon"].map({0.0: "Lower_Class", 1.0: "Middle_Class", 2.0: "Upper_Class"})
    if "V1_soci_class_imp_Yon" in df.columns:
        df["V1_soci_class_imp_Yon"] = df["V1_soci_class_imp_Yon"].map({"Lower_Class": 0, "Middle_Class": 1, "Upper_Class": 2})
    return df

def predict_gdm(df):
    df = preprocess(df)
    df_scaled = scaler.transform(df[feature_names])
    x = torch.tensor(df_scaled, dtype=torch.float)

    if len(df_scaled) < 2:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    else:
        knn_graph = kneighbors_graph(df_scaled, n_neighbors=5, mode='connectivity', include_self=False)
        edge_index = from_scipy_sparse_matrix(knn_graph)[0]

    out = model(x, edge_index)
    prob = torch.softmax(out, dim=1)[:, 1].detach().numpy()
    pred = (prob >= 0.5).astype(int)
    return pred, prob

# === UI Starts Here ===
st.set_page_config(page_title="GDM Predictor", layout="wide")
st.title("🤰 GDM Prediction Tool (GraphSAGE)")
st.markdown("Predict **Gestational Diabetes Mellitus (GDM)** using a trained **Graph Neural Network (GraphSAGE)**.")

# --- Manual Input ---
with st.expander("🔍 Single Patient Prediction", expanded=True):
    st.markdown("Please enter the patient's medical values below:")
    inputs = {}
    for col in feature_names:
        label = column_labels.get(col, col)
        if col == "V1_soci_class_imp_Yon":
            continue  # will be derived
        elif col == "Soci_class_code_Yon":
            inputs[col] = st.radio(f"{label} (0=Lower, 1=Middle, 2=Upper)", [0, 1, 2], index=default_values.get(col, 1))
        elif col in ["V2_family_his_diab_Yon", "V2_pt_of_hba1c_Yon"]:
            inputs[col] = st.radio(f"{label} (0=No, 1=Yes)", [0, 1], index=default_values.get(col, 0))
        else:
            min_val = 0.0
            max_val = 300.0
            default = default_values.get(col, 0.0)
            inputs[col] = st.slider(f"{label} (based on data/domain)", min_value=min_val, max_value=max_val, value=float(default), step=0.1)

    if st.button("🔎 Predict GDM"):
        try:
            input_df = pd.DataFrame([inputs])
            input_df["V1_soci_class_imp_Yon"] = input_df["Soci_class_code_Yon"].map({0: "Lower_Class", 1: "Middle_Class", 2: "Upper_Class"})
            input_df = preprocess(input_df)

            pred, prob = predict_gdm(input_df)
            risk = prob[0] * 100
            st.metric("Risk Probability", f"{risk:.2f} %")
            if pred[0] == 1:
                st.error("🟥 **Prediction: GDM Likely**")
            else:
                st.success("🟩 **Prediction: No GDM**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --- Batch Prediction ---
st.markdown("---")
st.subheader("📄 Batch Prediction via CSV/SPSS Upload")

st.markdown("**Instructions:** Upload a `.csv` or `.sav` (SPSS) file with all required features.")
file = st.file_uploader("Upload CSV or SPSS file", type=["csv", "sav"])

if file:
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            import pyreadstat
            df, _ = pyreadstat.read_sav(file)

        df = preprocess(df)
        missing = [f for f in feature_names if f not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            pred, prob = predict_gdm(df)
            df["GDM_Prediction"] = ["GDM" if p == 1 else "No GDM" for p in pred]
            df["Risk_Probability (%)"] = (prob * 100).round(2)

            st.success("✅ Prediction completed!")
            show_df = df.rename(columns=column_labels)
            st.dataframe(show_df)

            # --- Download option ---
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results (CSV)", csv, "GDM_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")
