import streamlit as st
import pandas as pd
import torch
import joblib
import numpy as np
from torch_geometric.nn import SAGEConv
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix

# === Load model, scaler, and feature metadata ===
model_path = "graphsage_cpu_model.pth"
scaler = joblib.load("scaler.joblib")
feature_names = joblib.load("feature_names.joblib")

# === Human-readable column labels ===
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

# === GraphSAGE Model Definition ===
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

# === Load and Prepare Model ===
model = GraphSAGE(in_channels=len(feature_names), hidden_channels=32, out_channels=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# === Prediction Logic ===
def predict_gdm(df):
    df_scaled = scaler.transform(df[feature_names])
    x = torch.tensor(df_scaled, dtype=torch.float)

    if len(df_scaled) < 2:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # Self-loop for single sample
    else:
        knn_graph = kneighbors_graph(df_scaled, n_neighbors=5, mode='connectivity', include_self=False)
        edge_index = from_scipy_sparse_matrix(knn_graph)[0]

    out = model(x, edge_index)
    prob = torch.softmax(out, dim=1)[:, 1].detach().numpy()
    pred = (prob >= 0.5).astype(int)
    return pred, prob

# === Streamlit Interface ===
st.set_page_config(page_title="GDM Predictor", layout="wide")
st.title("🤰 GDM Prediction Tool (GraphSAGE)")
st.markdown("Predict **Gestational Diabetes Mellitus** (GDM) using a trained Graph Neural Network (GraphSAGE).")

# --- Manual Prediction Form ---
with st.expander("🔍 Single Patient Prediction (Manual Input)", expanded=True):
    manual_inputs = {}
    for col in feature_names:
        label = column_labels.get(col, col)
        val = st.number_input(f"{label}", value=0.0, format="%.2f")
        manual_inputs[col] = val

    if st.button("🔎 Predict GDM (Manual)"):
        try:
            manual_df = pd.DataFrame([manual_inputs])
            pred, prob = predict_gdm(manual_df)
            st.success(f"**Prediction:** {'🟥 GDM' if pred[0] == 1 else '🟩 No GDM'}")
            st.metric("Risk Probability", f"{prob[0]*100:.2f} %")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --- CSV Upload for Batch Prediction ---
st.markdown("---")
st.subheader("📄 Multiple Patient Prediction (CSV Upload)")

st.markdown("""
**Instructions:**
- Upload a `.csv` file containing patient records.
- Each row should represent a single patient.
- The file **must contain the following columns**:
  - {}
- Column names should **exactly match** the format used in the training data.
- You will receive GDM prediction (Yes/No) and risk probability (%) for each row.
""".format(", ".join(feature_names)))

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        batch_df = pd.read_csv(uploaded_file)
        missing = [f for f in feature_names if f not in batch_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            pred, prob = predict_gdm(batch_df)
            batch_df["GDM_Prediction"] = ["GDM" if p == 1 else "No GDM" for p in pred]
            batch_df["Risk_Probability (%)"] = (prob * 100).round(2)
            st.dataframe(batch_df)
    except Exception as e:
        st.error(f"Error during prediction: {e}")

