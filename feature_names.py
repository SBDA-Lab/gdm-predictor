import joblib

feature_names = [
    "cg10139015", "cg23507676", "cg11001216", "cg04539775", "cg04985016",
    "Age_Yon", "V1_Height_Waist_ratio", "V2_family_his_diab_Yon", "V2_pt_of_hba1c_Yon",
    "V2_BP_rding1_sys_Yon", "V2_BP_rding1_dia_Yon", "V1_Gesti_age_by_LMP_weeks_Yon",
    "V1_BMI_calc_Yon", "Soci_class_code_Yon", "V1_soci_class_imp_Yon", "V2_venous_hba1c_Yon",
    "V1_Height_Yon", "V1_Weight_Yon", "V2_waist_r1_Yon", "V1_parity_Yon",
    "Pre_preg_wt", "V3_Ven_Fastin", "V3_Ven_60_min", "V3_Ven_120_min", "V3ven_HbA1c"
]

joblib.dump(feature_names, "feature_names.joblib")

