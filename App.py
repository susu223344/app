import shap
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt


def load_model1():
	  return joblib.load("clf.pkl")
def load_model2():
	  return joblib.load("clf2.pkl")
    
model1 = load_model1()
model2 = load_model2()

if 'model1' not in st.session_state:
    model1 = model1
    model2 = model2
    st.session_state["model1"] = model1
    st.session_state["model2"] = model2
else:
    model1 = st.session_state["model1"]
    model2 = st.session_state["model2"]

# if 'model1' not in st.session_state:
#     model1 = joblib.load('clf.pkl')
#     model2 = joblib.load('clf2.pkl')
#     st.session_state["model1"] = model1
#     st.session_state["model2"] = model2
# else:
#     model1 = st.session_state["model1"]
#     model2 = st.session_state["model2"]


scaler = joblib.load("Scaler.pkl")
continuous_vars = ['Age','Anion gap','Respiratory rate','APSIII']

st.set_page_config(layout='wide')

st.write("<h1 style='text-align: center'>Prediction for Device-Associated Infections and 30-Day Survival Outcome of Intensive Care Unit Patients Undergoing Invasive Device Procedures</h1>",
         unsafe_allow_html=True)
st.markdown('-----')
dic1 = {
    'Male': 1,
    'Female': 0
}
dic2 = {
    'Yes': 1,
    'No': 0
}
with st.sidebar:
    st.markdown("# Input variable")
    st.markdown('-----')
    IMV = st.selectbox("Invasive mechanical ventilation (IMV)", ["Yes", "No"])
    IUC = st.selectbox("Indwelling urinary catheter (IUC)", ["Yes", "No"])
    CVC = st.selectbox("Central venous catheter (CVC)", ["Yes", "No"])
    Age = st.text_input("Age (years)")
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Anion_gap = st.text_input("Anion gap (mmol/L)")
    Respiratory_rate = st.text_input("Respiratory rate (min)")
    Liver_disease = st.selectbox("Liver disease", ["Yes", "No"])
    Myocardial_infarct = st.selectbox("Myocardial infarct", ["Yes", "No"])
    Malignant_cancer = st.selectbox("Malignant cancer", ["Yes", "No"])
    Tracheostomy = st.selectbox("Tracheostomy", ["Yes", "No"])
    APSIII = st.text_input("APSIII")
    Black = st.selectbox("Black", ["Yes", "No"])
    MICU_SICU = st.selectbox("Medical or surgical intensive care unit (MICU/SICU)", ["Yes", "No"])
if st.sidebar.button("Predict"):
    with st.spinner("Forecast, please wait..."):
        st.header('1. Prediction for device-associated infection')
        test_df = pd.DataFrame([dic2[IMV], dic2[IUC], dic2[CVC], dic1[Gender],
                                dic2[Liver_disease], dic2[Myocardial_infarct], dic2[Malignant_cancer],
                                dic2[MICU_SICU]],
                               index=['IMV', 'IUC', 'CVC', 'Gender', 'Liver disease', 'Myocardial infarct',
                                      'Malignant cancer', 'MICU/SICU']).T
        explainer = shap.Explainer(model1)  # 创建解释器
        shap_ = explainer.shap_values(test_df)
        shap.waterfall_plot(
            shap.Explanation(values=shap_[0, :], base_values=explainer.expected_value, data=test_df.iloc[0, :]),
            show=False)
        plt.tight_layout()
        plt.savefig('shap1.png', dpi=300)
        col1, col2, col3 = st.columns([2, 5, 3])
        shap.initjs()
        shap.force_plot(explainer.expected_value, shap_[0, :], test_df.iloc[0, :], show=False, matplotlib=True,
                        figsize=(20, 5))
        plt.xticks(fontproperties='Times New Roman', size=16)
        plt.yticks(fontproperties='Times New Roman', size=16)
        plt.tight_layout()
        plt.savefig('shap2.png', dpi=300)
        with col2:
            st.image('shap1.png')
        st.image('shap2.png')
        st.success("Probability of device-associated infection: {:.3f}%".format(model1.predict_proba(test_df)[:, 1][0] * 100))
        st.header('2. 30-day Kaplan-Meier survival curve')
        
        test_df2 = pd.DataFrame(
            [dic2[IMV], float(Age), float(Anion_gap), float(Respiratory_rate), dic2[Tracheostomy], float(APSIII),
             dic2[Black],
             dic2[MICU_SICU]],
            index=['IMV', 'Age', 'Anion gap', 'Respiratory rate', 'Tracheostomy', 'APSIII',
                   'Black', 'MICU/SICU']).T
        test_df2[continuous_vars] = scaler.transform(test_df2[continuous_vars])
        surv_funcs = model2.predict_survival_function(test_df2)
        col4, col5, col6 = st.columns([2, 5, 3])
        fig, ax = plt.subplots()
        for fn in surv_funcs:
            ax.step(fn.x, fn(fn.x), where="post", color="#8dd3c7", lw=2)
        plt.ylabel("Probability of survival (%)")
        plt.xlabel("Time since received the first invasive procedure (days)")
        plt.grid()
        with col5:
            st.pyplot(fig)
