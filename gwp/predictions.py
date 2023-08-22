import joblib
import pandas as pd

def predict_data(csvfile, model, verbos=False, saveto: str="predicted.csv")-> pd.DataFrame:

    ML_model = joblib.load(model)

    X = pd.read_csv(csvfile)[['IP','EA','HomoLumoGap','MolWt','MolLogP','MolMR','HeavyAtomCount','LabuteASA','BalabanJ','BertzCT']]
    predicted = ML_model.predict(X)

    name = model.split('.')
    df_predicted = pd.DataFrame(predicted, columns=[name[0]])

    if saveto:
        df_predicted.to_csv(saveto, index=True, index_label='Index')

    return df_predicted
