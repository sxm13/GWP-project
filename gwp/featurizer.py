from pathlib import Path
import numpy as np
import pandas as pd
from typing import List
from rdkit import Chem
from rdkit.Chem import Descriptors

def featurize_smiles(csvfile,verbos=False, saveto: str="featuresforgwp.csv")-> pd.DataFrame:

    df = pd.read_csv(csvfile)
    smiles = pd.read_csv(csvfile)["SMILE"]
    
    features = []
    
    for smile in smiles:
        
        dft_features = df[df.SMILE==smile][["SMILE","IP","EA","HomoLumoGap"]].values[0].tolist()

        mol = Chem.MolFromSmiles(smile)
        MolWt = Chem.Descriptors.MolWt(mol)
        MolLogP = Chem.Descriptors.MolLogP(mol)
        MolMR = Chem.Descriptors.MolMR(mol)
        HeavyAtomCount = Chem.Descriptors.HeavyAtomCount(mol)
        LabuteASA = Chem.Descriptors.LabuteASA(mol)
        BalabanJ = Chem.Descriptors.BalabanJ(mol)
        BertzCT = Chem.Descriptors.BertzCT(mol)

        try:
            dft_target = df[df.SMILE==smile][["LF","GWP"]].values[0].tolist()
            rdkit_features = [MolWt,MolLogP,MolMR,HeavyAtomCount,LabuteASA,BalabanJ,BertzCT]
            all_features = dft_features + rdkit_features + dft_target
            features.append(all_features)
        except:
            rdkit_features = [MolWt,MolLogP,MolMR,HeavyAtomCount,LabuteASA,BalabanJ,BertzCT]
            all_features = dft_features + rdkit_features
            features.append(all_features)
    try:
        df_all_features = pd.DataFrame(features, columns=["SMILE",'IP','EA','HomoLumoGap','MolWt','MolLogP','MolMR','HeavyAtomCount','LabuteASA','BalabanJ','BertzCT'])
    except:
        df_all_features = pd.DataFrame(features, columns=["SMILE",'IP','EA','HomoLumoGap','MolWt','MolLogP','MolMR','HeavyAtomCount','LabuteASA','BalabanJ','BertzCT','LF','GWP'])
    
    if verbos:
        print(df_all_features)

    
        
    if saveto:
        df_all_features.to_csv(saveto, index=True, index_label='Number')
        
    return df_all_features
