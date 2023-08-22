from rdkit.Chem import Draw
import pandas as pd
from rdkit import Chem


def show(csvfile):

    df = pd.read_csv(csvfile)
    smiles = pd.read_csv(csvfile)["SMILE"]
    
    mols=[]
    names=smiles.tolist()
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        mols.append(mol)
        
    vis = Draw.MolsToGridImage(mols,
                          molsPerRow = 5,
                          subImgSize = (200,200),
                           legends=names
                          )
    return vis
