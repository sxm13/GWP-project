from rdkit.Chem import Draw
import pandas as pd
from rdkit import Chem
import pubchempy

def show(csvfile,name):

    df = pd.read_csv(csvfile,name)
    smiles = pd.read_csv(csvfile)["SMILE"]
    
    mols = []
    names = smiles.tolist()

    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        mols.append(mol)
        
    if name == "smiles":
        vis = Draw.MolsToGridImage(mols,
                          molsPerRow = 5,
                          subImgSize = (200,200),
                           legends=names
                          )
    elif name == "molecular_formula":
        name_molecular_formula = []
        for smile in smiles:
            compounds = pubchempy.get_compounds(smile, namespace='smiles')
            match = compounds[0]
            name_molecular_formula.append(match.molecular_formula)
            

        vis = Draw.MolsToGridImage(mols,
                          molsPerRow = 5,
                          subImgSize = (200,200),
                           legends=name_molecular_formula
                          )
        
    elif name == "iupac_name":

        name_iupac = []
        for smile in smiles:
            compounds = pubchempy.get_compounds(smile, namespace='smiles')
            match = compounds[0]
            name_iupac.append(match.iupac_name)
        vis = Draw.MolsToGridImage(mols,
                          molsPerRow = 5,
                          subImgSize = (200,200),
                           legends=name_iupac
                          )
    return vis
