import os
import pandas as pd
import torch
from utils.constants import LIGAND_DIR

def read_embedding(ligand_id: str, chain: str) -> tuple[torch.Tensor, torch.Tensor]:
        embedding_path_tar = os.path.join(LIGAND_DIR, ligand_id,  chain + '_scannet.pkl.gz')
        if not os.path.exists(embedding_path_tar):
            raise ValueError

def add_has_scannet_embedding_column(df_path: str, base_data_path: str) -> None:
    df = pd.read_csv(df_path)
    
    # Check if embeddings can be read for each row
    has_embedding_column = []
    for idx, row in df.iterrows():
        try:
            # Try reading the embeddings for the ligand and proteins
            read_embedding(ligand_id=row['Ligand_ID'], chain=row['tar_protein'])
            read_embedding(ligand_id=row['Ligand_ID'], chain=row['src_protein'])
            has_embedding_column.append(True)
        except:
            has_embedding_column.append(False)
    
    # Add the new column to the DataFrame
    df['has_scannet_embedding'] = has_embedding_column
    df.to_csv(df_path, index=False)

if __name__ == "__main__":
    data_paths = ['results/alignment_results/2024-07-17_19-21-30_10000.csv']
    
    for data_path in data_paths:
        add_has_scannet_embedding_column(data_path, LIGAND_DIR)