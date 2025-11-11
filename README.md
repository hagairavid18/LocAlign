
## How to run LocAlign inference:

### 1. Install the inference environment
```bash
conda env create -f inference_env.yaml
conda activate inference_env
```

### 2. Run inference
You can run inference either with a CSV file containing protein pairs or with a single protein pair:

**Option A: Using a CSV file**
```bash
python scripts/inference.py \
    --checkpoint checkpoints/your-model/best-checkpoint.ckpt \
    --csv_path path/to/protein_pairs.csv \
    --ligand_id ATP \
    --scannet_dir /path/to/scannet/embeddings \
    --base_save_dir inference_results
```

**Option B: Using a single protein pair**
```bash
python scripts/inference.py \
    --checkpoint checkpoints/your-model/best-checkpoint.ckpt \
    --protein_pair 1ABC A 2XYZ B \
    --ligand_id ATP \
    --scannet_dir /path/to/scannet/embeddings \
    --base_save_dir inference_results
```

Optional: Specify a custom output directory name with `--save_dir my_run_name`

### 3. Output files
For each protein pair, the inference pipeline generates a folder with the following visualization files:

**Generated PDB files:**
- `template_receptor.pdb` - Template protein structure (without ligand)
- `template_ligand.pdb` - Template ligand structure
- `transformed_query_receptor.pdb` - Query protein transformed to align with template
- `transformed_query_ligand.pdb` - Query ligand transformed to align with template

**Correspondence files:**
- `correspondences.pb` - Chimera pseudobond file showing atom-level correspondences between proteins

**Visualization scripts:**
- `chimera_script_pocket.cxc` - ChimeraX script highlighting binding pockets
- `chimera_script_motif.cxc` - ChimeraX script highlighting aligned structural motifs

### 4. Visualize results with ChimeraX
Open the visualization scripts directly in ChimeraX:
```bash
chimerax /path/to/inference_results/your_output_folder/chimera_script_pocket.cxc
```

Or manually open ChimeraX and load the script via: `File > Open > chimera_script_pocket.cxc`

The visualization shows:
- **Pocket version**: Highlights ligand binding pockets and ligands
- **Motif version**: Focuses on the aligned structural motifs and correspondences
- **Pseudobonds**: Lines connecting corresponding atoms between template and query proteins
