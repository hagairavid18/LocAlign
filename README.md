
How to test virtual best buddy inference:
1. install the environment: conda env create -f aligner_dl/env.yaml.
2. run python aligner_dl/scripts/inference.py --config aligner_dl/configs/inference_virtual_soft_bb.yaml

If you want to test a trained model, you can uncomment #ckpt_path: "checkpoints/maskedln-atom-256-pocketloss-dp002/epoch=3-step=9668.ckpt"
in the config itself.

The default level is 'atom' means that each sample contains up to 1200 atoms, that can from from common residues. 
You can set in the config the level (the trained model was made on atom level)

during inference, each batch contains, for both target and source:
{key}_embedding: The Scannet embedding features per atom/residue
{key}_frames: a 3x4 matrix represents the frame
{key}_residue_indices: residue index per residue
{key}_sequence_indices_atom: residue index per atom
{key}_mask

same for the pocket data:
{key}_pocket_embedding
{key}_pocket_frames
{key}_pocket_mask

The batch also contains 
'metadata', which has all protein information including the name, chain etc.
In addition, the Ground Truth also can be accessed with 'gt_R' and 'gt_T
