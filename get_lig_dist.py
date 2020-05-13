from iotbx.pdb import hierarchy
from giant.maths.geometry import is_within
import pandas as pd
import os
import functools

def within_5(pdb):
    return residues_near_ligs(pdb, 5)

def residues_near_ligs(pdb, cutoff):
    """
    Get residues within angstrom cutoff of LIG.

    Parameters
    ----------
    pdb: str, path
        path to pdb file
    cutoff: float
        angstrom cutoff for distance

    Returns
    -------
    ag_set: set
        a set of dicts
        ligand chain: str
        ligand resseq: str
        protein chain: str
        protein resname: str
        protein resseq: str
    """
    if not os.path.exists(pdb):
        return None
    # Load the structure
    prot_i = hierarchy.input(pdb)
    prot_h = prot_i.hierarchy

    # Extract the ligands from the hierarchy
    lig_ags = [ag for ag in prot_h.atom_groups() if ag.resname == 'LIG']

    # all non ligand atom groups
    not_lig_ags = [ag for ag in prot_h.atom_groups() if ag.resname != 'LIG']

    # atom_groups_near_lig
    ag_set = set()
    for lig in lig_ags:
        lig_chain = lig.parent().parent().id
        lig_resseq = lig.parent().resseq

        for ag in not_lig_ags:
            ag_chain = ag.parent().parent().id

            if is_within(cutoff,
                         ag.atoms().extract_xyz(),
                         lig.atoms().extract_xyz()):

                ag_set.add((lig_chain,
                            lig_resseq,
                            ag_chain,
                            ag.resname,
                            ag.parent().resseq,
                            ))

    return ag_set


if __name__ == "__main__":

    nudt5_master_tidied_path = '/dls/science/groups/i04-1/elliot-dev/' \
                               'NUDT5_occupancy/NUDT5_master_tidied.csv'

    nudt5_master_tidied_df = pd.read_csv(nudt5_master_tidied_path)

    data_dir = '/dls/labxchem/data/2018/lb18145-71/processing/analysis/occupancy_elliot/'

    nudt5_master_tidied_df['refine_pdb'] = data_dir + nudt5_master_tidied_df['crystal_id'] + '/refine.pdb'
    nudt5_master_tidied_df['refine_pdb_exists'] = nudt5_master_tidied_df['refine_pdb'].apply(os.path.exists)

    refined_non_exist_df = nudt5_master_tidied_df.query('refine_pdb_exists==False'
                                                        ' and Refined=="yes"', engine='python') # numexpr not in ccp4-python
    print(refined_non_exist_df)
    # get residues close to ligands
    #nudt5_master_tidied_df['nearby_residues'] =  nudt5_master_tidied_df['refine_pdb'].apply(within_5)
