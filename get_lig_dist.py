from iotbx.pdb import hierarchy
from giant.maths.geometry import is_within
import pandas as pd
import os
import sys
import functools
import re
import pickle


def match_residues(row):

    nearby_residue = row["nearby_residues"]

    if nearby_residue is None:
        return pd.Series((None, None, None, None, None))

    allocated_nearby_residues = set()
    allocated = set()

    for residue in nearby_residue:
        lig_chain = residue[0]
        lig_resid = int(residue[1])
        nearby_res_chain = residue[2]
        nearby_res = residue[3]
        nearby_res_num = int(residue[4])

        for col_num in [1, 2, 3, 4]:

            res_chain = row["residue_chain_" + str(col_num)]
            residue_1_chain = row["residue_1_chain_" + str(col_num)]

            allocated_res, num = split_numeric(row["residue_" + str(col_num)])

            if allocated_res is None or num is None:
                continue
            allocated_res = allocated_res.upper()
            num = int(num)

            res_1, num_1 = split_numeric(row["residue_1_" + str(col_num)])
            if res_1 is not None or num_1 is not None:
                num_1 = int(num_1)
                res_1 = res_1.upper()

            l_chain = row["chain_" + str(col_num)]

            if lig_chain == l_chain:
                if nearby_res_chain == res_chain:
                    if allocated_res == nearby_res:
                        if num == nearby_res_num:
                            allocated_nearby_residues.add(residue)
                            allocated.add((col_num, 0, residue))

                elif (
                    nearby_res_chain == residue_1_chain
                    and res_1 == nearby_res
                    and num_1 == nearby_res_num
                ):

                    allocated_nearby_residues.add(residue)
                    allocated.add((col_num, 1, residue))

    unallocated_nearby_residue = nearby_residue.difference(allocated_nearby_residues)

    # to not alter in loop
    allocated_copy = allocated.copy()

    # residues that match up those that match by ligand
    for residue in unallocated_nearby_residue:

        lig_chain = residue[0]
        lig_resid = int(residue[1])

        for allocated_res in allocated:
            col_num = allocated_res[0]
            place = allocated_res[1]
            l_chain = allocated_res[2][0]
            l_resid = int(allocated_res[2][1])

            if lig_chain == l_chain and lig_resid == l_resid:
                allocated_nearby_residues.add(residue)
                allocated_copy.add((col_num, place, residue))

    # to deal with any residues that remain after matching based on ligand
    unallocated_remaining = nearby_residue.difference(allocated_nearby_residues)

    # to return 4 objects
    allocated_1 = set()
    allocated_2 = set()
    allocated_3 = set()
    allocated_4 = set()

    for allocated_residue in allocated_copy:
        if allocated_residue[0] == 1:
            allocated_1.add(allocated_residue[2])
        if allocated_residue[0] == 2:
            allocated_2.add(allocated_residue[2])
        if allocated_residue[0] == 3:
            allocated_3.add(allocated_residue[2])
        if allocated_residue[0] == 4:
            allocated_4.add(allocated_residue[2])

    # https://apassionatechie.wordpress.com/2017/12/27/create-multiple-pandas-dataframe-columns-from-applying-a-function-with-multiple-returns/
    return pd.Series(
        (allocated_1, allocated_2, allocated_3, allocated_4, unallocated_remaining)
    )


def is_set_with_items(item):
    if item is None:
        return False
    if item == set([]):
        return False
    else:
        return True


def split_numeric(string):
    if not isinstance(string, basestring):
        return None, None

    r = re.compile("([a-zA-Z]+)([0-9]+)")
    return r.match(string).groups()


def get_lig_residues():
    """From nearby residues get ligand"""
    ligs = set()
    for residue in nearby_res_example:
        ligs.add((residue[0], residue[1]))

    return ligs


def drop_left(x):
    if pd.notnull(x):
        if len(str.split(str(x))) != 0:
            return str.split(str(x))[-1]
        else:
            return None
    else:
        return None


def split_residue_info_chain(x, side_of_and):
    side_of_and = 0 if side_of_and == "left" else 1

    if pd.notnull(x):
        if "and" in x:
            return x.split("and")[side_of_and].split(" ")[side_of_and + 1]
        elif x == " ":
            return None
        # only single residue
        elif len(x.split(" ")) > 1:
            return x.split(" ")[1]
        else:
            return None
    else:
        return None


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def split_residue_info(x, side_of_and):
    side_of_and = 0 if side_of_and == "left" else 1
    if pd.notnull(x):
        if "and" in x:
            res = x.split("and")[side_of_and].split(" ")[side_of_and + 2]
            if hasNumbers(res):
                return res
            else:
                return res + x.split("and")[side_of_and].split(" ")[side_of_and + 3]

        elif len(x.split(" ")) == 1 or x == " ":
            return None
        else:
            if side_of_and == 0:
                return x.split(" ")[2]
            else:
                return None
    else:
        return None


def split_residue_info_left(x):
    return split_residue_info(x, side_of_and="left")


def split_residue_info_right(x):
    return split_residue_info(x, side_of_and="right")


def split_residue_info_left_chain(x):
    return split_residue_info_chain(x, side_of_and="left")


def split_residue_info_right_chain(x):
    return split_residue_info_chain(x, side_of_and="right")


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
    lig_ags = [ag for ag in prot_h.atom_groups() if ag.resname == "LIG"]

    # all non ligand atom groups
    not_lig_ags = [ag for ag in prot_h.atom_groups() if ag.resname != "LIG"]

    # atom_groups_near_lig
    ag_set = set()
    for lig in lig_ags:
        lig_chain = lig.parent().parent().id
        lig_resseq = lig.parent().resseq

        for ag in not_lig_ags:
            ag_chain = ag.parent().parent().id

            if is_within(cutoff, ag.atoms().extract_xyz(), lig.atoms().extract_xyz()):

                ag_set.add(
                    (lig_chain, lig_resseq, ag_chain, ag.resname, ag.parent().resseq,)
                )

    return ag_set

def read_occupancy_b(pdb_path, selection):
    """Extract occupancy and B factor of pdb given selection"""

    # Read in single PDB file
    pdb_in = hierarchy.input(file_name=pdb_path)
    sel_cache = pdb_in.hierarchy.atom_selection_cache()
    sel = sel_cache.selection(selection)
    sel_hierarchy = pdb_in.hierarchy.select(sel)

    occ_b = []
    # Get occupancy & B factor of ligand
    for model in sel_hierarchy.models():
        for chain in model.chains():
            for rg in chain.residue_groups():
                for ag in rg.atom_groups():
                    for atom in ag.atoms():
                        occ_b.append([ag.resname, rg.resseq ,ag.altloc, atom.name, atom.occ, atom.b])

    occ_b_df = pd.DataFrame(occ_b, columns=["Residue", "resseq", "altloc", "Atom", "Occupancy", "B_factor"])

    return occ_b_df


def res_to_selection(res_tuple):
    """Convert tuple to iotbx selection"""
    res_chain = res_tuple[2]
    res_name = res_tuple[3]
    res_num = res_tuple[4]

    return "(chain " + res_chain + " and resid " + str(int(res_num)) + ")"
    #return "(chain " + res_chain + " and name " + res_name + " and resid " + str(int(res_num)) + ")"


def residue_list_to_selection(allocated):
    "Iotbx selection string based on itreable of residue objects"
    sel_list = []
    for residue_tuple in allocated:
        sel_list.append(res_to_selection(residue_tuple))

    #return sel_list[0]
    return ' or '.join(sel_list)

def allocated_occ_b_from_row(row):

    allocated_1_occ_b_df = occ_b_from_residues(row['allocated_1'], row['refine_pdb'])
    if allocated_1_occ_b_df is not None:
        allocated_1_mean = allocated_1_occ_b_df['B_factor'].mean()
        allocated_1_std_dev = allocated_1_occ_b_df['B_factor'].std()
    else:
        allocated_1_mean = None
        allocated_1_std_dev = None

    allocated_2_occ_b_df = occ_b_from_residues(row['allocated_2'], row['refine_pdb'])
    if allocated_2_occ_b_df is not None:
        allocated_2_mean = allocated_2_occ_b_df['B_factor'].mean()
        allocated_2_std_dev = allocated_2_occ_b_df['B_factor'].std()
    else:
        allocated_2_mean = None
        allocated_2_std_dev = None

    allocated_3_occ_b_df = occ_b_from_residues(row['allocated_3'], row['refine_pdb'])
    if allocated_3_occ_b_df is not None:
        allocated_3_mean = allocated_3_occ_b_df['B_factor'].mean()
        allocated_3_std_dev = allocated_3_occ_b_df['B_factor'].std()
    else:
        allocated_3_mean = None
        allocated_3_std_dev = None

    allocated_4_occ_b_df = occ_b_from_residues(row['allocated_4'], row['refine_pdb'])
    if allocated_4_occ_b_df is not None:
        allocated_4_mean = allocated_4_occ_b_df['B_factor'].mean()
        allocated_4_std_dev = allocated_4_occ_b_df['B_factor'].std()
    else:
        allocated_4_mean = None
        allocated_4_std_dev = None

    return pd.Series((allocated_1_occ_b_df, allocated_1_mean, allocated_1_std_dev,
                     allocated_2_occ_b_df, allocated_2_mean, allocated_2_std_dev,
                     allocated_3_occ_b_df, allocated_3_mean, allocated_3_std_dev,
                     allocated_4_occ_b_df, allocated_4_mean, allocated_4_std_dev))

def occ_b_from_residues(allocated_list, pdb_path):
    """From list of tuple of residues given occupancy B value df"""
    if allocated_list is None:
        return None
    selection_list = residue_list_to_selection(allocated_list)
    return read_occupancy_b(pdb_path, selection_list)



if __name__ == "__main__":

    nudt5_master_tidied_path = (
        "/dls/science/groups/i04-1/elliot-dev/"
        "NUDT5_occupancy/NUDT5_master_tidied_modified.csv"
    )

    pickle_path = (
        "/dls/science/groups/i04-1/elliot-dev/"
        "NUDT5_occupancy/after_allocated_residues_26_05_20.pkl"
    )

    nudt5_master_tidied_df = pd.read_csv(nudt5_master_tidied_path)

    data_dir = (
        "/dls/labxchem/data/2018/lb18145-71/processing/analysis/occupancy_elliot/"
    )

    nudt5_master_tidied_df["refine_pdb"] = (
        data_dir + nudt5_master_tidied_df["crystal_id"] + "/refine.pdb"
    )



    if not os.path.isfile(pickle_path):
        # add boolean column to check refine_pdb exists
        nudt5_master_tidied_df["refine_pdb_exists"] = nudt5_master_tidied_df[
            "refine_pdb"
        ].apply(os.path.exists)

        # produce usable columns from site information
        for col in nudt5_master_tidied_df.columns.values:
            if "in pdb file" in col.lower():
                chain_col = "chain_" + col.split()[-1]
                nudt5_master_tidied_df[chain_col] = nudt5_master_tidied_df[col].apply(
                    drop_left
                )

            if "position in the crystal" in col.lower():

                # column names
                res_chain = "residue_chain_" + col.split()[-1]
                res_chain_1 = "residue_1_chain_" + col.split()[-1]
                res = "residue_" + col.split()[-1]
                res_1 = "residue_1_" + col.split()[-1]

                # get chain of first residue listed
                nudt5_master_tidied_df[res_chain] = nudt5_master_tidied_df[col].apply(
                    split_residue_info_left_chain
                )
                # get chain of second residue listed
                nudt5_master_tidied_df[res_chain_1] = nudt5_master_tidied_df[col].apply(
                    split_residue_info_right_chain
                )
                # get residue name and number of first residue listed
                nudt5_master_tidied_df[res] = nudt5_master_tidied_df[col].apply(
                    split_residue_info_left
                )
                # get residue name and number of second residue listed
                nudt5_master_tidied_df[res_1] = nudt5_master_tidied_df[col].apply(
                    split_residue_info_right
                )

        # get residues close to ligands
        nudt5_master_tidied_df["nearby_residues"] = nudt5_master_tidied_df[
            "refine_pdb"
        ].apply(within_5)

        # match residues that are close to named site
        nudt5_master_tidied_df[
            ["allocated_1", "allocated_2", "allocated_3", "allocated_4", "unallocated"]
        ] = nudt5_master_tidied_df.apply(match_residues, axis=1)

        # get occupancy and b factor df, mean and std deviation
        # for nearby residues in allocated groups
        nudt5_master_tidied_df[
            ["allocated_1_occ_b_df","allocated_1_occ_b_mean", "allocated_1_occ_b_std_dev",
             "allocated_2_occ_b_df","allocated_2_occ_b_mean", "allocated_2_occ_b_std_dev",
             "allocated_3_occ_b_df","allocated_3_occ_b_mean", "allocated_3_occ_b_std_dev",
             "allocated_4_occ_b_df","allocated_4_occ_b_mean", "allocated_4_occ_b_std_dev"]
        ] = nudt5_master_tidied_df.apply(allocated_occ_b_from_row, axis=1)

        with open(pickle_path, 'wb') as pickle_output:
            pickle.dump(nudt5_master_tidied_df, pickle_output)
    else:
        with open(pickle_path, 'r') as pickle_output:
            nudt5_master_tidied_df = pickle.load(pickle_output)

