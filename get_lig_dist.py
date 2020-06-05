from iotbx.pdb import hierarchy
from giant.maths.geometry import is_within
import pandas as pd
import os
import sys
import functools
import re
import pickle
import numpy as np

import matplotlib

matplotlib.use("agg")

from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt


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
                    if allocated_res == nearby_res and num == nearby_res_num:
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

    lig_set = set()
    # residues that match up those that match by ligand
    for residue in unallocated_nearby_residue:

        lig_chain = residue[0]
        lig_resid = int(residue[1])

        for allocated_res in allocated:
            col_num = allocated_res[0]
            l_chain = allocated_res[2][0]
            l_resid = int(allocated_res[2][1])

            if lig_chain == l_chain and lig_resid == l_resid:
                allocated_nearby_residues.add(residue)
                place = allocated_res[1]
                allocated_copy.add((col_num, place, residue))
                lig_set.add((col_num, lig_chain, lig_resid))

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

    # to return allocated ligands
    lig_1 = set()
    lig_2 = set()
    lig_3 = set()
    lig_4 = set()

    for lig in lig_set:
        if lig[0] == 1:
            lig_1.add((lig[1], lig[2]))
        if lig[0] == 2:
            lig_2.add((lig[1], lig[2]))
        if lig[0] == 3:
            lig_3.add((lig[1], lig[2]))
        if lig[0] == 4:
            lig_4.add((lig[1], lig[2]))

    # https://apassionatechie.wordpress.com/2017/12/27/create-multiple-pandas-dataframe-columns-from-applying-a-function-with-multiple-returns/
    return pd.Series(
        (
            allocated_1,
            allocated_2,
            allocated_3,
            allocated_4,
            lig_1,
            lig_2,
            lig_3,
            lig_4,
            unallocated_remaining,
        )
    )


def is_set_with_items(item):
    if item is None:
        return False
    return item != set([])


def split_numeric(string):
    if not isinstance(string, basestring):
        return None, None

    r = re.compile("([a-zA-Z]+)([0-9]+)")
    return r.match(string).groups()


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
    """Does a string have numbers in it"""
    return any(char.isdigit() for char in inputString)


def split_residue_info(x, side_of_and):
    if not pd.notnull(x):
        return None
    side_of_and = 0 if side_of_and == "left" else 1
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

    if not os.path.exists(pdb_path):
        return None

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
                        occ_b.append(
                            [
                                ag.resname,
                                rg.resseq,
                                ag.altloc,
                                atom.name,
                                atom.occ,
                                atom.b,
                            ]
                        )

    return pd.DataFrame(
        occ_b,
        columns=[
            "Residue",
            "resseq",
            "altloc",
            "Atom",
            "Occupancy",
            "B_factor",
        ],
    )


def res_to_selection(res_tuple):
    """Convert tuple to iotbx selection"""
    return "(chain " + res_tuple[2] + " and resid " + str(int(res_tuple[4])) + ")"


def residue_list_to_selection(allocated):
    """
    Iotbx selection string based on itreable of residue objects

    """
    sel_list = [res_to_selection(residue_tuple) for residue_tuple in allocated]
    return " or ".join(sel_list)


def allocated_occ_b_from_row(row):
    allocated_1_occ_b_df = occ_b_from_residues(row["allocated_1"], row["refine_pdb"])
    if allocated_1_occ_b_df is not None:
        allocated_1_mean = allocated_1_occ_b_df["B_factor"].mean()
        allocated_1_std_dev = allocated_1_occ_b_df["B_factor"].std()
    else:
        allocated_1_mean = None
        allocated_1_std_dev = None

    allocated_2_occ_b_df = occ_b_from_residues(row["allocated_2"], row["refine_pdb"])
    if allocated_2_occ_b_df is not None:
        allocated_2_mean = allocated_2_occ_b_df["B_factor"].mean()
        allocated_2_std_dev = allocated_2_occ_b_df["B_factor"].std()
    else:
        allocated_2_mean = None
        allocated_2_std_dev = None

    allocated_3_occ_b_df = occ_b_from_residues(row["allocated_3"], row["refine_pdb"])
    if allocated_3_occ_b_df is not None:
        allocated_3_mean = allocated_3_occ_b_df["B_factor"].mean()
        allocated_3_std_dev = allocated_3_occ_b_df["B_factor"].std()
    else:
        allocated_3_mean = None
        allocated_3_std_dev = None

    allocated_4_occ_b_df = occ_b_from_residues(row["allocated_4"], row["refine_pdb"])
    if allocated_4_occ_b_df is not None:
        allocated_4_mean = allocated_4_occ_b_df["B_factor"].mean()
        allocated_4_std_dev = allocated_4_occ_b_df["B_factor"].std()
    else:
        allocated_4_mean = None
        allocated_4_std_dev = None

    return pd.Series(
        (
            allocated_1_occ_b_df,
            allocated_1_mean,
            allocated_1_std_dev,
            allocated_2_occ_b_df,
            allocated_2_mean,
            allocated_2_std_dev,
            allocated_3_occ_b_df,
            allocated_3_mean,
            allocated_3_std_dev,
            allocated_4_occ_b_df,
            allocated_4_mean,
            allocated_4_std_dev,
        )
    )


def occ_b_from_residues(allocated_list, pdb_path):
    """From list of tuple of residues given occupancy B value df"""
    if allocated_list is None:
        return None
    selection_list = residue_list_to_selection(allocated_list)
    return read_occupancy_b(pdb_path, selection_list)


def write_b_params(nearby_residues):
    """String of b factor parameters"""

    if nearby_residues is None:
        return None

    phenix_params_list = [
        "refinement.refine.strategy=individual_adp",
        'refinement.refine.adp.individual.isotropic="not ('
        + residue_list_to_selection(nearby_residues)
        + ')"',
        "refinement.refine.strategy=occupancies",
        "refinement.main.number_of_macro_cycles = 20",
    ]

    return "\n".join(phenix_params_list)


def set_b_factor_pdb(row, rerun=False):
    pdb_in_path = row["refine_pdb"]
    pdb_out_path = row["site_b_factor_path"]

    # don't do if output file exists, or refine pdb doesn't exist, unless rerun flag set
    if (not os.path.exists(pdb_out_path) and os.path.exists(pdb_in_path)) or rerun:
        pdb_in = hierarchy.input(file_name=pdb_in_path)
        sites = row["sites"]

        # for each site listed in site set the B factor of that site
        # to the mean b factor of that site
        for site in sites:
            allocated_col = site[0]
            sel = residue_list_to_selection(row[site[0]])
            lig_col = allocated_col.replace("allocated", "lig")
            if len(row[lig_col]) > 1:
                raise ValueError("More than one allocated residue")
            else:
                chain = list(row[lig_col])[0][0]
                lig_num = list(row[lig_col])[0][1]

            lig_sel = "(chain " + chain + " and resid " + str(int(lig_num)) + ")"
            sel = sel + " or " + lig_sel

            b_fac = site[1]
            pdb_in = set_b_factor(pdb_in=pdb_in, sel=sel, b_fac=b_fac)

        if len(sites) != 0:
            with open(pdb_out_path, "w") as out_pdb_file:
                out_pdb_file.write(
                    pdb_in.hierarchy.as_pdb_string(
                        crystal_symmetry=pdb_in.input.crystal_symmetry()
                    )
                )


def set_b_factor(pdb_in, sel, b_fac):
    # define a selection
    sel_cache = pdb_in.hierarchy.atom_selection_cache()
    sel = sel_cache.selection(sel)
    sel_hierarchy = pdb_in.hierarchy.select(sel)

    # set b factor
    for model in sel_hierarchy.models():
        for chain in model.chains():
            for rg in chain.residue_groups():
                for ag in rg.atom_groups():
                    for atom in ag.atoms():
                        atom.set_b(b_fac)

    # return object to hold a hierarchy object
    return pdb_in


def allocate_sites(row, site_b):
    sites = []
    for site, b_fac in site_b.iteritems():
        if site in str(row["Position in the crystal 1"]):
            sites.append(("allocated_1", b_fac))
        if site in str(row["Position in the crystal 2"]):
            sites.append(("allocated_2", b_fac))
        if site in str(row["Position in the crystal 3"]):
            sites.append(("allocated_3", b_fac))
        if site in str(row["Position in the crystal 4"]):
            sites.append(("allocated_4", b_fac))

    return sites


def generate_restraints(row):
    """Run giant.make_restriants on refine.pdb if it hasn't been run"""

    if not os.path.exists(
        str(row["refine_pdb"]).replace(
            "refine.pdb", "multi-state-restraints.phenix.params"
        )
    ) and os.path.exists(row["refine_pdb"]):
        os.system(
            "cd "
            + str(row["refine_pdb"]).replace("/refine.pdb", "")
            + ";"
            + "giant.make_restraints refine.pdb"
        )


def combine_restriants(row):
    phenix_restriants = row["phenix_restraints"]
    combined_restriant = phenix_restriants.replace("multi-state", "site_b_factor_fixed")
    if os.path.exists(phenix_restriants):
        existing = ""
        with open(phenix_restriants, "r") as pr:
            existing += pr.read()

        with open(combined_restriant, "w") as cr:
            cr.write(row["phenix_param_string"] + "\n")
            cr.write(existing)

        return combined_restriant
    else:
        return None


def get_cif(row):
    folder = row["refine_pdb"].replace("refine.pdb", "")
    cifs = []
    for file_path in os.listdir(folder):
        if (
            os.path.isfile(os.path.join(folder, file_path))
            and file_path.endswith(".cif")
            and "data_template" not in file_path
        ):
            cifs.append(file_path)

    if len(cifs) == 1:
        return cifs[0]
    else:
        cmpd = row["compound_id"]

        for cif in cifs:
            if cmpd in cif:
                return cif


def refine_b_factor_site_fixed(row):
    pdb = row["site_b_factor_path"]
    folder = row["refine_pdb"].replace("refine.pdb", "")

    if not os.path.isdir(folder):
        return None

    restraints = row["combined_phenix_restraints"]
    mtz = str(row["refine_pdb"]).replace("refine.pdb", row["crystal_id"] + ".free.mtz")
    csh_file = str(row["refine_pdb"]).replace(
        "refine.pdb", "site_b_factor_phenix_refine.csh"
    )
    cif = get_cif(row)
    out_pdb = str(row["refine_pdb"]).replace("refine.pdb", "site_fixed_b.pdb")

    if (
        os.path.exists(pdb)
        and os.path.exists(restraints)
        and os.path.exists(mtz)
        and not os.path.exists(out_pdb)
    ):
        with open(csh_file, "w") as csh_f:
            csh_f.write(
                "source /dls/science/groups/i04-1/software/i04-1/XChemExplorer/XChemExplorer/setup-scripts/pandda.setup-sh\n"
            )
            csh_f.write("cd " + folder + "\n")
            csh_f.write("module load phenix\n")
            csh_f.write(
                "giant.quick_refine input.pdb="
                + pdb
                + " mtz="
                + mtz
                + " cif="
                + cif
                + " program=phenix"
                + " params="
                + restraints
                + " dir_prefix='site_fixed_b'"
                + " out_prefix='site_fixed_b'"
                + " link_prefix='site_fixed_b'"
            )

    os.system("qsub " + csh_file)


def get_site_occ_conc(row, sites):
    concentration = row["concentration"]
    cmpd = row["compound_id"]

    if row["site_fixed_mean_occ_lig"] is None:
        # https://stackoverflow.com/questions/17839973/constructing-pandas-dataframe-from-values-in-variables-gives-valueerror-if-usi
        return None

    site_info = []

    for site in sites:
        for site_lig in row["site_fixed_mean_occ_lig"]:
            if site == site_lig[0]:
                site_info.append(
                    {
                        "concentration": concentration,
                        "compound_id": cmpd,
                        "site": site,
                        "occ": site_lig[1],
                    }
                )

    return site_info


def get_mean_occ(row):
    """ Get the mean occupancy of LIG residues based on the allocated site

    Takes into account the number of altlocs

    Matches to string describing site (Position in crystal columns) to site
    """
    site_fixed_df = row["site_fixed_lig"]

    if not isinstance(site_fixed_df, pd.DataFrame):
        return None

    site_occs = []
    for site in row["sites"]:
        col_num = site[0][-1]
        lig_chain = list(row["lig_" + str(col_num)])[0][0]
        lig_resseq = list(row["lig_" + str(col_num)])[0][1]

        site_df = site_fixed_df[
            pd.to_numeric(site_fixed_df["resseq"]) == int(lig_resseq)
        ]
        num_altlocs = len(site_df["altloc"].unique())
        site_mean_occ = site_df["Occupancy"].mean() * num_altlocs

        site_string = row["Position in the crystal " + col_num]
        site_occs.append((site_string, site_mean_occ))

    return site_occs


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

    rerun = False

    if not os.path.isfile(pickle_path) or rerun:
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
            [
                "allocated_1",
                "allocated_2",
                "allocated_3",
                "allocated_4",
                "lig_1",
                "lig_2",
                "lig_3",
                "lig_4",
                "unallocated",
            ]
        ] = nudt5_master_tidied_df.apply(match_residues, axis=1)

        # get occupancy and b factor df, mean and std deviation
        # for nearby residues in allocated groups
        nudt5_master_tidied_df[
            [
                "allocated_1_occ_b_df",
                "allocated_1_occ_b_mean",
                "allocated_1_occ_b_std_dev",
                "allocated_2_occ_b_df",
                "allocated_2_occ_b_mean",
                "allocated_2_occ_b_std_dev",
                "allocated_3_occ_b_df",
                "allocated_3_occ_b_mean",
                "allocated_3_occ_b_std_dev",
                "allocated_4_occ_b_df",
                "allocated_4_occ_b_mean",
                "allocated_4_occ_b_std_dev",
            ]
        ] = nudt5_master_tidied_df.apply(allocated_occ_b_from_row, axis=1)

        with open(pickle_path, "wb") as pickle_output:
            pickle.dump(nudt5_master_tidied_df, pickle_output)
    else:
        with open(pickle_path, "r") as pickle_output:
            nudt5_master_tidied_df = pickle.load(pickle_output)

    # get a string for fixing the b factor
    nudt5_master_tidied_df["phenix_param_string"] = nudt5_master_tidied_df[
        "nearby_residues"
    ].apply(write_b_params)

    # get a list of sites
    sites = set()
    sites.update(nudt5_master_tidied_df["Position in the crystal 1"].dropna().unique())
    sites.update(nudt5_master_tidied_df["Position in the crystal 2"].dropna().unique())
    sites.update(nudt5_master_tidied_df["Position in the crystal 3"].dropna().unique())
    sites.update(nudt5_master_tidied_df["Position in the crystal 4"].dropna().unique())

    # https://stackoverflow.com/questions/37147735/remove-nan-value-from-a-set
    sites.remove("check!!!!")
    sites.remove(" ")

    # get the mean b factor of a site
    site_b = {}
    for site in sites:
        site_b_factor_1 = nudt5_master_tidied_df[
            nudt5_master_tidied_df["Position in the crystal 1"] == site
        ]["allocated_1_occ_b_mean"]

        site_b_factor_2 = nudt5_master_tidied_df[
            nudt5_master_tidied_df["Position in the crystal 2"] == site
        ]["allocated_2_occ_b_mean"]

        site_b_factor_3 = nudt5_master_tidied_df[
            nudt5_master_tidied_df["Position in the crystal 3"] == site
        ]["allocated_3_occ_b_mean"]

        site_b_factor_4 = nudt5_master_tidied_df[
            nudt5_master_tidied_df["Position in the crystal 4"] == site
        ]["allocated_4_occ_b_mean"]

        mean_b_site_list = []
        if len(site_b_factor_1) != 0:
            mean_b_site_list.append(site_b_factor_1.mean())

        if len(site_b_factor_2) != 0:
            mean_b_site_list.append(site_b_factor_2.mean())

        if len(site_b_factor_3) != 0:
            mean_b_site_list.append(site_b_factor_3.mean())

        if len(site_b_factor_4) != 0:
            mean_b_site_list.append(site_b_factor_4.mean())

        mean_b = np.mean(mean_b_site_list)

        site_b[site] = mean_b

    nudt5_master_tidied_df["site_b_factor_path"] = nudt5_master_tidied_df[
        "refine_pdb"
    ].str.replace(pat="refine.pdb", repl="site_b_factor.pdb")

    # pandas 0.17.1: can pass arguments in apply:
    # https://pandas.pydata.org/pandas-docs/version/0.17/generated/pandas.DataFrame.apply.html?highlight=apply#pandas.DataFrame.apply
    nudt5_master_tidied_df["sites"] = nudt5_master_tidied_df.apply(
        allocate_sites, site_b=site_b, axis=1
    )

    # make a pdb file with fixed b factors according to the average for that site,
    # file is that in nudt5_master_tidied_df['site_b_factor_path']
    nudt5_master_tidied_df.apply(set_b_factor_pdb, axis=1)

    # run giant.make_restraints on all cases where refine pdb exists
    nudt5_master_tidied_df.apply(generate_restraints, axis=1)

    nudt5_master_tidied_df["phenix_restraints"] = nudt5_master_tidied_df[
        "refine_pdb"
    ].str.replace("refine.pdb", "multi-state-restraints.phenix.params")

    nudt5_master_tidied_df["combined_phenix_restraints"] = nudt5_master_tidied_df.apply(
        combine_restriants, axis=1
    )

    # phenix refine using giant quick_refine
    # TODO ~ 20 folders which are not refining to investigate
    # nudt5_master_tidied_df.apply(refine_b_factor_site_fixed, axis=1)

    # TODO Run this only when files exists?
    nudt5_master_tidied_df["site_fixed_lig"] = nudt5_master_tidied_df[
        "site_b_factor_path"
    ].apply(read_occupancy_b, selection="resname LIG")

    nudt5_master_tidied_df["site_fixed_mean_occ_lig"] = nudt5_master_tidied_df.apply(
        get_mean_occ, axis=1
    )

    # This is inefficent but a better solution is not forthcoming
    series = nudt5_master_tidied_df.apply(
        get_site_occ_conc, sites=sites, axis=1
    ).dropna()
    list_df = []
    for item in series:
        list_df.append(pd.DataFrame.from_records(item))

    site_fixed_df = pd.concat(list_df, ignore_index=True)

    plot = site_fixed_df.plot(x="occ", y="concentration", kind="scatter")
    fig = plot.get_figure()
    fig.savefig("occ_conc.png")

    for compound in site_fixed_df["compound_id"].unique():
        compound_df = site_fixed_df[site_fixed_df["compound_id"] == compound]
        for site in compound_df["site"].unique():
            site_compound_df = compound_df[compound_df["site"] == site]
            plot = site_compound_df.plot(x="occ", y="concentration", kind="scatter")
            fig = plot.get_figure()
            fig.savefig("site_cmpd/occ_conc_" + site + "_" + compound + ".png")

    for site in site_fixed_df["site"].unique():
        site_df = site_fixed_df[site_fixed_df["site"] == site]
        plot = site_df.plot(x="occ", y="concentration", kind="scatter")

        x = site_df["occ"]
        y = site_df["concentration"]
        plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
        fig = plot.get_figure()
        fig.savefig("site/occ_conc_" + site + ".png")

    # print(pd.DataFrame.from_dict(test_dict_drop_none, orient='index'))

    # nudt5_master_tidied_df.to_csv(
    #     "/dls/science/groups/i04-1/elliot-dev/NUDT5_occupancy/testB.csv"
    # )
