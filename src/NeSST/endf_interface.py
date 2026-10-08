import json
from dataclasses import dataclass

import endf
import numpy as np

from NeSST.constants import *
from NeSST.cross_sections import NeSST_DDX, NeSST_SDX

# Some defs
# See https://t2.lanl.gov/nis/endf/mts.html

MF_XSECTION = 3
MF_ENERGYANGLEDISTS = 4
MF_PRODENERGYANGLEDISTS = 6

MT_TOTXSECTION = 1
MT_ELXSECTION = 2
MT_N2NXSECTION = 16
MT_INELXSECTION_START = 51
MT_NPXSECTION = 103


@dataclass
class ENDFManifest:
    total: bool
    elastic: bool
    inelastic: bool
    n2n: bool
    np: bool = False



def retrieve_ENDF_data(json_file):
    mainfest = convert_json_to_manifest(data_dir + json_file)
    ENDF_data = retrieve_ENDF_data_from_manifest(mainfest)
    return ENDF_data


def convert_json_to_manifest(json_file):
    with open(json_file, "r") as js:
        json_dict = json.load(js)
    manifest_dict = {}
    for filename, file_manifest in json_dict.items():
        manifest_dict[filename] = ENDFManifest(
            total=file_manifest.get("total", False),
            elastic=file_manifest.get("elastic", False),
            inelastic=file_manifest.get("inelastic", False),
            n2n=file_manifest.get("n2n", False),
            np=file_manifest.get("np", False),
        )
    return manifest_dict


def retrieve_ENDF_data_from_manifest(manifest):
    return_dict = {"interactions": ENDFManifest(total=False, elastic=False, inelastic=False, n2n=False)}

    for filename, file_manifest in manifest.items():
        mat = endf.Material(ENDF_dir + filename)

        if file_manifest.total:
            if return_dict["interactions"].total:
                print(f"{filename}: Warning, loading total cross section data from multiple files...")
            else:
                return_dict["interactions"].total = True

            total_dataset = mat.section_data[MF_XSECTION, MT_TOTXSECTION]
            return_dict["A"] = total_dataset["AWR"]
            total_table = total_dataset["sigma"]
            total_E, total_sigma = total_table.x, total_table.y
            return_dict["total_xsec"] = {"E": total_E, "sig": total_sigma}

        if file_manifest.elastic:
            if return_dict["interactions"].elastic:
                print(f"{filename}: Warning, loading elastic cross section data from multiple files...")
            else:
                return_dict["interactions"].elastic = True

            elastic_table = mat.section_data[MF_XSECTION, MT_ELXSECTION]["sigma"]
            elastic_E, elastic_sigma = elastic_table.x, elastic_table.y
            return_dict["elastic_xsec"] = {"E": elastic_E, "sig": elastic_sigma}

            LTT = mat.section_data[MF_ENERGYANGLEDISTS, MT_ELXSECTION]["LTT"]

            if LTT == 0:  # Isotropic
                return_dict["elastic_dxsec"] = {
                    "legendre": True,
                    "E": np.array([elastic_E[0], elastic_E[-1]]),
                    "a_l": np.zeros((2, 1)),
                    "N_l": 1,
                }

            elif LTT == 1 or LTT == 3:  # Legendre
                if LTT == 3:
                    print(f"{filename}: Warning, LTT ({LTT}) for elastic scattering, using Legendre only")
                elastic_table = mat.section_data[MF_ENERGYANGLEDISTS, MT_ELXSECTION]["legendre"]
                elastic_E, elastic_al = elastic_table["E"], convert_to_NeSST_legendre_format(elastic_table["a_l"])
                return_dict["elastic_dxsec"] = {
                    "legendre": True,
                    "E": elastic_E,
                    "a_l": elastic_al,
                    "N_l": elastic_al.shape[1],
                }

            elif LTT == 2:  # Tabulated
                elastic_table = mat.section_data[MF_ENERGYANGLEDISTS, MT_ELXSECTION]["tabulated"]
                elastic_E = elastic_table["E"]
                NE = elastic_E.shape[0]
                elastic_Ein, elastic_mu, elastic_f = [], [], []
                for iE in range(NE):
                    elastic_mu.append(elastic_table["mu"][iE].x)
                    elastic_f.append(elastic_table["mu"][iE].y)
                    Nmu = elastic_table["mu"][iE].x.shape[0]
                    elastic_Ein.append(np.repeat(elastic_E[iE], Nmu))

                points = np.column_stack((np.concatenate(elastic_Ein), np.concatenate(elastic_mu)))
                values = np.concatenate(elastic_f)
                SDX = NeSST_SDX(Ein=elastic_E, points=points, values=values)
                return_dict["elastic_dxsec"] = {"legendre": False, "SDX": SDX}

            else:
                print(f"{filename}: Warning, LTT ({LTT}) for elastic scattering not recognised...")

            # Initialize the interaction manifest check if not already present
            if "np" not in return_dict["interactions"].__dict__:
                return_dict["interactions"].np = False

            # Check for (n,p) reaction flag (defaults to False if missing)
            if getattr(file_manifest, "np", False):
                if return_dict["interactions"].np:
                    print(f"{filename}: Warning, loading (n,p) cross section data from multiple files...")
                else:
                    return_dict["interactions"].np = True

                # 1. Load Cross-Section and Q-value from MF=3, MT=103
                np_table = mat.section_data[MF_XSECTION, MT_NPXSECTION]["sigma"]
                np_E, np_sigma = np_table.x, np_table.y
                return_dict["np_xsec"] = {"E": np_E, "sig": np_sigma}

                # Pull QI (Reaction Q-value) or default to Q
                np_sec3 = mat.section_data[MF_XSECTION, MT_NPXSECTION]
                np_Q = np_sec3.get("QI", np_sec3.get("QM", 0.0))

                # 2. Parse Angular / Energy Distributions from MF=6, MT=103
                if (MF_PRODENERGYANGLEDISTS, MT_NPXSECTION) in mat.section_data:
                    mf6_data = mat.section_data[MF_PRODENERGYANGLEDISTS, MT_NPXSECTION]
                    products = mf6_data.get("products", [])
                    # Target the relevant outgoing recoil product (e.g., Triton ZAP=1003 or Proton ZAP=1001)
                    target_product = None
                    for prod in products:
                        # Prefer Triton (1003) for downscatter calculations; fallback to Proton (1001)
                        if prod.get("ZAP") in [1003, 1001]:
                            target_product = prod
                            if prod.get("ZAP") == 1003:  # Priority on heavy recoil
                                break
                    if target_product is not None:
                        law = target_product.get("LAW")
                        if law == 2:  # Discrete 2-Body Recoil (Legendre in CM)
                            dist = target_product["distribution"]
                            np_E_dist = dist["E"]
                            np_al = convert_to_NeSST_legendre_format(dist["a_l"])

                            return_dict["np_dxsec"] = {
                                "legendre": True,
                                "E": np_E_dist,
                                "a_l": np_al,
                                "N_l": np_al.shape[1],
                                "Q": np_Q,
                                "ZAP": target_product.get("ZAP"),
                                "A_p": target_product.get("AWP"),
                            }

                        elif law == 1:  # Continuum / Tabulated double differential
                            target_E = return_dict.get("elastic_xsec", {}).get("E")
                            if target_E is None and "total_xsec" in return_dict:
                                target_E = return_dict["total_xsec"].get("E")
                            DDX = convert_to_NeSST_LAW1_format(
                                target_product["distribution"],
                                target_Ein=target_E
                            )
                            return_dict["np_dxsec"] = {
                                "legendre": False,
                                "LAW": 1,
                                "DDX": DDX,
                                "Q": np_Q,
                                "ZAP": target_product.get("ZAP"),
                            }
                        elif law == 7:  # Laboratory Angle-Energy Tabulated (mu -> E')
                            DDX = convert_to_NeSST_LAW7_format(target_product["distribution"])
                            return_dict["np_dxsec"] = {
                                "legendre": False,
                                "LAW": 7,
                                "DDX": DDX,
                                "Q": np_Q,
                                "ZAP": target_product.get("ZAP"),
                            }
                        else:
                            print(f"{filename}: LAW = {law} for (n,p) reaction not directly supported...")

                # Fallback to MF=4 if angular data is stored there instead of MF=6
                elif (4, MT_NPXSECTION) in mat.section_data:
                    LTT = mat.section_data[4, MT_NPXSECTION]["LTT"]
                    if LTT == 0:  # Isotropic
                        return_dict["np_dxsec"] = {
                            "legendre": True,
                            "E": np.array([np_E[0], np_E[-1]]),
                            "a_l": np.zeros((2, 1)),
                            "N_l": 1,
                            "Q": np_Q,
                        }
                    elif LTT in [1, 3]:  # Legendre
                        np_leg_table = mat.section_data[4, MT_NPXSECTION]["legendre"]
                        np_E_dist, np_al = np_leg_table["E"], convert_to_NeSST_legendre_format(np_leg_table["a_l"])
                        return_dict["np_dxsec"] = {
                            "legendre": True,
                            "E": np_E_dist,
                            "a_l": np_al,
                            "N_l": np_al.shape[1],
                            "Q": np_Q,
                        }
                else:
                    # Default to Isotropic in CM if no angular distribution data section exists
                    return_dict["np_dxsec"] = {
                        "legendre": True,
                        "E": np.array([np_E[0], np_E[-1]]),
                        "a_l": np.zeros((2, 1)),
                        "N_l": 1,
                        "Q": np_Q,
                    }

        if file_manifest.inelastic:
            if return_dict["interactions"].inelastic:
                print(f"{filename}: Warning, loading inelastic cross section data from multiple files...")
            else:
                return_dict["interactions"].inelastic = True

            i_inelastic = 1
            while True:
                MT = MT_INELXSECTION_START + (i_inelastic - 1)
                try:
                    inelastic_table = mat.section_data[MF_XSECTION, MT]["sigma"]
                    inelastic_E, inelastic_sigma = inelastic_table.x, inelastic_table.y
                    return_dict[f"inelastic_xsec_n{i_inelastic}"] = {"E": inelastic_E, "sig": inelastic_sigma}
                    inelastic_Q = mat.section_data[MF_XSECTION, MT]["QI"]
                except:  # noqa: E722
                    break

                LTT = mat.section_data[MF_ENERGYANGLEDISTS, MT]["LTT"]

                if LTT == 0:  # Isotropic
                    return_dict[f"inelastic_dxsec_n{i_inelastic}"] = {
                        "legendre": True,
                        "E": np.array([inelastic_E[0], inelastic_E[-1]]),
                        "a_l": np.zeros((2, 1)),
                        "N_l": 1,
                        "Q": inelastic_Q,
                    }

                elif LTT == 1 or LTT == 3:  # Legendre
                    if LTT == 3:
                        print(
                            f"{filename}: Warning, LTT ({LTT}) for inelastic scattering, level {i_inelastic}, using Legendre only"
                        )
                    inelastic_table = mat.section_data[MF_ENERGYANGLEDISTS, MT]["legendre"]
                    inelastic_E, inelastic_al = (
                        inelastic_table["E"],
                        convert_to_NeSST_legendre_format(inelastic_table["a_l"]),
                    )
                    return_dict[f"inelastic_dxsec_n{i_inelastic}"] = {
                        "legendre": True,
                        "E": inelastic_E,
                        "a_l": inelastic_al,
                        "N_l": inelastic_al.shape[1],
                        "Q": inelastic_Q,
                    }

                elif LTT == 2:  # Tabulated
                    inelastic_table = mat.section_data[MF_ENERGYANGLEDISTS, MT]["tabulated"]
                    inelastic_E = inelastic_table["E"]
                    NE = inelastic_E.shape[0]
                    inelastic_mu, Nmu, inelastic_f = [], [], []
                    for iE in range(NE):
                        inelastic_mu.append(inelastic_table["mu"][iE].x)
                        inelastic_f.append(inelastic_table["mu"][iE].y)
                        Nmu.append(inelastic_table["mu"][iE].x.shape[0])
                    SDX = NeSST_SDX(NEin=NE, Ein=inelastic_E, Ncos=Nmu, cos=inelastic_mu, f=inelastic_f)
                    return_dict[f"inelastic_dxsec_n{i_inelastic}"] = {"legendre": False, "SDX": SDX, "Q": inelastic_Q}
                else:
                    print(
                        f"{filename}: Warning, LTT ({LTT}) for inelastic scattering, level {i_inelastic}, not recognised..."
                    )

                i_inelastic += 1

            return_dict["n_inelastic"] = i_inelastic - 1

        if file_manifest.n2n:
            if return_dict["interactions"].n2n:
                print(f"{filename}: Warning, loading n,2n cross section data from multiple files...")
            else:
                return_dict["interactions"].n2n = True

            n2n_table = mat.section_data[MF_XSECTION, MT_N2NXSECTION]["sigma"]
            n2n_E, n2n_sigma = n2n_table.x, n2n_table.y
            return_dict["n2n_xsec"] = {"E": n2n_E, "sig": n2n_sigma}

            n2n_Q = mat.section_data[MF_XSECTION, MT_N2NXSECTION]["QI"]
            n2n_table = mat.section_data[MF_PRODENERGYANGLEDISTS, MT_N2NXSECTION]
            n2n_products = n2n_table["products"]
            for n2n_product in n2n_products:
                if n2n_product["AWP"] == 1.0:  # is neutron?
                    if n2n_product["LAW"] == 6:
                        return_dict["n2n_dxsec"] = {
                            "LAW": 6,
                            "A_i": n2n_product["AWP"],
                            "A_e": n2n_product["AWP"],
                            "A_t": n2n_table["AWR"],
                            "A_p": n2n_product["AWP"],
                            "A_tot": n2n_product["distribution"]["APSX"],
                            "Q_react": n2n_Q,
                        }
                    elif n2n_product["LAW"] == 7:
                        DDX = convert_to_NeSST_LAW7_format(n2n_product["distribution"])
                        return_dict["n2n_dxsec"] = {"LAW": 7, "DDX": DDX, "Q_react": n2n_Q}
                    else:
                        print(f"{filename}: n2n data LAW = {n2n_product['LAW']}, NeSST cannot use...")

    return return_dict


def retrieve_total_cross_section_from_ENDF_file(filename):
    mat = endf.Material(ENDF_dir + filename)

    total_dataset = mat.section_data[MF_XSECTION, MT_TOTXSECTION]
    total_table = total_dataset["sigma"]
    total_E, total_sigma = total_table.x, total_table.y

    return total_E, total_sigma


def convert_to_NeSST_legendre_format(ENDF_al):
    max_Nl = max([len(a) for a in ENDF_al])
    uniform_len_al = [np.pad(a, (0, max_Nl - len(a))) for a in ENDF_al]
    return np.array(uniform_len_al)


def convert_to_NeSST_LAW1_format(ENDF_dist, target_Ein=None):
    if isinstance(ENDF_dist, dict):
        dist_list = ENDF_dist.get("distribution", [])
        NE = ENDF_dist.get("NE", len(dist_list))
        E_grid = ENDF_dist.get("energy") or ENDF_dist.get("Ein") or ENDF_dist.get("E")
    else:
        dist_list = ENDF_dist
        NE = len(dist_list)
        E_grid = None
    raw_Ein = []
    for iE in range(NE):
        if E_grid is not None and iE < len(E_grid):
            val = E_grid[iE]
        else:
            val = dist_list[iE].get("E", 0.0) if iE < len(dist_list) else 0.0
        if val is not None:
            raw_Ein.append(float(val))
    raw_Ein = np.asarray(raw_Ein, dtype=np.float64)
    """
    if target_Ein is not None:
        Ein_final = np.asarray(target_Ein, dtype=np.float64)
    else:
        Ein_final = raw_Ein
    """
    Ein_final = raw_Ein
    NE_final = len(Ein_final)
    default_cos = np.array([-1.0, 1.0], dtype=np.float64)
    n_cos = len(default_cos)
    Ein = Ein_final.tolist()
    Ncos = [n_cos] * NE_final
    cos = [default_cos] * NE_final
    NEout = {}
    Eout = {}
    f = {}
    Emax = {}
    for iE in range(NE_final):
        raw_idx = int(np.argmin(np.abs(raw_Ein - Ein_final[iE]))) if len(raw_Ein) > 0 else 0
        dist_Ein = dist_list[raw_idx]
        e_out_raw = dist_Ein.get("E'") or dist_Ein.get("Eprime") or [0.0]
        e_out_arr = np.atleast_1d(np.asarray(e_out_raw, dtype=np.float64))
        if "b" in dist_Ein and len(dist_Ein["b"]) > 0:
            f_val = np.atleast_1d(np.asarray(dist_Ein["b"], dtype=np.float64))
        else:
            f_val = np.ones_like(e_out_arr, dtype=np.float64) / max(len(e_out_arr), 1)
        if len(f_val) != len(e_out_arr):
            f_val = np.resize(f_val, e_out_arr.shape)
        for iM in range(n_cos):
            key = (iE, iM)
            NEout[key] = int(len(e_out_arr))
            Eout[key] = e_out_arr
            f[key] = f_val
            Emax[key] = float(np.amax(e_out_arr)) if len(e_out_arr) > 0 else 0.0

    if any(e is None for e in Ein):
        print("WARNING: Missing incident energy values in LAW=1 conversion!")
    DDX = NeSST_DDX(NEin=NE_final, Ein=Ein_final, Ncos=Ncos, cos=cos, NEout=NEout, Eout=Eout, f=f, Emax=Emax)

    return DDX

def convert_to_NeSST_LAW7_format(ENDF_dist):
    NE = ENDF_dist["NE"]
    Ein = []
    Ncos = []
    cos = []
    NEout = {}
    Eout = {}
    f = {}
    Emax = {}
    for iE in range(NE):
        dist_Ein = ENDF_dist["distribution"][iE]
        Ein.append(dist_Ein["E"])
        NMU = dist_Ein["NMU"]
        Ncos.append(NMU)
        cos_arr = np.zeros(NMU)
        for iM in range(NMU):
            dist_mu = dist_Ein["mu"][iM]
            cos_arr[iM] = dist_mu["mu"]
            NEout[(iE, iM)] = dist_mu["f"].x.shape[0]
            Eout[(iE, iM)] = dist_mu["f"].x
            f[(iE, iM)] = dist_mu["f"].y
            Emax[(iE, iM)] = np.amax(dist_mu["f"].x)
        cos.append(cos_arr)

    DDX = NeSST_DDX(NEin=NE, Ein=Ein, Ncos=Ncos, cos=cos, NEout=NEout, Eout=Eout, f=f, Emax=Emax)

    return DDX
