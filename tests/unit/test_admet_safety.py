from rdkit import Chem

from bioforge.core.admet_safety import profile_admet


def test_profile_admet_known_molecule():
    mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")  # aspirin
    profile = profile_admet(mol)
    assert profile.method == "rule_based_v1"
    assert isinstance(profile.human_intestinal_absorption, bool)
    assert profile.herg_liability_risk in ("low", "high")
