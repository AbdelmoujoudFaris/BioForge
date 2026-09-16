from rdkit import Chem
from rdkit.Chem import AllChem

from pharmaforge_core.analysis.diversity import (
    internal_diversity,
    novelty_against_reference,
    scaffold_diversity,
    uniqueness,
)
from pharmaforge_core.analysis.physchem import profile_candidates, summarize
from pharmaforge_core.utils.chem import compute_physchem, is_valid, lipinski_pass, murcko_scaffold_smiles, tanimoto


def _embedded(smiles: str, seed: int = 1):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return Chem.RemoveHs(mol)


SAMPLE_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CCN(CC)CCOC(=O)c1ccc(N)cc1",
    "c1ccc2c(c1)ccc1ccccc12",  # anthracene-like scaffold
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
]


def test_compute_physchem_and_lipinski():
    mol = _embedded(SAMPLE_SMILES[0])
    props = compute_physchem(mol)
    assert 0.0 <= props["qed"] <= 1.0
    assert 1.0 <= props["sa_score"] <= 10.0
    assert lipinski_pass(props) is True


def test_is_valid_rejects_none():
    assert is_valid(None) is False
    assert is_valid(_embedded(SAMPLE_SMILES[0])) is True


def test_tanimoto_self_similarity_is_one():
    mol = _embedded(SAMPLE_SMILES[0])
    assert tanimoto(mol, mol) == 1.0


def test_murcko_scaffold_nonempty_for_aromatic():
    mol = _embedded(SAMPLE_SMILES[2])
    assert murcko_scaffold_smiles(mol) != ""


def test_profile_candidates_dataframe_shape():
    mols = [_embedded(s, seed=i) for i, s in enumerate(SAMPLE_SMILES)]
    df = profile_candidates(mols)
    assert len(df) == len(mols)
    assert "qed" in df.columns and "lipinski_pass" in df.columns

    summary = summarize(df)
    assert summary["n_valid"] == len(mols)
    assert 0.0 <= summary["mean_qed"] <= 1.0


def test_scaffold_diversity_and_uniqueness():
    mols = [_embedded(s, seed=i) for i, s in enumerate(SAMPLE_SMILES)]
    stats = scaffold_diversity(mols)
    assert stats["n_molecules"] == len(mols)
    assert stats["n_unique_scaffolds"] <= len(mols)
    assert uniqueness(mols) == 1.0  # all distinct SMILES


def test_internal_diversity_between_zero_and_one():
    mols = [_embedded(s, seed=i) for i, s in enumerate(SAMPLE_SMILES)]
    div = internal_diversity(mols)
    assert 0.0 <= div <= 1.0


def test_novelty_against_reference():
    mols = [_embedded(s, seed=i) for i, s in enumerate(SAMPLE_SMILES)]
    result = novelty_against_reference(mols, reference_smiles=["CCO", "c1ccccc1"])
    assert result["novelty_rate"] is not None
    assert 0.0 <= result["novelty_rate"] <= 1.0
