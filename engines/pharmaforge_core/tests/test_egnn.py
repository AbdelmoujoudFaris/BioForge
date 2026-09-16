import torch

from pharmaforge_core.models.egnn import PocketConditionedEGNN
from pharmaforge_core.utils.geometry import radius_graph, random_rotation_matrix


def test_pocket_atoms_stay_rigid(small_egnn_config, toy_pocket, toy_ligand):
    model = PocketConditionedEGNN(small_egnn_config, physchem_dim=6)
    n_pocket = toy_pocket["coords"].shape[0]
    n_ligand = toy_ligand["coords"].shape[0]

    coords = torch.cat([toy_pocket["coords"], toy_ligand["coords"]])
    atom_type_idx = torch.cat([toy_pocket["atom_type_idx"], toy_ligand["atom_type_idx"]])
    residue_type_idx = torch.cat([toy_pocket["residue_type_idx"], torch.zeros(n_ligand, dtype=torch.long)])
    physchem = torch.cat([toy_pocket["physchem"], toy_ligand["physchem"]])
    is_ligand = torch.cat([torch.zeros(n_pocket, dtype=torch.long), torch.ones(n_ligand, dtype=torch.long)])
    edge_index = radius_graph(coords, cutoff=8.0, max_neighbors=8)

    _, x_out = model(atom_type_idx, residue_type_idx, is_ligand, physchem, coords, edge_index)
    assert torch.allclose(x_out[:n_pocket], coords[:n_pocket])


def test_e3_equivariance(small_egnn_config, toy_pocket, toy_ligand):
    model = PocketConditionedEGNN(small_egnn_config, physchem_dim=6)
    n_pocket = toy_pocket["coords"].shape[0]
    n_ligand = toy_ligand["coords"].shape[0]

    coords = torch.cat([toy_pocket["coords"], toy_ligand["coords"]])
    atom_type_idx = torch.cat([toy_pocket["atom_type_idx"], toy_ligand["atom_type_idx"]])
    residue_type_idx = torch.cat([toy_pocket["residue_type_idx"], torch.zeros(n_ligand, dtype=torch.long)])
    physchem = torch.cat([toy_pocket["physchem"], toy_ligand["physchem"]])
    is_ligand = torch.cat([torch.zeros(n_pocket, dtype=torch.long), torch.ones(n_ligand, dtype=torch.long)])
    edge_index = radius_graph(coords, cutoff=8.0, max_neighbors=8)

    h1, x1 = model(atom_type_idx, residue_type_idx, is_ligand, physchem, coords, edge_index)

    R = random_rotation_matrix()
    t = torch.randn(3)
    coords_transformed = coords @ R.t() + t
    h2, x2 = model(atom_type_idx, residue_type_idx, is_ligand, physchem, coords_transformed, edge_index)

    expected = x1[n_pocket:] @ R.t() + t
    assert torch.allclose(x2[n_pocket:], expected, atol=1e-4)
    assert torch.allclose(h1, h2, atol=1e-4)


def test_radius_graph_respects_cutoff():
    coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    edge_index = radius_graph(coords, cutoff=5.0, max_neighbors=8)
    pairs = set(map(tuple, edge_index.t().tolist()))
    assert (0, 1) in pairs and (1, 0) in pairs
    assert (0, 2) not in pairs and (2, 0) not in pairs


def test_radius_graph_max_neighbors_cap():
    coords = torch.randn(50, 3) * 2.0  # dense cluster, well within any reasonable cutoff
    edge_index = radius_graph(coords, cutoff=100.0, max_neighbors=4)
    counts = torch.bincount(edge_index[0], minlength=50)
    assert counts.max().item() <= 4
