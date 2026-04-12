"""Tests for CSP solver."""

from backend.app.services.choreography.csp_solver import solve_layout


def test_basic_free_skate():
    inventory = {
        "jumps": ["3Lz", "3F", "3Lo", "3S", "2A", "2T", "1Eu"],
        "spins": ["CSp4", "LSp4", "FSp4"],
        "combinations": ["3Lz+2T", "3Lz+3T", "3F+2T", "3Lo+2T"],
    }
    music_features = {
        "duration": 240.0,
        "peaks": [15.0, 40.0, 65.0, 90.0, 120.0, 150.0, 180.0, 200.0, 220.0],
        "structure": [
            {"type": "intro", "start": 0.0, "end": 15.0},
            {"type": "verse", "start": 15.0, "end": 60.0},
            {"type": "chorus", "start": 60.0, "end": 120.0},
            {"type": "bridge", "start": 120.0, "end": 160.0},
            {"type": "chorus", "start": 160.0, "end": 220.0},
            {"type": "outro", "start": 220.0, "end": 240.0},
        ],
    }
    layouts = solve_layout(
        inventory=inventory,
        music_features=music_features,
        discipline="mens_singles",
        segment="free_skate",
        num_layouts=3,
    )
    assert len(layouts) >= 1
    assert len(layouts) <= 3

    layout = layouts[0]
    assert layout["total_tes"] > 0
    assert len(layout["elements"]) > 0


def test_layout_has_required_elements():
    inventory = {
        "jumps": ["3Lz", "3F", "3Lo", "3S", "2A", "2T", "1Eu"],
        "spins": ["CSp4", "LSp4", "FSp4"],
        "combinations": ["3Lz+2T", "3F+2T"],
    }
    music_features = {"duration": 240.0, "peaks": [15.0, 40.0, 65.0, 90.0, 120.0], "structure": []}
    layouts = solve_layout(
        inventory=inventory,
        music_features=music_features,
        discipline="mens_singles",
        segment="free_skate",
    )
    assert len(layouts) >= 1

    for layout in layouts:
        codes = [el["code"] for el in layout["elements"]]
        assert any("StSq" in c for c in codes), f"Missing StSq in {codes}"
        assert any("ChSq" in c for c in codes), f"Missing ChSq in {codes}"
        assert any("Sp" in c for c in codes), f"Missing spins in {codes}"


def test_short_program():
    inventory = {
        "jumps": ["3Lz", "3F", "3Lo", "2A", "2T"],
        "spins": ["CSp4", "LSp4", "FSp4"],
        "combinations": ["3Lz+2T"],
    }
    music_features = {"duration": 160.0, "peaks": [10.0, 30.0, 60.0, 100.0], "structure": []}
    layouts = solve_layout(
        inventory=inventory,
        music_features=music_features,
        discipline="mens_singles",
        segment="short_program",
    )
    assert len(layouts) >= 1
