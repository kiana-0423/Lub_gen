"""Periodic table data and element lookup helpers."""

from __future__ import annotations

from dataclasses import dataclass

ELEMENT_SYMBOLS: tuple[str, ...] = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)

COMMON_ELEMENT_SYMBOLS: tuple[str, ...] = (
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "H",
)

PERIODIC_TABLE_GRID: tuple[tuple[str, ...], ...] = (
    ("H", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "He"),
    ("Li", "Be", "", "", "", "", "", "", "", "", "", "", "B", "C", "N", "O", "F", "Ne"),
    ("Na", "Mg", "", "", "", "", "", "", "", "", "", "", "Al", "Si", "P", "S", "Cl", "Ar"),
    ("K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"),
    ("Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe"),
    ("Cs", "Ba", "", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"),
    ("Fr", "Ra", "", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"),
    ("La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "", "", ""),
    ("Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "", "", ""),
)

_SYMBOL_TO_ATOMIC_NUMBER = {symbol: index for index, symbol in enumerate(ELEMENT_SYMBOLS, start=1)}


@dataclass(frozen=True, slots=True)
class ElementInfo:
    """Normalized element description used by the UI and editor state."""

    atomic_number: int
    symbol: str


def atomic_number_for_symbol(symbol: str) -> int:
    """Return the atomic number for an element symbol."""
    normalized_symbol = normalize_symbol(symbol)
    try:
        return _SYMBOL_TO_ATOMIC_NUMBER[normalized_symbol]
    except KeyError as exc:
        raise ValueError(f"Unknown element symbol: {symbol}") from exc


def symbol_for_atomic_number(atomic_number: int) -> str:
    """Return the symbol for an atomic number."""
    if atomic_number < 1 or atomic_number > len(ELEMENT_SYMBOLS):
        raise ValueError(f"Atomic number out of range: {atomic_number}")
    return ELEMENT_SYMBOLS[atomic_number - 1]


def element_info_for_symbol(symbol: str) -> ElementInfo:
    """Return normalized element info for a symbol."""
    atomic_number = atomic_number_for_symbol(symbol)
    return ElementInfo(atomic_number=atomic_number, symbol=symbol_for_atomic_number(atomic_number))


def element_info_for_atomic_number(atomic_number: int) -> ElementInfo:
    """Return normalized element info for an atomic number."""
    return ElementInfo(atomic_number=atomic_number, symbol=symbol_for_atomic_number(atomic_number))


def normalize_symbol(symbol: str) -> str:
    """Return a canonical element symbol."""
    stripped = symbol.strip()
    if not stripped:
        raise ValueError("Element symbol cannot be empty.")
    return stripped[0].upper() + stripped[1:].lower()
