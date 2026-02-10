"""
Canonical physical constants for the slingshot-solver package.

ALL modules in the slingshot package import constants from here.
Unit system: km-kg-s throughout.

    Distances : km
    Velocities: km/s
    Masses    : kg
    Time      : s
    Energy    : km²/s² (≡ MJ/kg, since 1 km²/s² = 10⁶ m²/s² = 10⁶ J/kg = 1 MJ/kg)
"""

# Gravitational constant [km³ kg⁻¹ s⁻²]
G_KM = 6.67430e-20

# Solar mass [kg]
M_SUN = 1.98847e30

# Jupiter mass [kg]
M_JUP = 1.898e27

# Jupiter radius [km]
R_JUP = 71492.0

# Solar radius [km]
R_SUN = 696000.0

# Astronomical Unit [km]
AU_KM = 1.495978707e8


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def mu_star(M_star_Msun: float) -> float:
    """Gravitational parameter μ = G·M for a star, in km³/s².

    Parameters
    ----------
    M_star_Msun : float
        Star mass in solar masses.
    """
    return G_KM * (M_star_Msun * M_SUN)


def mu_planet(M_planet_Mjup: float) -> float:
    """Gravitational parameter μ = G·M for a planet, in km³/s².

    Parameters
    ----------
    M_planet_Mjup : float
        Planet mass in Jupiter masses.
    """
    return G_KM * (M_planet_Mjup * M_JUP)


def au_to_km(au: float) -> float:
    """Convert astronomical units to km."""
    return au * AU_KM


def km_to_au(km: float) -> float:
    """Convert km to astronomical units."""
    return km / AU_KM
