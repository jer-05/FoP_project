#!/usr/bin/env python3
"""
well_mirror_tracer.py - optimise the shape parameters of a conic mirror at the bottom of a circular, liquid-filled well so that as much as possible of the light emitted inside the liquid exits the well at an angle within a sensor's acceptance cone.

Physics model
-------------
* The well is a cylinder of radius `WELL_RADIUS`. Its bottom is a rotationally symmetric
  conic mirror  z = conic(r);  the lowest point of the mirror sits at z = 0.
* The well contains a liquid (refractive index `REFRACTIVE_INDEX_LIQUID`) with uniformly
  distributed, isotropically emitting point sources.
* Bottom mirror and side walls reflect 100 % (no Fresnel losses anywhere).
* At the liquid/air surface a ray is totally internally reflected when
  n_liquid * sin(theta) >= 1, otherwise it refracts into air (Snell) and leaves the well.
* The liquid surface is flat, or a meniscus: a rotationally symmetric conic
  z = centre_height + conic(r) (the same family as the mirror) set by the contact angle of the
  liquid on the wall and a conic constant. The liquid volume is the same as for the flat fill.
  A ray that has left the liquid can still bounce off the dry wall above the contact line or
  re-enter the liquid through the raised rim of the meniscus; it has escaped for good once it
  moves upward above the highest point of the surface.
* The sensor is far away and large, so only the exit DIRECTION matters: a ray is
  collected iff sin(theta_air) <= numerical_aperture.

Two independent Monte Carlo estimators of the collection efficiency eta are provided:
* forward  : emit rays from the sources, trace, count what leaves inside the NA cone.
* backward : launch rays from the sensor side into the well at angles within the NA cone and score the path length they travel inside the liquid (reciprocity: uniform emitters make eta proportional to the liquid path length of the time-reversed rays). You can view this as a covering of phase space by ray bundles. Since weonly trace rays that end up at the sensor, this is more efficient than the forward method.
Agreement between the two (and with the analytic flat-mirror result) validates the code.

Reading order (top to bottom is also the order of the physics):
  1. Well, Meniscus           geometry, optical constants, file-wide flags
  2. ConicMirror              the mirror  z = conic(s),  s = x^2 + y^2, analytic ray hit
     LiquidSurface            the liquid surface, flat or meniscus, same conic machinery
  3. optics helpers           reflection, cylinder wall hit, direction sampling
  4. trace_rays               vectorised bounce loop (mirror / wall / surface)
  5. forward_efficiency       sources -> trace -> NA cone  => eta
  6. backward_efficiency      sensor  -> trace -> liquid path length  => eta
  7. optimisation             global grid scan + Nelder-Mead on (curvature, conic constant)
  8. figures                  cross-sections, efficiency landscape, convergence
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32 | np.float64]

# =============================================================================
# 0. File-wide switches
# =============================================================================
# False: the liquid surface stays at Well.liquid_height whatever the mirror shape, so the liquid volume changes with the mirror (a deeper bowl holds more liquid).
# True : the liquid volume is kept equal to the flat-mirror volume pi R^2 liquid_height and the surface height moves up or down with the mirror.
CONSTANT_VOLUME = False

# When a ray somehow ends up exactly on a surface, we need to make sure that when it is reflected it actually leaves that local surface and does not immediately hit it again. When the next reflection is whithin SELF_HIT_TOLERANCE of the previous hit, we consider it a "self-hit" and terminate the ray.
SELF_HIT_TOLERANCE = 1e-7

# To make sure that when solving the quadratic root we only consider the real relevant side of the quadratic and not the wrong branch/other sheet. Maximum relative mismatch between a quadratic root and the conic sag used to decide that a root lies on the real mirror branch of the conic (and not on the mathematically identical "other sheet" of the same quadric).
MIRROR_BRANCH_TOLERANCE = 1e-5

# A ray in air that moves upward and is within this distance (relative to the well radius) below the highest point of the liquid surface can not hit the surface any more and has escaped the well.
SURFACE_TOP_TOLERANCE = 1e-9


# =============================================================================
# 1. Geometry
# =============================================================================
@dataclass(frozen=True)
class Well:
    # radius of the cylindrical well
    well_radius: float = 0.5
    # height of the liquid surface above the lowest point of the mirror (fixed-height mode); in constant-volume mode this is the fill height of the flat-mirror reference well
    liquid_height: float = 0.6
    # height of the rim of the well
    rim_height: float = 1.0
    # refractive index of the liquid
    refractive_index: float = 1.33
    # numerical aperture of the sensor in air, rays are collected iff sin(theta_air) <= NA
    numerical_aperture: float = 0.5
    # the highest point of the mirror must stay this many units below the lowest point of the liquid surface; the mirror must be fully submerged
    surface_margin: float = 0.02
    # shape of the liquid surface, see Meniscus (the default is a flat surface)
    meniscus: Meniscus = field(default_factory=lambda: Meniscus())

    @property
    def reference_volume(self) -> float:
        """Liquid volume of the well with a flat mirror."""
        return np.pi * self.well_radius**2 * self.liquid_height

    @property
    def critical_sine(self) -> float:
        """sin(theta_liquid) above which the surface totally internally reflects."""
        return 1.0 / self.refractive_index

    @property
    def collection_solid_angle(self) -> float:
        """Solid angle (in air) of the sensor acceptance cone sin(theta) <= NA."""
        return 2 * np.pi * (1 - np.sqrt(1 - self.numerical_aperture**2))


@dataclass(frozen=True)
class Meniscus:
    """Shape of the liquid surface: a rotationally symmetric conic
        sag(s) = curvature * s / (1 + sqrt(1 - (1 + conic_constant) * curvature^2 * s)), s = r^2,
    measured from the centre of the surface. The curvature follows from the slope at the wall, which is set by the contact angle: dz/dr = cot(contact_angle) at r = R, so
        curvature * R = cot(contact_angle) / sqrt(1 + (1 + conic_constant) * cot(contact_angle)^2).
    """

    # angle between the wall and the liquid surface, measured through the liquid, in degrees. 90 is a flat surface, smaller angles are a wetting liquid that climbs the wall (concave meniscus, water on glass is roughly 30-50 degrees), larger angles give a dome (non-wetting liquid or an overfilled well)
    contact_angle_degrees: float = 90.0
    # 0 is a spherical cap, the meniscus of a well much narrower than the capillary length (no gravity). Larger values flatten the centre and confine the curvature to the wall region, which is what gravity does in a wide well. The slope at the wall is always set by the contact angle.
    conic_constant: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.contact_angle_degrees - 90.0) < 1e-12

    def curvature(self, well_radius: float) -> float:
        """Vertex curvature of the surface conic for a well of this radius (0 when flat)."""
        if self.is_flat:
            return 0.0
        cotangent = 1.0 / np.tan(np.radians(self.contact_angle_degrees))
        denominator = 1 + (1 + self.conic_constant) * cotangent**2
        if denominator <= 0:
            return np.nan  # hyperbolic meniscus that can not reach the wall slope: invalid
        return float(cotangent / np.sqrt(denominator) / well_radius)


# =============================================================================
# 2. Mirror: conic sag as a function of s = r^2
# =============================================================================
class ConicMirror:
    """Rotationally symmetric conic mirror

        sag(s) = curvature * s / (1 + sqrt(1 - (1 + conic_constant) * curvature^2 * s)),
        s = r^2 = x^2 + y^2,

    i.e. the standard optical-design conic (conic_constant: 0 sphere, -1 paraboloid, < -1 hyperboloid, between -1 and 0 prolate ellipsoid, > 0 oblate ellipsoid). ISO 10110
    A positive curvature gives a bowl (vertex is the lowest point), a negative curvature a dome (edge is the lowest point).
    The profile is shifted so that its lowest point is at z = 0, which is the floor of the well.

    Radial positions are passed around as s = r^2 to avoid square root computations in the hot loop.
    """

    def __init__(self, well_radius: float, curvature: float, conic_constant: float) -> None:
        self.well_radius = float(well_radius)
        self.curvature = float(curvature)
        self.conic_constant = float(conic_constant)
        self.parameters = np.array([curvature, conic_constant])

        # the sqrt in the sag must stay real (and the slope finite) out to the wall
        edge_root_argument = 1 - (1 + conic_constant) * curvature**2 * well_radius**2
        self.is_valid = bool(np.isfinite(edge_root_argument) and edge_root_argument > 1e-9)
        if not self.is_valid:
            self.vertex_height = self.edge_height = self.max_height = np.nan
            return

        # the conic sag is monotonic in s, so its extreme values are at the vertex and the edge
        edge_sag = self._sag_from_vertex(np.array(well_radius**2))[()]
        self.vertex_height = float(max(0.0, -edge_sag))  # shift so that min(z) = 0
        self.edge_height = float(edge_sag + self.vertex_height)
        self.max_height = max(self.vertex_height, self.edge_height)

    # --- shape -------------------------------------------------------------------------
    def _sag_from_vertex(self, radius_squared: FloatArray) -> FloatArray:
        """Conic sag measured from the vertex (unshifted). The vertex is the lowest point of the mirror, so the sag is negative for a bowl and positive for a dome."""
        root_argument = 1 - (1 + self.conic_constant) * self.curvature**2 * radius_squared
        return self.curvature * radius_squared / (1 + np.sqrt(root_argument))

    def height(self, radius_squared: FloatArray) -> FloatArray:
        """Mirror height z above the well floor at s = r^2 (nan outside the valid domain)."""
        return self._sag_from_vertex(radius_squared) + self.vertex_height

    def profile(self, radius: FloatArray) -> FloatArray:
        """Mirror height as a function of the radius r (convenience for plotting)."""
        return self.height(np.asarray(radius, float) ** 2)

    def slope_wrt_radius_squared(self, radius_squared: FloatArray) -> FloatArray:
        """d height / d s. From dz/dr = c r / sqrt(...) and ds/dr = 2r: dz/ds = c / (2 sqrt(...))."""
        root_argument = 1 - (1 + self.conic_constant) * self.curvature**2 * radius_squared
        return self.curvature / (2 * np.sqrt(root_argument))

    def mean_height(self, quadrature_order: int = 64) -> float:
        """Area-averaged mirror height over the well floor:
        (1 / pi R^2) * integral z dA = (1 / R^2) * integral_0^{R^2} height(s) ds,
        computed with Gauss-Legendre quadrature in s (exact for the flat and paraboloid
        cases, better than 1e-6 for the others)."""
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        radius_squared_nodes = (nodes + 1) / 2 * self.well_radius**2
        return float(np.sum(weights * self.height(radius_squared_nodes)) / 2)

    def unit_normals(self, points: FloatArray) -> FloatArray:
        """Upward unit normals of the surface z = height(x^2 + y^2) at the given (N, 3) points.

        With z = H(s) and s = x^2 + y^2 the gradient of z - H(s) is
        (-2 x H'(s), -2 y H'(s), 1), which is normal to the surface."""
        radius_squared = points[:, 0] ** 2 + points[:, 1] ** 2
        gradient_factor = 2 * self.slope_wrt_radius_squared(radius_squared)
        normals = np.stack(
            [-gradient_factor * points[:, 0], -gradient_factor * points[:, 1], np.ones(len(points))],
            axis=1,
        )
        return normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # --- ray intersection ------------------------------------------------------------
    def intersection_distance(self, origins: FloatArray, directions: FloatArray, max_distance: FloatArray) -> FloatArray:
        """Distance along each ray  P(t) = origin + t * direction  to the mirror, or inf. Where t is the "distance"/"time" along the ray. The ray is assumed to be in the liquid, so it can only hit the mirror from above. The ray may also hit the wall or the liquid surface first, so `max_distance` is the distance to the nearer of those two surfaces.

        The conic sag satisfies the implicit quadric
            c s - 2 w + (1 + k) c w^2 = 0 ; w = z - vertex_height,
        (multiply the sag formula out and rearrange). Substituting the ray gives a plain
        quadratic in t, so the hit is found analytically for all rays at once:
            s(t) = s_2 t^2 + s_1 t + s_0,   w(t) = w_0 + t d_z.
        The quadric also contains the second sheet of the conic (e.g. the upper half of the sphere), so each root is checked against the sag function and rejected if it does not lie on the mirror branch. Roots behind the origin, on the surface the ray just left,
        or beyond `max_distance` (the nearer of wall and liquid surface) are rejected too.

        Args:
            origins: (N, 3) array of ray origins
            directions: (N, 3) array of ray directions
            max_distance: (N,) array of distances to the nearer of the wall and liquid surface

        Returns:
            (N,) array of distances to the mirror, or inf if the ray does not hit
        """
        origin_x, origin_y, origin_z = origins[:, 0], origins[:, 1], origins[:, 2]
        direction_x, direction_y, direction_z = directions[:, 0], directions[:, 1], directions[:, 2]
        # the vertex is the lowest point of the mirror, so the height above the vertex is the origin's z minus the vertex height
        height_above_vertex = origin_z - self.vertex_height

        radial_quadratic = direction_x**2 + direction_y**2
        radial_linear = 2 * (origin_x * direction_x + origin_y * direction_y)
        radial_constant = origin_x**2 + origin_y**2

        conic_factor = (1 + self.conic_constant) * self.curvature
        quadratic_coefficient = self.curvature * radial_quadratic + conic_factor * direction_z**2
        linear_coefficient = self.curvature * radial_linear - 2 * direction_z + 2 * conic_factor * height_above_vertex * direction_z
        constant_coefficient = self.curvature * radial_constant - 2 * height_above_vertex + conic_factor * height_above_vertex**2
        candidate_distances = solve_quadratic(quadratic_coefficient, linear_coefficient, constant_coefficient)  # (N, 2), nan where there is no real root

        with np.errstate(invalid="ignore", divide="ignore"):
            hit_x = origin_x[:, None] + candidate_distances * direction_x[:, None]
            hit_y = origin_y[:, None] + candidate_distances * direction_y[:, None]
            hit_z = origin_z[:, None] + candidate_distances * direction_z[:, None]
            height_mismatch = np.abs(hit_z - self.height(hit_x**2 + hit_y**2))
            on_mirror_branch = height_mismatch < MIRROR_BRANCH_TOLERANCE * self.well_radius

        is_usable = (
            np.isfinite(candidate_distances) & (candidate_distances > SELF_HIT_TOLERANCE * self.well_radius) & (candidate_distances <= max_distance[:, None]) & on_mirror_branch
        )
        candidate_distances = np.where(is_usable, candidate_distances, np.inf)
        return candidate_distances.min(axis=1)


def solve_quadratic(quadratic_coefficient: FloatArray, linear_coefficient: FloatArray, constant_coefficient: FloatArray) -> FloatArray:
    """Real roots of a t^2 + b t + c = 0 for arrays of coefficients, shape (N, 2), nan where
    a root does not exist. Uses the numerically stable q-form (no cancellation when a is
    small) and falls back to the linear solution where a is exactly zero (flat mirror)."""
    quadratic_coefficient = np.asarray(quadratic_coefficient, float)
    linear_coefficient = np.asarray(linear_coefficient, float)
    constant_coefficient = np.asarray(constant_coefficient, float)
    roots = np.full((len(linear_coefficient), 2), np.nan)

    is_linear = quadratic_coefficient == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        linear_root = -constant_coefficient / linear_coefficient
        discriminant = linear_coefficient**2 - 4 * quadratic_coefficient * constant_coefficient
        has_real_roots = ~is_linear & (discriminant >= 0)
        stable_term = -0.5 * (linear_coefficient + np.copysign(np.sqrt(np.abs(discriminant)), linear_coefficient))
        first_root = stable_term / quadratic_coefficient
        second_root = constant_coefficient / stable_term

    roots[is_linear, 0] = linear_root[is_linear]
    roots[has_real_roots, 0] = first_root[has_real_roots]
    roots[has_real_roots, 1] = second_root[has_real_roots]
    return roots


class LiquidSurface(ConicMirror):
    """The liquid/air surface  z = centre_height + sag(s)  with the sag of the meniscus conic (see Meniscus). It is the same conic as the mirror but shifted to the centre height of the surface instead of to the well floor, so the sag, the analytic ray hit and the unit normals of ConicMirror are reused as they are (the ray hit works from both sides of the surface). A flat surface is the conic with curvature 0.

    The centre height is chosen such that the liquid volume equals the flat fill (the pipetted volume): centre_height = flat surface height - mean sag.
    """

    def __init__(self, well_radius: float, meniscus: Meniscus, flat_height: float) -> None:
        self.well_radius = float(well_radius)
        self.curvature = meniscus.curvature(well_radius)
        self.conic_constant = float(meniscus.conic_constant)
        self.meniscus = meniscus
        self.parameters = np.array([self.curvature, self.conic_constant])

        # the sqrt in the sag must stay real (and the slope finite) out to the wall
        edge_root_argument = 1 - (1 + self.conic_constant) * self.curvature**2 * well_radius**2
        self.is_valid = bool(np.isfinite(edge_root_argument) and edge_root_argument > 1e-9)
        if not self.is_valid:
            self.vertex_height = self.edge_height = self.max_height = self.min_height = np.nan
            return

        # the conic sag is monotonic in s, so its extreme values are at the centre and the edge
        edge_sag = self._sag_from_vertex(np.array(well_radius**2))[()]
        # vertex_height is the height of the centre of the surface here (the "vertex" of the surface conic); the shift keeps the liquid volume equal to the flat fill
        self.vertex_height = float(flat_height - self.mean_sag())
        self.edge_height = float(edge_sag + self.vertex_height)
        self.max_height = max(self.vertex_height, self.edge_height)
        self.min_height = min(self.vertex_height, self.edge_height)

    @property
    def is_flat(self) -> bool:
        return self.curvature == 0.0

    @property
    def centre_height(self) -> float:
        return self.vertex_height

    def mean_sag(self, quadrature_order: int = 64) -> float:
        """Area-averaged sag over the well: (1 / R^2) * integral_0^{R^2} sag(s) ds, Gauss-Legendre in s like ConicMirror.mean_height."""
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        radius_squared_nodes = (nodes + 1) / 2 * self.well_radius**2
        return float(np.sum(weights * self._sag_from_vertex(radius_squared_nodes)) / 2)


# ---- buildability -----------------------------------------------------------------------
# check if the mirror is fully submerged, the liquid does not overflow the rim and the conic is defined defined for the full radius of the well
def liquid_surface_height(well: Well, mirror: ConicMirror) -> float:
    """Height of the flat liquid surface above the well floor for this mirror; with a meniscus this is the height of the flat surface that holds the same liquid volume."""
    if CONSTANT_VOLUME:
        # volume = pi R^2 (surface - mean mirror height) = pi R^2 liquid_height
        return well.liquid_height + mirror.mean_height()
    return well.liquid_height


def liquid_surface(well: Well, mirror: ConicMirror) -> LiquidSurface:
    """The actual liquid surface (flat or meniscus) above this mirror."""
    return LiquidSurface(well.well_radius, well.meniscus, liquid_surface_height(well, mirror))


def liquid_volume(well: Well, mirror: ConicMirror) -> float:
    return np.pi * well.well_radius**2 * (liquid_surface_height(well, mirror) - mirror.mean_height())


# TODO dit zouden ook asserts kunnen zijn, maar dan moet de code die de mirror maakt dat ook checken
def why_not_buildable(well: Well, mirror: ConicMirror) -> str | None:
    """Return the reason this design cannot be built, or None if it is fine."""
    if not mirror.is_valid:
        return "conic is not defined out to the wall"
    surface = liquid_surface(well, mirror)
    if not surface.is_valid:
        return "meniscus conic is not defined out to the wall"
    if mirror.max_height > surface.min_height - well.surface_margin:
        return "mirror sticks out of the liquid"
    if surface.max_height > well.rim_height:
        return "liquid overflows the rim"
    return None


# =============================================================================
# 3. Optics helpers
# =============================================================================
def reflect(directions: FloatArray, unit_normals: FloatArray) -> FloatArray:
    """Mirror reflection of (N, 3) direction vectors about (N, 3) unit normals."""
    normal_components = np.sum(directions * unit_normals, axis=1, keepdims=True)
    return directions - 2 * normal_components * unit_normals


def refract(directions: FloatArray, unit_normals: FloatArray, index_ratio: float) -> FloatArray:
    """Snell refraction of (N, 3) unit direction vectors through a surface with (N, 3) unit normals. The normals must point towards the side the ray comes from (direction . normal < 0); index_ratio = n_from / n_to. Total internal reflection must have been excluded by the caller."""
    cosine_incidence = -np.sum(directions * unit_normals, axis=1, keepdims=True)
    cosine_transmitted = np.sqrt(1 - index_ratio**2 * (1 - cosine_incidence**2))
    return index_ratio * directions + (index_ratio * cosine_incidence - cosine_transmitted) * unit_normals


def cylinder_wall_distance(origins: FloatArray, directions: FloatArray, well_radius: float) -> FloatArray:
    """Distance along each ray to the cylinder wall r = well_radius, for rays that start inside the cylinder (the larger root of the quadratic). inf for vertical rays."""
    radial_quadratic = directions[:, 0] ** 2 + directions[:, 1] ** 2
    radial_linear = 2 * (origins[:, 0] * directions[:, 0] + origins[:, 1] * directions[:, 1])
    radial_constant = origins[:, 0] ** 2 + origins[:, 1] ** 2 - well_radius**2
    discriminant = np.maximum(radial_linear**2 - 4 * radial_quadratic * radial_constant, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        stable_term = -0.5 * (radial_linear + np.copysign(np.sqrt(discriminant), radial_linear))
        distance = np.fmax(stable_term / radial_quadratic, radial_constant / stable_term)
    is_usable = (radial_quadratic > 1e-14) & (distance > SELF_HIT_TOLERANCE * well_radius)
    return np.where(is_usable, distance, np.inf)


def cylinder_wall_normals(points: FloatArray) -> FloatArray:
    """Inward unit normals of the cylinder wall at (N, 3) points on it."""
    radius = np.hypot(points[:, 0], points[:, 1])
    return np.stack([-points[:, 0] / radius, -points[:, 1] / radius, np.zeros(len(points))], axis=1)


def sample_isotropic_directions(count: int, rng: np.random.Generator) -> FloatArray:
    """Unit vectors uniformly distributed over the sphere, shape (count, 3).  cos(theta) is uniform in [-1, 1] (NOT theta itself: the solid angle element is sin(theta) dtheta dphi) and the azimuth is uniform in [0, 2 pi)."""
    cosine_polar = rng.uniform(-1.0, 1.0, count)
    azimuth = rng.uniform(0.0, 2 * np.pi, count)
    sine_polar = np.sqrt(1 - cosine_polar**2)
    return np.stack([sine_polar * np.cos(azimuth), sine_polar * np.sin(azimuth), cosine_polar], axis=1)


def sample_cone_directions(count: int, sine_half_angle: float, rng: np.random.Generator) -> FloatArray:
    """Unit vectors uniformly distributed (per solid angle) inside the cone sin(theta) <= sine_half_angle around +z, shape (count, 3)."""
    cosine_min = np.sqrt(1 - sine_half_angle**2)
    cosine_polar = rng.uniform(cosine_min, 1.0, count)
    azimuth = rng.uniform(0.0, 2 * np.pi, count)
    sine_polar = np.sqrt(1 - cosine_polar**2)
    return np.stack([sine_polar * np.cos(azimuth), sine_polar * np.sin(azimuth), cosine_polar], axis=1)


def sample_disc_points(count: int, disc_radius: float, rng: np.random.Generator) -> FloatArray:
    """(count, 2) points uniformly distributed over a disc (r = R sqrt(u) keeps the density uniform)."""
    radius = disc_radius * np.sqrt(rng.random(count))
    azimuth = rng.uniform(0.0, 2 * np.pi, count)
    return np.stack([radius * np.cos(azimuth), radius * np.sin(azimuth)], axis=1)


# =============================================================================
# 4. Bounce loop inside the liquid
# =============================================================================
class Event(IntEnum):
    """What a ray hits next."""

    MIRROR = 0  # specular reflection on the conic bottom (rays in the liquid only)
    WALL = 1  # specular reflection on the cylinder wall (wet or dry part)
    SURFACE = 2  # from the liquid: total internal reflection, or refraction out of the liquid; from the air: refraction back into the liquid


@dataclass
class TraceResult:
    # (N, 3) unit direction in AIR of every ray that left the liquid (zeros otherwise)
    exit_directions_air: np.ndarray
    # (N,) True for rays that left the liquid through the surface and escaped the well
    has_escaped: np.ndarray
    # (N,) total distance travelled inside the liquid
    liquid_path_length: np.ndarray
    # (N,) number of mirror + wall + surface events per ray
    bounce_count: np.ndarray
    # (N, 3) final position (the exit point on the surface for escaped rays)
    final_positions: np.ndarray
    # list of (N, 3) position snapshots after every bounce (only when record_paths=True)
    path_snapshots: list

    @property
    def unresolved_fraction(self) -> float:
        """Rays still bouncing when the bounce budget ran out (should be ~0 for a curved
        mirror; for a flat mirror it is exactly the fraction trapped by TIR forever)."""
        return float(1 - self.has_escaped.mean())


def trace_rays(
    origins: FloatArray,
    directions: FloatArray,
    well: Well,
    mirror: ConicMirror,
    surface: LiquidSurface,
    max_bounces: int = 200,
    record_paths: bool = False,
    start_in_air: bool = False,
) -> TraceResult:
    """Vectorised bounce loop. Every pass moves EVERY still-active ray to its next event:
        mirror  -> specular reflection (lossless)
        wall    -> specular reflection (lossless), the wet and the dry part alike
        surface -> from the liquid: total internal reflection if n sin(theta_incidence) >= 1 (theta_incidence relative to the local normal), otherwise the ray refracts into air
                   from the air: the ray refracts back into the liquid (it met the raised rim of the meniscus again)
    A ray in air that moves upward above the highest point of the surface has escaped the well for good.
    # NOTE dit is mss niet handig voor de gpu implementatie want dan maakt de hoeveelheid werk toch niet veel uit
    Only the indices of active rays are processed, so the work shrinks as rays leave.

    Args:
        origins: (N, 3) array of ray origins
        directions: (N, 3) array of ray directions
        well: Well object with geometry and optical constants
        mirror: ConicMirror object with the mirror shape
        surface: LiquidSurface object with the shape of the liquid surface (flat or meniscus)
        max_bounces: maximum number of bounces before a ray is considered trapped
        record_paths: if True, store a snapshot of the positions after every bounce (costs a lot of memory)
        start_in_air: if True, the rays start in the air above the surface (time-reversed sensor rays) and are refracted into the liquid by the loop

    Returns:
        TraceResult with the exit directions, escape flags, path lengths, bounce counts, final positions, and optional path snapshots.
    """
    positions, directions = np.array(origins, float), np.array(directions, float)
    ray_count = len(positions)
    refractive_index = well.refractive_index
    surface_top = surface.max_height - SURFACE_TOP_TOLERANCE * well.well_radius

    is_active = np.ones(ray_count, bool)
    is_in_liquid = np.full(ray_count, not start_in_air)
    has_escaped = np.zeros(ray_count, bool)
    exit_directions_air = np.zeros((ray_count, 3))
    liquid_path_length = np.zeros(ray_count)
    bounce_count = np.zeros(ray_count, int)
    # kan gebruikt worden om de paden van de rays te visualiseren, maar kost veel geheugen
    path_snapshots = [positions.copy()] if record_paths else []

    for _ in range(max_bounces):
        # a ray in air that moves upward above the whole surface has nothing left to hit: it has escaped
        escapes_now = np.flatnonzero(is_active & ~is_in_liquid & (directions[:, 2] > 0) & (positions[:, 2] >= surface_top))
        exit_directions_air[escapes_now] = directions[escapes_now]
        has_escaped[escapes_now] = True
        is_active[escapes_now] = False

        active_indices = np.flatnonzero(is_active)
        if active_indices.size == 0:
            break
        active_positions, active_directions = positions[active_indices], directions[active_indices]
        active_in_liquid = is_in_liquid[active_indices]

        # --- distance to each of the three things a ray can hit ---
        distance_to_wall = cylinder_wall_distance(active_positions, active_directions, well.well_radius)
        # the surface is hit from below (liquid) or from above (air), the conic ray hit handles both
        distance_to_surface = surface.intersection_distance(active_positions, active_directions, distance_to_wall)

        # the mirror only needs to be searched up to the nearer of the other two, and only by rays in the liquid
        distance_to_mirror = np.where(
            active_in_liquid,
            mirror.intersection_distance(active_positions, active_directions, np.minimum(distance_to_wall, distance_to_surface)),
            np.inf,
        )
        # one column per Event, filled by name, so the argmin index is the Event value
        all_distances = np.empty((active_indices.size, len(Event)))
        all_distances[:, Event.MIRROR] = distance_to_mirror
        all_distances[:, Event.WALL] = distance_to_wall
        all_distances[:, Event.SURFACE] = distance_to_surface
        event = all_distances.argmin(axis=1)
        travelled = all_distances[np.arange(active_indices.size), event]

        # a ray that cannot hit anything is dropped as lost (numerical corner case) in the liquid, in air it has escaped (vertical ray)
        is_stuck = ~np.isfinite(travelled)
        travelled = np.where(is_stuck, 0.0, travelled)
        escaped_in_air = active_indices[is_stuck & ~active_in_liquid]
        exit_directions_air[escaped_in_air] = directions[escaped_in_air]
        has_escaped[escaped_in_air] = True
        is_active[active_indices[is_stuck]] = False

        active_positions = active_positions + travelled[:, None] * active_directions
        liquid_path_length[active_indices] += np.where(active_in_liquid, travelled, 0.0)
        bounce_count[active_indices] += 1

        hits_mirror = event == Event.MIRROR
        if hits_mirror.any():
            active_directions[hits_mirror] = reflect(active_directions[hits_mirror], mirror.unit_normals(active_positions[hits_mirror]))

        hits_wall = event == Event.WALL
        if hits_wall.any():
            active_directions[hits_wall] = reflect(active_directions[hits_wall], cylinder_wall_normals(active_positions[hits_wall]))

        hits_surface = np.flatnonzero((event == Event.SURFACE) & ~is_stuck & active_in_liquid)
        if hits_surface.size:
            # Snell about the local (upward) normal of the surface: n sin(theta_incidence) = sin(theta_air); TIR when that exceeds 1.
            surface_normals = surface.unit_normals(active_positions[hits_surface])
            cosine_incidence = np.sum(active_directions[hits_surface] * surface_normals, axis=1)
            sine_air_squared = refractive_index**2 * (1 - cosine_incidence**2)

            is_total_internal_reflection = sine_air_squared >= 1.0
            TIR_reflected = hits_surface[is_total_internal_reflection]
            active_directions[TIR_reflected] = reflect(active_directions[TIR_reflected], surface_normals[is_total_internal_reflection])
            transmitted = hits_surface[~is_total_internal_reflection]

            # refract expects normals that point against the rays; the surface normals point up, into the air, so flip them for rays that come from the liquid
            active_directions[transmitted] = refract(active_directions[transmitted], -surface_normals[~is_total_internal_reflection], refractive_index)
            is_in_liquid[active_indices[transmitted]] = False

            # a ray that leaves above the whole surface has escaped at once; the others stay in the air part of the well (they may re-enter the liquid through the raised rim, or bounce off the dry wall first)
            escaped_now = transmitted[(active_directions[transmitted, 2] > 0) & (active_positions[transmitted, 2] >= surface_top)]
            escaped_global = active_indices[escaped_now]
            exit_directions_air[escaped_global] = active_directions[escaped_now]
            has_escaped[escaped_global] = True
            is_active[escaped_global] = False

        re_enters_liquid = np.flatnonzero((event == Event.SURFACE) & ~is_stuck & ~active_in_liquid)
        if re_enters_liquid.size:
            # from air into the liquid refraction is always possible; the upward normals already point against these downward rays
            surface_normals = surface.unit_normals(active_positions[re_enters_liquid])
            active_directions[re_enters_liquid] = refract(active_directions[re_enters_liquid], surface_normals, 1.0 / refractive_index)
            is_in_liquid[active_indices[re_enters_liquid]] = True

        positions[active_indices], directions[active_indices] = active_positions, active_directions
        if record_paths:
            path_snapshots.append(positions.copy())

    return TraceResult(exit_directions_air, has_escaped, liquid_path_length, bounce_count, positions, path_snapshots)


# =============================================================================
# 5. Forward estimator: emitters -> sensor
# =============================================================================
@dataclass
class EfficiencyEstimate:
    eta: float
    standard_error: float
    surface_height: float
    unresolved_fraction: float
    ray_count: int


class EmitterCloud:
    """Emitter positions and directions sampled ONCE in the cylinder r <= R, 0 <= z <= z_top
    that contains every possible liquid volume. For a given mirror the emitters that actually
    lie in the liquid (above the mirror, below the surface) are selected with a mask, so all
    designs are scored with the same random numbers (common random numbers: the optimiser
    sees a smooth objective instead of Monte Carlo noise). Though the whole optimization could now also a bit biased to produce a shape that cuts off certain inconvenient rays, but that is a small effect and the alternative (resample for every mirror) is much more expensive."""

    def __init__(self, well: Well, count: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        # with a meniscus the surface climbs above the flat fill height near the wall, so the sampled cylinder has to reach the highest point of the surface
        flat_surface = LiquidSurface(well.well_radius, well.meniscus, well.liquid_height)
        self.sampling_height = well.rim_height if CONSTANT_VOLUME else min(well.rim_height, flat_surface.max_height)
        xy = sample_disc_points(count, well.well_radius, rng)
        heights = rng.uniform(0.0, self.sampling_height, count)
        self.positions = np.column_stack([xy, heights])
        self.directions = sample_isotropic_directions(count, rng)

    def inside_liquid(self, mirror: ConicMirror, surface: LiquidSurface):
        radius_squared = self.positions[:, 0] ** 2 + self.positions[:, 1] ** 2
        return (self.positions[:, 2] <= surface.height(radius_squared)) & (self.positions[:, 2] >= mirror.height(radius_squared))


def is_inside_acceptance_cone(exit_directions_air, well: Well):
    """sin(theta_air)^2 = dx^2 + dy^2 of the unit exit direction in air, compared with NA^2."""
    sine_squared = exit_directions_air[:, 0] ** 2 + exit_directions_air[:, 1] ** 2
    return sine_squared <= well.numerical_aperture**2


def forward_efficiency(well: Well, mirror: ConicMirror, emitters: EmitterCloud, max_bounces: int = 200) -> EfficiencyEstimate:
    """eta = (number of emitted rays that leave the liquid inside the NA cone) / (number emitted)."""
    if why_not_buildable(well, mirror):
        # TODO misschien beter een exception gooien, maar dan moet de optimiser dat ook kunnen afvangen
        return EfficiencyEstimate(0.0, 0.0, np.nan, np.nan, 0)
    surface = liquid_surface(well, mirror)
    selected = emitters.inside_liquid(mirror, surface)
    trace = trace_rays(emitters.positions[selected], emitters.directions[selected], well, mirror, surface, max_bounces)
    is_collected = trace.has_escaped & is_inside_acceptance_cone(trace.exit_directions_air, well)
    ray_count = int(selected.sum())
    eta = is_collected.mean()
    return EfficiencyEstimate(
        float(eta),
        float(np.sqrt(eta * (1 - eta) / ray_count)),
        surface.centre_height,
        trace.unresolved_fraction,
        ray_count,
    )


# =============================================================================
# 6. Backward (adjoint) estimator: sensor -> liquid path length
# =============================================================================
class SensorRayBundle:
    """Time-reversed rays sampled ONCE: start points uniform on the liquid surface disc, air
    directions uniform (per solid angle) inside the acceptance cone, pointing DOWN into the
    liquid after refraction. The cos(theta_air) factor of the etendue is carried as a weight.
    With a meniscus the rays start in the air on the plane through the highest point of the surface, pointing down with the air directions, and the trace refracts them in where they meet the surface (after a bounce off the dry wall, if they meet that first). The etendue element cos(theta_air) dA dOmega_air of the beam through that plane is conserved along the air path, so the weight is the same."""

    def __init__(self, well: Well, count: int, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.surface_points = sample_disc_points(count, well.well_radius, rng)
        air_directions = sample_cone_directions(count, well.numerical_aperture, rng)
        self.cosine_air = air_directions[:, 2]
        self.downward_air_directions = air_directions * np.array([1.0, 1.0, -1.0])
        # Snell into the liquid: sin(theta_liquid) = sin(theta_air) / n, going downwards
        sine_liquid_x = air_directions[:, 0] / well.refractive_index
        sine_liquid_y = air_directions[:, 1] / well.refractive_index
        # use the identity sin^2 + cos^2 = 1 to avoid numerical issues with arcsin
        cosine_liquid = np.sqrt(1 - sine_liquid_x**2 - sine_liquid_y**2)
        self.liquid_directions = np.stack([sine_liquid_x, sine_liquid_y, -cosine_liquid], axis=1)


def backward_efficiency(well: Well, mirror: ConicMirror, sensor_rays: SensorRayBundle, max_bounces: int = 200) -> EfficiencyEstimate:
    """Reciprocity estimator.

    eta = (1 / (4 pi V)) * integral over the liquid of the solid angle of directions that
    reach the sensor. Parametrise that set of (position, direction) pairs by where and in
    which direction the ray leaves the surface: the phase-space measure dV dOmega_liquid
    equals (path length) x (etendue), and the etendue is conserved through the surface,
        cos(theta_liquid) dA dOmega_liquid = cos(theta_air) dA dOmega_air / n^2.
    Hence
        eta = (pi R^2 Omega_NA) / (4 pi V n^2) * < cos(theta_air) * L >,
    with L the total path length a time-reversed ray travels inside the liquid before it
    leaves again, averaged over surface points and cone directions."""

    # TODO misschien beter een exception gooien, maar dan moet de optimiser dat ook kunnen afvangen
    if why_not_buildable(well, mirror):
        return EfficiencyEstimate(0.0, 0.0, np.nan, np.nan, 0)

    surface = liquid_surface(well, mirror)
    if surface.is_flat:
        # flat surface: start on the surface, already refracted into the liquid
        start_positions = np.column_stack([sensor_rays.surface_points, np.full(len(sensor_rays.surface_points), surface.centre_height)])
        trace = trace_rays(start_positions, sensor_rays.liquid_directions, well, mirror, surface, max_bounces)
    else:
        # meniscus: start in the air on the plane through the top of the surface, the trace refracts the rays in
        start_positions = np.column_stack([sensor_rays.surface_points, np.full(len(sensor_rays.surface_points), surface.max_height)])
        trace = trace_rays(start_positions, sensor_rays.downward_air_directions, well, mirror, surface, max_bounces, start_in_air=True)

    # The prefactor is the constant part of the integral that does not depend on the ray path. The average of cos(theta_air) * L over the rays gives the efficiency.
    prefactor = well.well_radius**2 * well.collection_solid_angle / (4 * liquid_volume(well, mirror) * well.refractive_index**2)
    weighted_lengths = prefactor * sensor_rays.cosine_air * trace.liquid_path_length

    ray_count = len(weighted_lengths)

    return EfficiencyEstimate(
        float(weighted_lengths.mean()),
        float(weighted_lengths.std() / np.sqrt(ray_count)),
        surface.centre_height,
        trace.unresolved_fraction,
        ray_count,
    )


def flat_mirror_efficiency_analytic(well: Well) -> float:
    """Flat mirror + flat surface + mirror walls preserve sin(theta_liquid) at every bounce, so a ray is
    collected iff sin(theta_liquid) <= NA / n (its |cos(theta)| >= sqrt(1 - NA^2/n^2)), and
    otherwise either leaves outside the NA cone or is trapped by TIR forever. For isotropic
    emission that fraction is 1 - sqrt(1 - (NA / n)^2)."""
    sine_limit = well.numerical_aperture / well.refractive_index
    return 1 - np.sqrt(1 - sine_limit**2)


# =============================================================================
# 7. Optimisation of (curvature, conic constant)
# =============================================================================
class EfficiencyScorer:
    """Scores a design with one of the two estimators, always with the same pre-sampled rays
    (common random numbers), so that the optimiser sees a deterministic objective.
    The backward estimator is ~10x cheaper per ray and has lower variance (no rays are
    wasted on emitters that never reach the sensor), the forward estimator is the literal
    definition of eta; both must agree, which is checked in the final re-score."""

    def __init__(self, well: Well, ray_count: int, estimator: Literal["forward", "backward"] = "forward", seed: int = 1):
        self.well, self.estimator = well, estimator
        if estimator == "forward":
            self.rays = EmitterCloud(well, ray_count, seed)
        elif estimator == "backward":
            self.rays = SensorRayBundle(well, ray_count, seed)
        else:
            raise ValueError("estimator must be 'forward' or 'backward'")

    def __call__(self, mirror: ConicMirror) -> EfficiencyEstimate:
        if self.estimator == "forward":
            return forward_efficiency(self.well, mirror, self.rays)
        return backward_efficiency(self.well, mirror, self.rays)


def scan_landscape(well: Well, scorer: EfficiencyScorer, normalised_curvatures: FloatArray, conic_constants: FloatArray) -> FloatArray:
    """Efficiency on a grid of (curvature * R, conic constant). nan where not buildable.
    Shows the whole landscape (local optima, infeasible regions) before the local search."""
    eta_grid = np.full((len(conic_constants), len(normalised_curvatures)), np.nan)
    # TODO this seems very parallelisable, but the scorer is not thread-safe because it reuses the same pre-sampled rays. Could make a thread-safe scorer that copies the rays for each thread, or just use multiprocessing.
    for row, conic_constant in enumerate(conic_constants):
        for column, normalised_curvature in enumerate(normalised_curvatures):
            mirror = ConicMirror(well.well_radius, normalised_curvature / well.well_radius, conic_constant)
            if why_not_buildable(well, mirror) is None:
                eta_grid[row, column] = scorer(mirror).eta
        print(f"  scan: conic constant {conic_constant:+.2f} done, best so far {np.nanmax(eta_grid):.4f}", flush=True)
    return eta_grid


def optimise(
    well: Well,
    scorer: EfficiencyScorer,
    start_parameters: FloatArray,
    simplex_steps: tuple[float, float] = (0.2, 0.5),
    max_evaluations: int = 80,
    callback=None,
) -> tuple[FloatArray, list[tuple[FloatArray, float]]]:
    """Nelder-Mead on x = (curvature * R, conic constant), maximising the efficiency
    scored with the same ray sample every time (deterministic objective).
    Returns the best parameters and the evaluation history [(x, eta), ...].

    Args:
        well: Well object with geometry and optical constants
        scorer: EfficiencyScorer object to evaluate the efficiency of a mirror
        start_parameters: initial guess for (curvature * R, conic constant)
        simplex_steps: initial step sizes for the Nelder-Mead simplex in each parameter
        max_evaluations: maximum number of efficiency evaluations before stopping
        callback: optional function to call after each evaluation with (parameters, eta)

    Returns:
        best_parameters: array of best (curvature * R, conic constant) found
        history: list of tuples (parameters, eta) for each evaluation
    """
    from scipy.optimize import minimize

    start_parameters = np.asarray(start_parameters, float)
    history = []

    def negative_efficiency(parameters):
        mirror = ConicMirror(well.well_radius, parameters[0] / well.well_radius, parameters[1])
        eta = scorer(mirror).eta
        history.append((parameters.copy(), eta))
        print(
            f"  eval {len(history):3d}: curvature*R={parameters[0]:+.4f}  k={parameters[1]:+.3f}  eta={eta:.4f}",
            flush=True,
        )
        if callback is not None:
            callback(parameters, eta)
        return -eta

    initial_simplex = np.vstack(
        [
            start_parameters,
            start_parameters + [simplex_steps[0], 0.0],
            start_parameters + [0.0, simplex_steps[1]],
        ]
    )
    result = minimize(
        negative_efficiency,
        start_parameters,
        method="Nelder-Mead",
        # xatol and fatol are the tolerances for convergence in the parameters and function value, respectively. The initial simplex is provided to guide the search.
        # xatol parameter tolerance for convergence
        # fatol function value tolerance for convergence
        options=dict(initial_simplex=initial_simplex, maxfev=max_evaluations, xatol=1e-3, fatol=1e-5),
    )
    return result.x, history


# =============================================================================
# 8. Figures
# NOTE: This section has been written with the assistance of AI
# NOTE: Scienceplots zou hier nice zijn, maar is traag, en heeft latex afhankelijkheid dan moet de gebruiker dat eerst installeren. Voor nu gewoon matplotlib defaults.
# =============================================================================
def meridional_ray_paths(well: Well, mirror: ConicMirror, backward: bool = False, columns: int = 5, rows: int = 2, rays_per_emitter: int = 4, max_bounces: int = 5, seed: int = 3):
    """Rays in the plane y = 0 (they stay in it by rotational symmetry) for a 2D picture.
    forward : a jittered grid of emitters across the liquid, `rays_per_emitter` in-plane directions each
    backward: rays entering from the sensor side across the surface, inside the NA cone
    Returns (polylines in (x, z), is_collected flags)."""
    rng = np.random.default_rng(seed)
    surface = liquid_surface(well, mirror)
    if backward:
        start_x = np.linspace(-0.9, 0.9, 7) * well.well_radius
        air_angles = np.linspace(-1, 1, 3) * np.arcsin(well.numerical_aperture)
        start_x, air_angles = np.repeat(start_x, 3), np.tile(air_angles, 7)
        if surface.is_flat:
            sine_liquid = np.sin(air_angles) / well.refractive_index
            origins = np.column_stack([start_x, np.zeros_like(start_x), np.full_like(start_x, surface.centre_height)])
            directions = np.column_stack([sine_liquid, np.zeros_like(start_x), -np.sqrt(1 - sine_liquid**2)])
        else:
            origins = np.column_stack([start_x, np.zeros_like(start_x), np.full_like(start_x, surface.max_height)])
            directions = np.column_stack([np.sin(air_angles), np.zeros_like(start_x), -np.cos(air_angles)])
    else:
        grid_x = (-well.well_radius + (np.arange(columns)[:, None] + rng.uniform(0.2, 0.8, (columns, rows))) * 2 * well.well_radius / columns).ravel()
        depth_fraction = ((np.arange(rows)[None, :] + rng.uniform(0.2, 0.8, (columns, rows))) / rows).ravel()
        floor_z = mirror.profile(np.abs(grid_x))
        grid_z = floor_z + depth_fraction * (surface.profile(np.abs(grid_x)) - floor_z)
        angles = (np.arange(rays_per_emitter)[None, :] / rays_per_emitter * 2 * np.pi + rng.uniform(0, 2 * np.pi, (len(grid_x), 1))).ravel()
        origins = np.repeat(np.column_stack([grid_x, np.zeros_like(grid_x), grid_z]), rays_per_emitter, axis=0)
        directions = np.column_stack([np.sin(angles), np.zeros_like(angles), np.cos(angles)])

    trace = trace_rays(origins, directions, well, mirror, surface, max_bounces=max_bounces, record_paths=True, start_in_air=backward and not surface.is_flat)
    snapshots = np.stack(trace.path_snapshots, axis=1)  # (N, bounces + 1, 3)
    is_collected = trace.has_escaped & is_inside_acceptance_cone(trace.exit_directions_air, well)
    if backward:  # every sensor ray is by construction connected to the sensor
        is_collected = np.ones(len(origins), bool)
    polylines = []
    emitter_points = origins[::rays_per_emitter][:, [0, 2]] if not backward else origins[:, [0, 2]]
    for ray_index in range(len(origins)):
        points = snapshots[ray_index][:, [0, 2]]
        if trace.has_escaped[ray_index] and not backward:  # short leg in air so the exit direction is visible
            air_leg = points[-1] + 0.25 * well.well_radius * trace.exit_directions_air[ray_index][[0, 2]]
            points = np.vstack([points, air_leg])
        polylines.append(points)
    return polylines, is_collected, emitter_points


def draw_cross_section(axis, well: Well, mirror: ConicMirror, title: str, backward: bool = False):
    import matplotlib.pyplot as plt  # noqa: F401  (keeps the import local to the figure code)

    surface = liquid_surface(well, mirror)
    radius_grid = np.linspace(-well.well_radius, well.well_radius, 401)
    mirror_profile = mirror.profile(np.abs(radius_grid))
    surface_profile = surface.profile(np.abs(radius_grid))
    axis.fill_between(radius_grid, mirror_profile, surface_profile, color="#dbe8f7", lw=0, zorder=0)
    polylines, is_collected, emitter_points = meridional_ray_paths(well, mirror, backward=backward)
    axis.scatter(emitter_points[:, 0], emitter_points[:, 1], s=12, color="#1f1f1e", zorder=4)
    for points, collected in zip(polylines, is_collected):
        axis.plot(
            points[:, 0], points[:, 1], color="#2a78d6" if collected else "#b8b7b1", lw=0.7 if collected else 0.5, alpha=0.8 if collected else 0.6, zorder=2 if collected else 1
        )
    axis.plot(radius_grid, mirror_profile, color="#1f1f1e", lw=2.2, zorder=3)
    for side in (-1, 1):
        axis.plot([side * well.well_radius] * 2, [mirror.edge_height, well.rim_height], color="#1f1f1e", lw=2.2)
    axis.plot(radius_grid, surface_profile, color="#7fa9dd", lw=1.2)
    cone_half_angle = np.arcsin(well.numerical_aperture)
    cone_length = 0.3 * well.well_radius
    for side in (-1, 1):
        axis.plot([0, side * cone_length * np.tan(cone_half_angle)], [surface.centre_height, surface.centre_height + cone_length], color="#eb6834", lw=1.4, ls="--")
    subtitle = "rays from the sensor traced back into the liquid" if backward else f"in-plane emitter rays, {is_collected.mean():.0%} of the drawn ones collected"
    axis.set_title(f"{title}\n{subtitle}", loc="left", fontsize=10)
    axis.set_aspect("equal")
    axis.set_xlim(-1.15 * well.well_radius, 1.15 * well.well_radius)
    axis.set_ylim(-0.05, well.rim_height * 1.02)
    axis.set_xlabel("x (slice through the axis)")
    axis.set_ylabel("z")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def make_figures(well: Well, flat_mirror: ConicMirror, best_mirror: ConicMirror, scan: dict, history, results: dict, output_prefix: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold", "figure.facecolor": "white"})
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)  # create the output folder if needed

    # --- figure 1: cross-sections (flat vs optimised, forward and backward view) ---
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    draw_cross_section(axes[0], well, flat_mirror, f"Flat mirror, eta = {results['flat_forward'].eta:.3f}")
    draw_cross_section(axes[1], well, best_mirror, f"Optimised conic, eta = {results['best_forward'].eta:.3f}")
    draw_cross_section(axes[2], well, best_mirror, "Optimised conic, backward view", backward=True)
    figure.suptitle(
        f"Well R = {well.well_radius:g}, liquid height {results['best_forward'].surface_height:.3f} at the centre "
        f"({'constant volume' if CONSTANT_VOLUME else 'fixed height'}), n = {well.refractive_index:g}, NA = {well.numerical_aperture:g}, "
        f"contact angle {well.meniscus.contact_angle_degrees:g} deg, meniscus k = {well.meniscus.conic_constant:g}.  "
        "Blue = reaches the sensor (or comes from it), grey = lost; orange dashes = acceptance cone.",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(f"{output_prefix}_cross_sections.png", dpi=160)
    print(f"figure written to {output_prefix}_cross_sections.png")

    # --- figure 2: efficiency landscape with the optimiser path ---
    figure, axis = plt.subplots(figsize=(8, 6))
    normalised_curvatures, conic_constants, eta_grid = scan["normalised_curvatures"], scan["conic_constants"], scan["eta_grid"]
    image = axis.pcolormesh(normalised_curvatures, conic_constants, eta_grid, cmap="Blues", shading="nearest")
    figure.colorbar(image, ax=axis, label="efficiency eta")
    path = np.array([parameters for parameters, _ in history])
    axis.plot(path[:, 0], path[:, 1], color="#eb6834", lw=1, marker=".", ms=4, label="Nelder-Mead evaluations")
    axis.plot(best_mirror.curvature * well.well_radius, best_mirror.conic_constant, marker="*", ms=14, color="#eb6834", mec="black", label="optimum")
    axis.axvline(0, color="#9b9a95", lw=0.8, ls=":")
    axis.set_xlabel("vertex curvature x well radius  (bowl > 0 > dome)")
    axis.set_ylabel("conic constant k  (0 sphere, -1 paraboloid)")
    axis.set_title("Efficiency landscape (white = not buildable)", loc="left")
    axis.legend(frameon=False, loc="lower left")
    figure.tight_layout()
    figure.savefig(f"{output_prefix}_landscape.png", dpi=160)
    print(f"figure written to {output_prefix}_landscape.png")

    # --- figure 3: convergence + forward/backward agreement ---
    figure, (axis_left, axis_right) = plt.subplots(1, 2, figsize=(12, 4.5))
    etas = np.array([eta for _, eta in history])
    axis_left.plot(np.arange(1, len(etas) + 1), etas, ".", color="#9b9a95", label="evaluation")
    axis_left.plot(np.arange(1, len(etas) + 1), np.maximum.accumulate(etas), color="#2a78d6", lw=2, label="best so far")
    axis_left.axhline(results["flat_forward"].eta, color="#1f1f1e", lw=1, ls="--", label="flat mirror")
    axis_left.set_xlabel("Nelder-Mead evaluation")
    axis_left.set_ylabel("efficiency eta (optimiser estimator)")
    axis_left.set_title("Optimiser progress", loc="left")
    axis_left.legend(frameon=False)

    labels = ["flat\nanalytic", "flat\nforward", "flat\nbackward", "best\nforward", "best\nbackward"]
    values = [results["flat_analytic"], results["flat_forward"].eta, results["flat_backward"].eta, results["best_forward"].eta, results["best_backward"].eta]
    errors = [0, results["flat_forward"].standard_error, results["flat_backward"].standard_error, results["best_forward"].standard_error, results["best_backward"].standard_error]
    axis_right.bar(labels, values, yerr=np.array(errors) * 2, color=["#1f1f1e", "#2a78d6", "#7fa9dd", "#2a78d6", "#7fa9dd"], capsize=4)
    for position, value in enumerate(values):
        axis_right.text(position, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    axis_right.set_ylabel("eta")
    axis_right.set_title("Forward vs backward estimator (error bars = 2 sigma)", loc="left")
    for axis in (axis_left, axis_right):
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(f"{output_prefix}_convergence.png", dpi=160)
    print(f"figure written to {output_prefix}_convergence.png")


# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimise a conic well mirror with forward + backward Monte Carlo.")
    parser.add_argument("--rays", type=int, default=100_000, help="rays per evaluation during optimisation")
    parser.add_argument("--scan-rays", type=int, default=20_000, help="rays per grid point in the landscape scan")
    parser.add_argument("--final-rays", type=int, default=1_000_000, help="rays for the final re-score")
    parser.add_argument("--grid", type=int, default=21, help="grid points per axis in the landscape scan")
    parser.add_argument("--max-evaluations", type=int, default=80)
    parser.add_argument(
        "--estimator", choices=["forward", "backward"], default="forward", help="estimator used inside the scan and the optimiser (the final re-score always uses both)"
    )
    parser.add_argument("--contact-angle", type=float, default=90.0, help="contact angle of the liquid on the wall in degrees: 90 flat surface, < 90 wetting meniscus, > 90 dome")
    parser.add_argument("--meniscus-k", type=float, default=0.0, help="conic constant of the meniscus: 0 spherical cap, larger values flatten the centre")
    parser.add_argument("--output-prefix", default="well_mirror")
    arguments = parser.parse_args()

    well = Well(meniscus=Meniscus(arguments.contact_angle, arguments.meniscus_k))
    print(f"well: {well}\nconstant volume: {CONSTANT_VOLUME}")

    # --- 1. validation on the flat mirror (analytic result available) ---
    flat_mirror = ConicMirror(well.well_radius, 0.0, 0.0)
    eta_flat_analytic = flat_mirror_efficiency_analytic(well)
    emitters_final = EmitterCloud(well, arguments.final_rays, seed=99)
    sensor_rays_final = SensorRayBundle(well, arguments.final_rays // 4, seed=98)
    start_time = time.time()
    flat_forward = forward_efficiency(well, flat_mirror, emitters_final)
    flat_backward = backward_efficiency(well, flat_mirror, sensor_rays_final)
    print(
        f"flat mirror: analytic {eta_flat_analytic:.4f} (flat surface) | forward {flat_forward.eta:.4f} +- {flat_forward.standard_error:.4f} "
        f"| backward {flat_backward.eta:.4f} +- {flat_backward.standard_error:.4f}   ({time.time() - start_time:.1f} s)"
    )

    # --- 2. coarse landscape scan ---
    scorer_scan = EfficiencyScorer(well, arguments.scan_rays, arguments.estimator, seed=0)
    normalised_curvatures = np.linspace(-2.0, 3.0, arguments.grid)
    conic_constants = np.linspace(-4.0, 1.0, arguments.grid)
    start_time = time.time()
    eta_grid = scan_landscape(well, scorer_scan, normalised_curvatures, conic_constants)
    best_row, best_column = np.unravel_index(np.nanargmax(eta_grid), eta_grid.shape)
    scan_best = np.array([normalised_curvatures[best_column], conic_constants[best_row]])
    print(f"scan done in {time.time() - start_time:.0f} s: best grid point curvature*R={scan_best[0]:+.3f}, k={scan_best[1]:+.3f}, eta={eta_grid[best_row, best_column]:.4f}")

    # --- 3. local refinement from the best grid point ---
    scorer_opt = EfficiencyScorer(well, arguments.rays, arguments.estimator, seed=1)
    start_time = time.time()
    best_parameters, history = optimise(well, scorer_opt, scan_best, max_evaluations=arguments.max_evaluations)
    print(f"optimisation done in {time.time() - start_time:.0f} s")

    # --- 4. unbiased re-score with fresh rays, both estimators ---
    best_mirror = ConicMirror(well.well_radius, best_parameters[0] / well.well_radius, best_parameters[1])
    best_forward = forward_efficiency(well, best_mirror, emitters_final)
    best_backward = backward_efficiency(well, best_mirror, sensor_rays_final)
    print(
        f"best conic: curvature={best_mirror.curvature:.4f} (x R = {best_parameters[0]:+.4f}), k={best_mirror.conic_constant:+.4f}, "
        f"edge height={best_mirror.edge_height:.4f}, vertex height={best_mirror.vertex_height:.4f}, liquid surface (centre)={best_forward.surface_height:.4f}\n"
        f"  forward  eta = {best_forward.eta:.4f} +- {best_forward.standard_error:.4f}  (unresolved {best_forward.unresolved_fraction:.2%})\n"
        f"  backward eta = {best_backward.eta:.4f} +- {best_backward.standard_error:.4f}  (unresolved {best_backward.unresolved_fraction:.2%})\n"
        f"  gain over flat mirror: {best_forward.eta / flat_forward.eta:.2f} x"
    )

    make_figures(
        well,
        flat_mirror,
        best_mirror,
        dict(normalised_curvatures=normalised_curvatures, conic_constants=conic_constants, eta_grid=eta_grid),
        history,
        dict(flat_analytic=eta_flat_analytic, flat_forward=flat_forward, flat_backward=flat_backward, best_forward=best_forward, best_backward=best_backward),
        arguments.output_prefix,
    )
