#!/usr/bin/env python3
# Author: Paulo R. A. Gilz
"""Generate AOE result files from a simulated aircraft, not from noise.

    python tools/gen_flight_data.py out.aoe --size 500 --runs 25
    python tools/gen_flight_data.py set1.aoe set2.aoe set3.aoe --size 500

Why bother simulating anything. Test data made of sine waves and random walks
lets a viewer look correct while being wrong in the ways that matter: it has no
transients worth preserving, so decimation appears lossless whatever it does;
every channel has the same shape, so a wrong column index is invisible; and
nothing is constant per run, so the ISO machinery has nothing to key on. Flight
data has short-period oscillations that must survive at every zoom level,
channels spanning eleven orders of magnitude between radians and metres, and
per-run constants (mass, centre of gravity, aerodynamic coefficients) that are
exactly what ISO filtering exists for.

The model
---------
Longitudinal rigid-body dynamics with a pitch-rate autopilot, plus a
point-mass lateral model with a coordinated-turn approximation:

    V̇  = (T cos α − D)/m − g sin γ
    γ̇  = (L + T sin α − m g cos γ) / (m V)
    q̇  = M / I_yy
    θ̇  = q,        α = θ − γ
    ψ̇  = g tan φ / V          (coordinated turn)
    ḣ  = V sin γ

with ISA atmosphere, a drag polar C_D = C_D0 + k C_L², and a pitching moment
whose static stability follows the centre of gravity:

    C_mα = C_Lα (x_cg − x_ac) / c̄

so moving the CG aft genuinely reduces the short-period damping and frequency,
and the runs differ in ways a flight-test engineer would recognise rather than
in a random seed.

Integration is explicit RK2 (midpoint) at a fixed step, vectorised across runs:
every run advances together as one NumPy array, so the Python-level loop runs
once per time step rather than once per sample. That is what makes a 500 MB
file take a couple of minutes instead of an hour.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import List, Sequence

import numpy as np

# ---------------------------------------------------------------- constants

G0 = 9.80665           # m/s^2
R_AIR = 287.05287      # J/(kg K)
GAMMA_AIR = 1.4
T0 = 288.15            # K at sea level
P0 = 101325.0          # Pa
LAPSE = 0.0065         # K/m, troposphere
H_TROP = 11000.0       # m

# Channels written, in order. The names follow the convention of the files this
# viewer was built for: upper case, underscore separated, no units in the name.
CHANNELS: Sequence[str] = (
    "TIME",
    "ALPHA", "BETA", "THETA", "PHI", "PSI", "GAMMA",
    "P", "Q", "R",
    "ALTITUDE", "SPEED", "MACH", "EAS", "DYNAMIC_PRESSURE",
    "EXTERNAL_TEMPERATURE", "AIR_DENSITY",
    "NX", "NY", "NZ",
    "MASS", "CG", "FUEL_FLOW", "THRUST", "DRAG", "LIFT",
    "ELEVATOR", "AILERON", "RUDDER", "THROTTLE",
    "AIRCRAFT_POSITION_X_INERTIAL_FRAME",
    "AIRCRAFT_POSITION_Y_INERTIAL_FRAME",
    "AIRCRAFT_POSITION_Z_INERTIAL_FRAME",
    # Constant for the whole run: this is what the ISO tab keys on.
    "INITIAL_MASS", "CG_REF", "CL_ALPHA", "THRUST_RATING", "CONFIGURATION",
)


# --------------------------------------------------------------- atmosphere


def isa(altitude: np.ndarray):
    """ISA temperature, density and speed of sound. Valid to ~20 km.

    Above the tropopause the temperature is held and the pressure decays
    exponentially, which is the standard model and matters here because these
    aircraft cruise right at that boundary -- a linear extrapolation would put
    the temperature 30 K wrong at 15 km.
    """
    altitude = np.clip(altitude, -500.0, 20000.0)
    below = altitude < H_TROP

    temperature = np.where(below, T0 - LAPSE * altitude, T0 - LAPSE * H_TROP)
    exponent = G0 / (LAPSE * R_AIR)

    pressure_trop = P0 * np.power(np.maximum(temperature / T0, 1e-6), exponent)
    p_tropopause = P0 * math.pow((T0 - LAPSE * H_TROP) / T0, exponent)
    t_tropopause = T0 - LAPSE * H_TROP
    pressure_strat = p_tropopause * np.exp(
        -G0 * (altitude - H_TROP) / (R_AIR * t_tropopause)
    )
    pressure = np.where(below, pressure_trop, pressure_strat)

    density = pressure / (R_AIR * temperature)
    sound = np.sqrt(GAMMA_AIR * R_AIR * temperature)
    return temperature, density, sound


# ------------------------------------------------------------------- runs


class Fleet:
    """A set of runs integrated together.

    Every quantity is an array with one entry per run, so a time step is a
    handful of NumPy operations regardless of how many runs there are.
    """

    def __init__(self, count: int, seed: int = 7):
        rng = np.random.default_rng(seed)
        n = count

        # --- per-run configuration, held constant and written out as ISO keys
        # Three discrete mass cases rather than a continuous spread: ISO
        # grouping and filtering are far more useful to exercise against
        # repeated values, which is also how real test points are flown.
        mass_cases = np.array([58000.0, 68000.0, 78000.0])
        self.initial_mass = mass_cases[np.arange(n) % 3] + rng.normal(0, 400, n)

        cg_cases = np.array([0.18, 0.28, 0.38])       # fraction of chord
        self.cg_ref = cg_cases[(np.arange(n) // 3) % 3] + rng.normal(0, 0.004, n)

        self.cl_alpha = 5.1 + 0.35 * ((np.arange(n) // 9) % 3)   # per radian
        self.thrust_rating = np.where(np.arange(n) % 2 == 0, 1.0, 0.86)
        self.configuration = (np.arange(n) % 4).astype(np.float64)

        # --- airframe
        self.wing_area = 122.6 + rng.normal(0, 0.5, n)   # m^2
        self.chord = 4.29 + rng.normal(0, 0.02, n)       # m
        self.span = 34.1
        self.x_ac = 0.25                                  # aerodynamic centre
        self.i_yy = 4.2e6 + rng.normal(0, 4e4, n)         # kg m^2

        self.cl0 = 0.22 + rng.normal(0, 0.01, n)
        self.cd0 = 0.021 + 0.004 * (self.configuration > 1) + rng.normal(0, 6e-4, n)
        self.oswald_k = 1.0 / (math.pi * 9.45 * 0.82)
        self.cl_de = 0.42
        self.cm0 = 0.035 + rng.normal(0, 0.004, n)
        self.cm_q = -22.0 - 2.0 * rng.random(n)
        self.cm_de = -1.42 - 0.08 * rng.random(n)
        self.cl_q = 4.4

        self.max_thrust = 2 * 120000.0 * self.thrust_rating   # N, two engines
        self.tsfc = 1.65e-5                                    # kg/(N s)

        # --- state
        self.mass = self.initial_mass.copy()
        self.altitude = 450.0 + rng.uniform(-60, 60, n)
        self.speed = 150.0 + rng.uniform(-4, 4, n)             # TAS, m/s
        self.gamma = np.radians(6.5 + rng.uniform(-0.6, 0.6, n))

        # Start trimmed. Starting at an arbitrary attitude means every run
        # opens with a large transient that has nothing to do with the
        # manoeuvre, and with a light static margin that transient takes most
        # of the flight to damp out.
        _t, density, _a = isa(self.altitude)
        q_bar = 0.5 * density * self.speed ** 2
        cl_needed = self.mass * G0 / np.maximum(q_bar * self.wing_area, 1.0)
        alpha_trim = np.clip((cl_needed - self.cl0) / self.cl_alpha, -0.05, 0.25)
        self.theta = self.gamma + alpha_trim
        self.pitch_rate = np.zeros(n)
        self.psi = np.radians(rng.uniform(0, 20, n))
        self.phi = np.zeros(n)
        self.beta = np.zeros(n)
        self.roll_rate = np.zeros(n)
        self.yaw_rate = np.zeros(n)
        self.x = np.zeros(n)
        self.y = np.zeros(n)
        self.fuel_used = np.zeros(n)

        # --- autopilot targets, per run, so the runs are not carbon copies
        self.target_altitude = 9000.0 + 900.0 * (np.arange(n) % 5)
        self.target_mach = 0.74 + 0.015 * ((np.arange(n) // 5) % 4)
        self.target_eas = 152.0 + 4.0 * (np.arange(n) % 3)      # m/s EAS
        self.target_heading = np.radians(rng.uniform(-40, 40, n))

        self.elevator = np.zeros(n)
        self.aileron = np.zeros(n)
        self.rudder = np.zeros(n)
        self.throttle = np.full(n, 0.85)

        self.count = n
        self._rng = rng

    # ------------------------------------------------------------- dynamics

    def _forces(self, altitude, speed, alpha, pitch_rate, elevator, throttle):
        temperature, density, sound = isa(altitude)
        speed = np.maximum(speed, 40.0)          # keep the model out of a stall
        q_bar = 0.5 * density * speed * speed

        c_l = (self.cl0 + self.cl_alpha * alpha
               + self.cl_q * pitch_rate * self.chord / (2.0 * speed)
               + self.cl_de * elevator)
        c_d = self.cd0 + self.oswald_k * c_l * c_l

        # Static margin, and therefore the whole short-period character, comes
        # straight from where the CG is. This is the one coupling that makes
        # the runs genuinely different rather than merely offset.
        c_m_alpha = self.cl_alpha * (self.cg_ref - self.x_ac)
        c_m = (self.cm0 + c_m_alpha * alpha
               + self.cm_q * pitch_rate * self.chord / (2.0 * speed)
               + self.cm_de * elevator)

        lift = q_bar * self.wing_area * c_l
        drag = q_bar * self.wing_area * c_d
        moment = q_bar * self.wing_area * self.chord * c_m

        # Thrust falls off with density and rises a little with Mach.
        mach = speed / sound
        thrust = (self.max_thrust * throttle
                  * (density / 1.225) ** 0.7 * (1.0 + 0.18 * mach))
        return temperature, density, sound, q_bar, lift, drag, moment, thrust, mach

    def _derivatives(self, state, elevator, throttle, bank_command):
        altitude, speed, gamma, theta, pitch_rate, psi, phi, x, y, mass = state
        alpha = theta - gamma

        (_t, _rho, _a, _q, lift, drag, moment,
         thrust, _mach) = self._forces(altitude, speed, alpha, pitch_rate,
                                       elevator, throttle)

        speed_safe = np.maximum(speed, 40.0)
        # In a banked turn the vertical component of lift is what holds the
        # aircraft up, so the bank angle feeds straight into the flight-path
        # equation; without this a turning aircraft would not descend.
        cos_phi = np.cos(phi)

        speed_dot = (thrust * np.cos(alpha) - drag) / mass - G0 * np.sin(gamma)
        gamma_dot = ((lift * cos_phi + thrust * np.sin(alpha)
                      - mass * G0 * np.cos(gamma)) / (mass * speed_safe))
        pitch_rate_dot = moment / self.i_yy
        theta_dot = pitch_rate
        psi_dot = G0 * np.tan(np.clip(phi, -1.2, 1.2)) / speed_safe
        phi_dot = 2.2 * (bank_command - phi)          # roll subsidence
        altitude_dot = speed * np.sin(gamma)
        x_dot = speed * np.cos(gamma) * np.cos(psi)
        y_dot = speed * np.cos(gamma) * np.sin(psi)
        mass_dot = -self.tsfc * thrust

        return np.array([altitude_dot, speed_dot, gamma_dot, theta_dot,
                         pitch_rate_dot, psi_dot, phi_dot, x_dot, y_dot,
                         mass_dot])

    # ----------------------------------------------------------- autopilot

    def _commands(self, elapsed: float):
        """Elevator, throttle and bank demand for the current instant.

        A deliberately simple set of loops -- the point is a plausible
        trajectory with real transients, not a certifiable autopilot.
        """
        _temperature, density, sound = isa(self.altitude)
        mach = self.speed / sound
        alpha = self.theta - self.gamma
        # The standard climb schedule: hold an equivalent airspeed until the
        # Mach number catches up, then hold Mach. Targeting the cruise Mach
        # directly from sea level asks for 250 m/s TAS at 450 m, which the
        # aircraft can only get by bunting -- which is exactly what the first
        # version did, and it showed up as a negative load factor at t = 0.
        sigma = np.maximum(density / 1.225, 1e-3)
        target_speed = np.minimum(self.target_eas / np.sqrt(sigma),
                                  self.target_mach * sound)

        # Altitude capture through a commanded flight path, limited so the
        # aircraft rotates onto it rather than snapping. Late in the run the
        # target becomes the ground, so every run has a full climb-cruise-
        # descent profile rather than a climb and then nothing.
        if elapsed > 0.72 * self.total_time:
            altitude_error = 1500.0 - self.altitude
        else:
            altitude_error = self.target_altitude - self.altitude
        gamma_command = np.clip(altitude_error * 0.00055, -0.09, 0.12)

        # Speed protection, and the reason the first version of this model
        # tore itself apart: with the flight path commanded from altitude
        # alone, a climb the engines cannot sustain bleeds the speed away, the
        # lift falls, the loop demands more climb, and the phugoid diverges.
        # Trading commanded climb against speed error is what closes it.
        speed_error = (self.speed - target_speed) / np.maximum(target_speed, 1.0)
        gamma_command = np.clip(
            gamma_command + np.clip(0.55 * speed_error, -0.05, 0.05),
            -0.10, 0.13,
        )

        # Trim first, then correct. Solving cm = 0 for the elevator gives the
        # deflection that holds the current angle of attack, so the feedback
        # term only has to handle the difference -- without it the loop carries
        # a standing error and rings at the phugoid frequency for ever.
        c_m_alpha = self.cl_alpha * (self.cg_ref - self.x_ac)
        elevator_trim = -(self.cm0 + c_m_alpha * alpha) / self.cm_de

        # Rate-limited: the load factor a passenger feels is essentially this
        # number times the speed, so an unlimited pitch-rate demand produces a
        # level-off at 0.1 g, which no aircraft would ever fly.
        pitch_rate_command = np.clip(0.28 * (gamma_command - self.gamma),
                                     -0.022, 0.030)
        # cm_de is negative, so nose-up needs negative elevator.
        elevator = elevator_trim - 1.4 * (pitch_rate_command - self.pitch_rate)

        # Doublets: an abrupt elevator input at fixed times, which is what
        # actually excites the short period. These are the transients the
        # min/max decimation has to preserve -- a stride-sampled viewer loses
        # them completely, and that is precisely the bug this data can catch.
        for start in self._doublet_times:
            if start <= elapsed < start + 1.2:
                elevator = elevator + 0.035
            elif start + 1.2 <= elapsed < start + 2.4:
                elevator = elevator - 0.035

        elevator = np.clip(elevator, -0.30, 0.30)

        throttle = np.clip(
            self.throttle + 0.06 * (target_speed - self.speed) * 0.02,
            0.20, 1.0,
        )

        heading_error = np.arctan2(np.sin(self.target_heading - self.psi),
                                   np.cos(self.target_heading - self.psi))
        bank_command = np.clip(1.2 * heading_error, -0.45, 0.45)

        del mach, density
        return elevator, throttle, bank_command

    # -------------------------------------------------------------- output

    def sample(self, elapsed: float):
        """One row per run, in CHANNELS order."""
        alpha = self.theta - self.gamma
        (temperature, density, sound, q_bar, lift, drag, _moment,
         thrust, mach) = self._forces(self.altitude, self.speed, alpha,
                                      self.pitch_rate, self.elevator,
                                      self.throttle)

        eas = self.speed * np.sqrt(density / 1.225)
        # Load factors in body axes, which is what an accelerometer reads.
        n_z = (lift * np.cos(self.phi) + thrust * np.sin(alpha)) / (self.mass * G0)
        n_x = (thrust * np.cos(alpha) - drag) / (self.mass * G0) - np.sin(self.gamma)
        n_y = np.sin(self.beta) * q_bar * self.wing_area * 0.9 / (self.mass * G0)

        fuel_flow = self.tsfc * thrust
        # The CG creeps aft as the wing tanks empty, which is the real reason a
        # long flight ends less stable than it began.
        cg = self.cg_ref + 0.05 * (self.fuel_used / np.maximum(self.initial_mass * 0.30, 1.0))

        return [
            np.full(self.count, elapsed),
            np.degrees(alpha), np.degrees(self.beta), np.degrees(self.theta),
            np.degrees(self.phi), np.degrees(self.psi), np.degrees(self.gamma),
            np.degrees(self.roll_rate), np.degrees(self.pitch_rate),
            np.degrees(self.yaw_rate),
            self.altitude, self.speed, mach, eas, q_bar,
            temperature, density,
            n_x, n_y, n_z,
            self.mass, cg, fuel_flow, thrust, drag, lift,
            np.degrees(self.elevator), np.degrees(self.aileron),
            np.degrees(self.rudder), self.throttle,
            self.x, self.y, -self.altitude,     # z down, inertial frame
            self.initial_mass, self.cg_ref, self.cl_alpha,
            self.thrust_rating, self.configuration,
        ]

    # ------------------------------------------------------------- stepping

    def prepare(self, total_time: float) -> None:
        self.total_time = total_time
        # Staggered per run would be more realistic but makes the ISO plots
        # harder to read; fixed times let the doublets line up across runs.
        self._doublet_times = [
            0.18 * total_time, 0.34 * total_time,
            0.55 * total_time, 0.80 * total_time,
        ]

    def step(self, dt: float, elapsed: float) -> None:
        """One midpoint (RK2) step.

        RK2 rather than Euler because the short period is the fastest mode in
        the model and Euler visibly adds energy to it at any step long enough
        to be worth using; RK2 at 20 ms is stable and cheap.
        """
        elevator, throttle, bank_command = self._commands(elapsed)
        self.elevator = elevator
        self.throttle = throttle
        self.aileron = 0.25 * (bank_command - self.phi)
        self.rudder = -0.4 * self.beta

        state = np.array([self.altitude, self.speed, self.gamma, self.theta,
                          self.pitch_rate, self.psi, self.phi, self.x, self.y,
                          self.mass])

        k1 = self._derivatives(state, elevator, throttle, bank_command)
        mid = state + 0.5 * dt * k1
        k2 = self._derivatives(mid, elevator, throttle, bank_command)
        state = state + dt * k2

        (self.altitude, self.speed, self.gamma, self.theta, self.pitch_rate,
         self.psi, self.phi, self.x, self.y, self.mass) = state

        self.roll_rate = 2.2 * (bank_command - self.phi)
        # Sideslip: a lightly damped dutch roll excited by rolling, which gives
        # the lateral channels something to show beyond a smooth turn.
        self.beta += dt * (self.yaw_rate - 0.9 * self.beta - 0.12 * self.roll_rate)
        self.yaw_rate += dt * (-1.8 * self.beta - 0.6 * self.yaw_rate
                               + 0.25 * self.roll_rate)
        self.fuel_used = self.initial_mass - self.mass

        self.altitude = np.maximum(self.altitude, 0.0)
        self.speed = np.maximum(self.speed, 45.0)


# ------------------------------------------------------------------ writing


def _prepare_output(path: str) -> None:
    """Make sure `path` can be written, creating its directory if needed.

    Creating the directory rather than complaining about it: the documented
    invocation writes into data/, which is generated output and is in
    .gitignore, so on a fresh checkout it legitimately does not exist yet.
    Failing there would be technically correct and practically useless.
    """
    directory = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        raise SystemExit("Cannot create {0}: {1}".format(directory, exc))

    try:
        # Truncates any previous file, which is what the real write does too,
        # so this cannot succeed here and fail later.
        with open(path, "w", encoding="ascii"):
            pass
    except OSError as exc:
        raise SystemExit("Cannot write {0}: {1}".format(path, exc))


def _measure_row_bytes(runs: int, seed: int, dt: float, probe: int = 200) -> float:
    """Average formatted row length, from a short throwaway integration.

    Cheap -- a couple of hundred steps -- and it makes --size mean what it
    says regardless of which channels are being written or how large the
    numbers happen to be.
    """
    fleet = Fleet(max(1, runs), seed=seed)
    fleet.prepare(probe * dt)
    row_format = " ".join(["%.6g"] * len(CHANNELS)) + "\n"

    total = 0
    counted = 0
    for step_index in range(probe):
        elapsed = step_index * dt
        row = fleet.sample(elapsed)
        # One run per step is a large enough sample and keeps this negligible.
        column = step_index % fleet.count
        total += len(row_format % tuple(float(values[column]) for values in row))
        counted += 1
        fleet.step(dt, elapsed)

    return max(1.0, total / max(1, counted))


def write_file(path: str, runs: int, target_bytes: int, dt: float,
               seed: int, quiet: bool = False) -> None:
    fleet = Fleet(runs, seed=seed)

    # Work out how many samples per run fit the requested size, by measuring
    # rather than guessing. %.6g emits anything from one character to twelve
    # depending on the value, so a fixed estimate was 35% out on this channel
    # set -- asking for 500 MB and getting 328 MB.
    bytes_per_row = _measure_row_bytes(runs, seed, dt)
    total_rows = max(runs * 64, int(target_bytes // bytes_per_row))
    rows_per_run = int(total_rows // runs)
    total_time = rows_per_run * dt
    fleet.prepare(total_time)

    if not quiet:
        print("{0}: {1} runs x {2:,} samples, {3:.0f} s of flight each"
              .format(os.path.basename(path), runs, rows_per_run, total_time))

    # Open the destination *before* integrating. Discovering that data/ does
    # not exist after twenty-two seconds of arithmetic, with the results held
    # in memory and nowhere to put them, is an avoidable way to waste someone's
    # time -- and at 500 MB a run that is most of a minute thrown away.
    _prepare_output(path)

    # Integrate everything first, then write. Holding it costs
    # runs * rows * channels * 8 bytes, which for the default 500 MB target is
    # about 1.4x the file -- acceptable, and it means the writer can format
    # whole blocks at a time instead of a row at a time, which is the
    # difference between minutes and the better part of an hour.
    samples = np.empty((rows_per_run, len(CHANNELS), runs), dtype=np.float64)

    started = time.time()
    for step_index in range(rows_per_run):
        elapsed = step_index * dt
        row = fleet.sample(elapsed)
        for channel_index, values in enumerate(row):
            samples[step_index, channel_index, :] = values
        fleet.step(dt, elapsed)

        if not quiet and step_index and step_index % 20000 == 0:
            print("   integrated {0:,}/{1:,}".format(step_index, rows_per_run))

    if not quiet:
        print("   integration took {0:.1f} s".format(time.time() - started))

    started = time.time()
    row_format = " ".join(["%.6g"] * len(CHANNELS)) + "\n"
    block_rows = 512
    block_format = row_format * block_rows

    with open(path, "w", encoding="ascii", newline="\n") as handle:
        handle.write("*PARAMETRES\n")
        handle.write(" ".join("'{0}'".format(name) for name in CHANNELS))
        handle.write("\n")

        for run in range(runs):
            handle.write("*TABLEAU : {0}\n".format(run + 1))
            block = samples[:, :, run]

            # One % over a flat tuple formats half a million values in a single
            # C-level pass. Doing it row by row in Python is roughly 30x
            # slower and is what makes naive generators unusable at this size.
            full_blocks = block.shape[0] // block_rows
            for index in range(full_blocks):
                chunk = block[index * block_rows:(index + 1) * block_rows]
                handle.write(block_format % tuple(chunk.ravel()))

            remainder = block[full_blocks * block_rows:]
            if remainder.size:
                handle.write((row_format * remainder.shape[0])
                             % tuple(remainder.ravel()))

    if not quiet:
        size = os.path.getsize(path)
        print("   wrote {0:.1f} MB in {1:.1f} s"
              .format(size / (1024 * 1024), time.time() - started))


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("outputs", nargs="+", help="files to write")
    parser.add_argument("--size", type=float, default=500.0,
                        help="approximate size of each file in MB (default 500)")
    parser.add_argument("--runs", type=int, default=25,
                        help="simulations per file (default 25)")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="integration and sampling step, seconds (default 0.02)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.runs < 1:
        print("--runs must be at least 1", file=sys.stderr)
        return 2

    target = int(args.size * 1024 * 1024)
    for index, path in enumerate(args.outputs):
        write_file(path, args.runs, target, args.dt, args.seed + index * 101,
                   quiet=args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
