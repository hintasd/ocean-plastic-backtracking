import os

import numpy as np
import xarray as xr

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_pom_taiwan.nc")


def rankine_vortex(lon, lat, t_hours, center_lon=121.5, center_lat=23.7, radius_km=180.0, max_vel=0.8):
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    dx = (lon2d - center_lon) * np.cos(np.deg2rad(center_lat))
    dy = lat2d - center_lat
    dist_km = np.hypot(dx * 111.0, dy * 111.0)

    r = np.maximum(dist_km, 1e-6)
    speed = np.zeros_like(r, dtype=np.float32)
    mask = r <= radius_km
    speed[mask] = max_vel * (1.0 - r[mask] / radius_km)

    theta = np.arctan2(dy, dx)
    u = -np.sin(theta) * speed
    v = np.cos(theta) * speed

    phase = (t_hours / 24.0) * np.pi * 2.0
    u += 0.12 * np.sin(theta + phase)
    v += 0.10 * np.cos(theta - phase)
    return u.astype(np.float32), v.astype(np.float32)


def taiwan_mask(lon, lat):
    lon2d, lat2d = np.meshgrid(lon, lat, indexing="xy")
    island = (
        (lon2d >= 119.8) & (lon2d <= 122.2)
        & (lat2d >= 21.8) & (lat2d <= 25.5)
        & ~((lon2d > 121.7) & (lat2d < 22.4))
        & ~((lon2d > 121.9) & (lat2d < 22.1))
    )
    return island.astype(np.int8)


def build_mock_dataset(output_path=OUTPUT_PATH):
    lon = np.linspace(119.0, 123.0, 101, dtype=np.float32)
    lat = np.linspace(21.0, 26.0, 81, dtype=np.float32)
    times = np.arange(72, dtype=np.int32) * 3600.0

    u = np.zeros((72, len(lat), len(lon)), dtype=np.float32)
    v = np.zeros((72, len(lat), len(lon)), dtype=np.float32)

    island_mask = taiwan_mask(lon, lat)
    mask = (~island_mask).astype(np.int8)

    for t_idx, t_hours in enumerate(np.arange(72)):
        u_t, v_t = rankine_vortex(lon, lat, t_hours, center_lon=121.4, center_lat=23.5, radius_km=250.0, max_vel=1.2)
        u[t_idx] = u_t
        v[t_idx] = v_t

    ds = xr.Dataset(
        data_vars={
            "u": (("time", "lat", "lon"), u),
            "v": (("time", "lat", "lon"), v),
            "mask_rho": (("lat", "lon"), mask),
            "land_mask": (("lat", "lon"), island_mask.astype(np.int8)),
        },
        coords={
            "time": ("time", times),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
        attrs={"description": "Mock Taiwan POM field for testing backtracking logic"},
    )

    ds["time"].attrs = {"units": "seconds since 2026-01-01 00:00:00 UTC", "calendar": "standard"}
    ds["mask_rho"].attrs = {"long_name": "Land mask"}
    ds["land_mask"].attrs = {"long_name": "Island mask"}

    ds.to_netcdf(output_path, engine="scipy")
    print(f"Created mock NetCDF: {output_path}")
    print(f"Dataset size approx: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    return output_path


if __name__ == "__main__":
    build_mock_dataset()
