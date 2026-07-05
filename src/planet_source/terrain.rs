use std::sync::OnceLock;

use bevy::prelude::*;
use fastnoise_lite::{FastNoiseLite, NoiseType};

use super::config::TERRAIN_HEIGHT;

#[derive(Clone, Copy)]
pub(super) enum Biome {
	LushLowlands,
	BasinMeadows,
	DryLowlands,
	GrassUplands,
	WindblownPlateau,
	RockyHighlands,
	AlpineRock,
	ColdPeaks,
}

#[derive(Clone, Copy)]
pub(super) struct TerrainSample {
	pub(super) height: f32,
	pub(super) shade: f32,
	pub(super) moisture: f32,
	pub(super) temperature: f32,
	pub(super) mountainness: f32,
	pub(super) roughness: f32,
	pub(super) steepness: f32,
	pub(super) biome: Biome,
}

#[derive(Clone, Copy)]
struct TerrainFields {
	height: f32,
	shade: f32,
	moisture: f32,
	temperature: f32,
	mountainness: f32,
	roughness: f32,
	plateau_mask: f32,
	basin_mask: f32,
	rounded_weight: f32,
}

pub(super) fn terrain_sample(planet_pos: Vec3) -> TerrainSample {
	const SLOPE_SAMPLE_DISTANCE: f32 = 48.0;

	let center = terrain_fields(planet_pos);
	let dir = planet_pos.normalize_or_zero();
	let (axis_x, axis_y) = tangent_basis(dir);
	let hx0 = terrain_fields((planet_pos + axis_x * SLOPE_SAMPLE_DISTANCE).normalize_or_zero() * planet_pos.length()).height;
	let hx1 = terrain_fields((planet_pos - axis_x * SLOPE_SAMPLE_DISTANCE).normalize_or_zero() * planet_pos.length()).height;
	let hy0 = terrain_fields((planet_pos + axis_y * SLOPE_SAMPLE_DISTANCE).normalize_or_zero() * planet_pos.length()).height;
	let hy1 = terrain_fields((planet_pos - axis_y * SLOPE_SAMPLE_DISTANCE).normalize_or_zero() * planet_pos.length()).height;

	let slope_x = (hx0 - hx1).abs() / (2.0 * SLOPE_SAMPLE_DISTANCE);
	let slope_y = (hy0 - hy1).abs() / (2.0 * SLOPE_SAMPLE_DISTANCE);
	let steepness = ((slope_x * slope_x + slope_y * slope_y).sqrt() * 3.6).clamp(0.0, 1.0);

	let height01 = center.height / TERRAIN_HEIGHT;
	let rock_exposure = (steepness * 0.95
		+ center.roughness * 0.35
		+ center.mountainness * 0.18
		+ (height01 - 0.60).max(0.0) * 0.22
		- center.moisture * 0.14
		- center.rounded_weight * 0.10)
		.clamp(0.0, 1.0);

	let biome = if center.temperature < 0.20 && height01 > 0.80 {
		Biome::ColdPeaks
	} else if rock_exposure > 0.70 && height01 > 0.58 {
		if center.temperature < 0.32 {
			Biome::ColdPeaks
		} else {
			Biome::AlpineRock
		}
	} else if center.plateau_mask > 0.56 && center.moisture < 0.44 {
		Biome::WindblownPlateau
	} else if center.basin_mask > 0.52 && center.moisture > 0.58 && center.temperature > 0.35 {
		Biome::BasinMeadows
	} else if center.moisture > 0.60 {
		if height01 < 0.52 {
			Biome::LushLowlands
		} else {
			Biome::GrassUplands
		}
	} else if height01 > 0.58 || rock_exposure > 0.52 {
		Biome::RockyHighlands
	} else {
		Biome::DryLowlands
	};

	TerrainSample {
		height: center.height,
		shade: center.shade,
		moisture: center.moisture,
		temperature: center.temperature,
		mountainness: center.mountainness,
		roughness: center.roughness,
		steepness,
		biome,
	}
}

fn terrain_fields(planet_pos: Vec3) -> TerrainFields {
	let sample_pos = planet_pos / 512.0;
	let planet_dir = planet_pos.normalize_or_zero();
	let latitude = planet_dir.y.abs();

	let macro0 = sample_noise_3d(sample_pos * 0.035);
	let macro1 = sample_noise_3d(sample_pos * 0.07);
	let c0 = sample_noise_3d(sample_pos * 0.12);
	let c1 = sample_noise_3d(sample_pos * 0.24);
	let c2 = sample_noise_3d(sample_pos * 0.45);
	let n0 = sample_noise_3d(sample_pos * 0.9);
	let n1 = sample_noise_3d(sample_pos * 1.8);
	let n2 = sample_noise_3d(sample_pos * 3.6);

	let moisture_noise = sample_noise_3d(sample_pos * 0.18 + Vec3::splat(37.0));
	let biome_soft_noise = sample_noise_3d(sample_pos * 0.10 + Vec3::splat(91.0));
	let temp_noise = sample_noise_3d(sample_pos * 0.14 + Vec3::splat(151.0));
	let style_noise = sample_noise_3d(sample_pos * 0.08 + Vec3::splat(211.0));
	let plateau_noise = sample_noise_3d(sample_pos * 0.055 + Vec3::splat(271.0));
	let basin_noise = sample_noise_3d(sample_pos * 0.05 + Vec3::splat(331.0));
	let valley_a = sample_noise_3d(sample_pos * 0.55 + Vec3::splat(389.0));
	let valley_b = sample_noise_3d(sample_pos * 1.1 + Vec3::splat(433.0));

	let macro_relief = macro0 * 0.7 + macro1 * 0.3;
	let continent_relief = c0 * 0.55 + c1 * 0.3 + c2 * 0.15;
	let uplift_mask = ((((continent_relief * 0.5 + macro_relief * 0.25) + 0.5) - 0.38).clamp(0.0, 1.0)) / 0.62;

	let style01 = ((style_noise + 1.0) * 0.5).clamp(0.0, 1.0);
	let mut rounded_weight = (1.0 - style01).powf(1.35);
	let mut ridged_weight = style01.powf(1.6);
	let mut broken_weight = (1.0 - (style01 * 2.0 - 1.0).abs()).clamp(0.0, 1.0);
	let style_sum = rounded_weight + ridged_weight + broken_weight + 1e-5;
	rounded_weight /= style_sum;
	ridged_weight /= style_sum;
	broken_weight /= style_sum;

	let plateau_mask = ((((plateau_noise + 1.0) * 0.5) - 0.66).clamp(0.0, 1.0)) / 0.34;
	let basin_mask = (((((-basin_noise) + 1.0) * 0.5) - 0.68).clamp(0.0, 1.0)) / 0.32;

	let ridge_base = 1.0 - n1.abs();
	let ridge_strength = ridge_base * ridge_base * ridge_base;
	let valley_line = (1.0 - valley_a.abs()).max(0.0);
	let valley_mask = valley_line.powf(4.0) * (0.65 + ((valley_b + 1.0) * 0.5).clamp(0.0, 1.0) * 0.35);

	let base_relief = n0 * 0.32 + n1 * 0.14 + n2 * 0.05;
	let rounded_uplift = (n0 * 0.45 + n1 * 0.10 + 0.10).max(0.0);
	let ridged_uplift = (n0 * 0.18 + n1 * 0.10 + ridge_strength * 0.95).max(0.0);
	let broken_uplift = (n0 * 0.28 + n1.abs() * 0.08 + n2.abs() * 0.06 + ridge_strength * 0.25).max(0.0);
	let styled_mountains = rounded_weight * rounded_uplift
		+ broken_weight * broken_uplift
		+ ridged_weight * ridged_uplift;

	let plateau_relief = n0 * 0.12 + n1 * 0.03;
	let relief = base_relief * (1.0 - plateau_mask * 0.65) + plateau_relief * plateau_mask * 0.65;
	let plateau_lift = plateau_mask * 0.18;
	let basin_drop = basin_mask * (0.18 + macro_relief.max(0.0) * 0.06);
	let valley_cut = valley_mask * uplift_mask * (0.08 + ridged_weight * 0.08 + broken_weight * 0.04);

	let combined = macro_relief * 0.30
		+ continent_relief * 0.52
		+ relief
		+ styled_mountains * uplift_mask * 1.10
		+ plateau_lift
		- basin_drop
		- valley_cut;
	let normalized = ((combined * 0.85).tanh() * 0.5 + 0.5).clamp(0.0, 1.0);
	let shaped = normalized.powf(1.08);
	let height = shaped * TERRAIN_HEIGHT;
	let height01 = height / TERRAIN_HEIGHT;

	let mountainness = (uplift_mask * (0.48 + styled_mountains * 0.42) + valley_mask * 0.08).clamp(0.0, 1.0);
	let roughness = (n1.abs() * 0.18
		+ n2.abs() * 0.18
		+ ridge_strength * ridged_weight * 0.32
		+ broken_weight * 0.14
		+ valley_mask * 0.24
		+ mountainness * 0.12)
		.clamp(0.0, 1.0);

	let moisture = ((((moisture_noise * 0.65 + biome_soft_noise * 0.35) + 1.0) * 0.5)
		+ basin_mask * 0.12
		- plateau_mask * 0.08
		- uplift_mask * 0.05)
		.clamp(0.0, 1.0);
	let temperature = (0.82
		- latitude * 0.55
		- height01 * 0.30
		+ temp_noise * 0.10
		+ basin_mask * 0.05
		- plateau_mask * 0.03)
		.clamp(0.0, 1.0);

	TerrainFields {
		height,
		shade: 0.84 + normalized * 0.16,
		moisture,
		temperature,
		mountainness,
		roughness,
		plateau_mask,
		basin_mask,
		rounded_weight,
	}
}

fn tangent_basis(dir: Vec3) -> (Vec3, Vec3) {
	let axis_x = if dir.y.abs() > 0.99 {
		Vec3::X
	} else {
		dir.cross(Vec3::Y).normalize_or_zero()
	};
	let axis_y = dir.cross(axis_x).normalize_or_zero();
	(axis_x, axis_y)
}

pub(super) fn terrain_color(column: TerrainSample, planet_pos: Vec3) -> [u8; 4] {
	let sample_pos = planet_pos / 320.0;
	let variation = sample_noise_3d(sample_pos * 0.45 + Vec3::splat(17.0));
	let cool = column.temperature < 0.32;
	let rough = column.roughness > 0.58;
	let steep = column.steepness > 0.42;
	let very_steep = column.steepness > 0.62;
	let alpine = column.mountainness > 0.62;
	let wet = column.moisture > 0.58;

	let surface_rgb = match column.biome {
		Biome::LushLowlands => {
			if variation > 0.05 {
				hsv_to_rgb(0.30, 0.68, 0.43 * column.shade)
			} else {
				hsv_to_rgb(0.28, 0.72, 0.39 * column.shade)
			}
		}
		Biome::BasinMeadows => {
			if wet && variation > -0.02 {
				hsv_to_rgb(0.32, 0.62, 0.42 * column.shade)
			} else {
				hsv_to_rgb(0.27, 0.58, 0.37 * column.shade)
			}
		}
		Biome::DryLowlands => {
			if variation > 0.08 {
				hsv_to_rgb(0.095, 0.50, 0.40 * column.shade)
			} else {
				hsv_to_rgb(0.085, 0.46, 0.36 * column.shade)
			}
		}
		Biome::GrassUplands => {
			if very_steep {
				hsv_to_rgb(0.13, 0.20, 0.34 * column.shade)
			} else if rough || steep {
				hsv_to_rgb(0.20, 0.34, 0.36 * column.shade)
			} else if variation > 0.02 {
				hsv_to_rgb(0.25, 0.54, 0.39 * column.shade)
			} else {
				hsv_to_rgb(0.23, 0.58, 0.35 * column.shade)
			}
		}
		Biome::WindblownPlateau => {
			if variation > 0.0 {
				hsv_to_rgb(0.11, 0.30, 0.37 * column.shade)
			} else {
				hsv_to_rgb(0.10, 0.26, 0.33 * column.shade)
			}
		}
		Biome::RockyHighlands => {
			if alpine || very_steep {
				hsv_to_rgb(0.080, 0.18, 0.34 * column.shade)
			} else if rough || steep {
				hsv_to_rgb(0.086, 0.20, 0.35 * column.shade)
			} else {
				hsv_to_rgb(0.090, 0.24, 0.36 * column.shade)
			}
		}
		Biome::AlpineRock => {
			if cool || variation > 0.0 {
				hsv_to_rgb(0.072, 0.10, 0.33 * column.shade)
			} else {
				hsv_to_rgb(0.078, 0.12, 0.29 * column.shade)
			}
		}
		Biome::ColdPeaks => {
			if variation > 0.0 {
				hsv_to_rgb(0.060, 0.05, 0.40 * column.shade)
			} else {
				hsv_to_rgb(0.065, 0.07, 0.34 * column.shade)
			}
		}
	};

	[surface_rgb.x as u8, surface_rgb.y as u8, surface_rgb.z as u8, 255]
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Vec3 {
	let h = h * 6.0;
	let i = h.floor();
	let f = h - i;
	let p = v * (1.0 - s);
	let q = v * (1.0 - s * f);
	let t = v * (1.0 - s * (1.0 - f));
	let (r, g, b) = match i as i32 % 6 {
		0 => (v, t, p),
		1 => (q, v, p),
		2 => (p, v, t),
		3 => (p, q, v),
		4 => (t, p, v),
		_ => (v, p, q),
	};
	Vec3::new(r * 255.0, g * 255.0, b * 255.0)
}

fn sample_noise_3d(p: Vec3) -> f32 {
	terrain_noise().get_noise_3d(p.x, p.y, p.z)
}

fn terrain_noise() -> &'static FastNoiseLite {
	static NOISE: OnceLock<FastNoiseLite> = OnceLock::new();
	NOISE.get_or_init(|| {
		let mut noise = FastNoiseLite::with_seed(1337);
		noise.set_frequency(Some(1.0));
		noise.set_noise_type(Some(NoiseType::OpenSimplex2));
		noise
	})
}
