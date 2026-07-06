use std::sync::OnceLock;

use bevy::prelude::*;
use fastnoise_lite::{FastNoiseLite, NoiseType};

use super::config::TERRAIN_HEIGHT;

static TERRAIN_NOISE: OnceLock<FastNoiseLite> = OnceLock::new();

fn terrain_noise() -> &'static FastNoiseLite {
	TERRAIN_NOISE.get_or_init(|| {
		let mut noise = FastNoiseLite::new();
		noise.set_noise_type(Some(NoiseType::Perlin));
		noise.set_frequency(Some(512.0));
		noise
	})
}

pub(super) fn terrain_height(planet_pos: Vec3) -> f32 {
	let unit_planet_pos = planet_pos.normalize_or_zero();
	let noise = terrain_noise();

	TERRAIN_HEIGHT + noise.get_noise_3d(unit_planet_pos.x, unit_planet_pos.y, unit_planet_pos.z) * 32.0
}
