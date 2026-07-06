use std::sync::OnceLock;

use bevy::prelude::*;
use fastnoise_lite::{FastNoiseLite, NoiseType};

use super::config::TERRAIN_HEIGHT;

pub(super) fn terrain_height(_planet_pos: Vec3) -> f32 {
	TERRAIN_HEIGHT
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
