use bevy::math::U16Vec3;
use bevy::prelude::*;
use tracy_client::span;
use voxel_data::voxels::{Voxel, Voxels};
use voxel_streaming::{CHUNK_SIZE, chunk_origin};

use super::config::{PLANET_RADIUS, TILE_INWARD_DEPTH, TILE_OUTWARD_HEIGHT, TILE_SHAPE_EPSILON};
use super::terrain::{terrain_color, terrain_sample};
use super::tiles::{PlanetTile, planet_tiles};

pub(super) fn build_planet_chunk(tile_index: usize, chunk: IVec3) -> Option<Voxels> {
	let _zone = span!("planet build chunk");
	let tile = planet_tiles().get(tile_index)?;
	let origin = chunk_origin(chunk);
	let mut points = Vec::new();
	append_planet_samples(
		tile,
		origin,
		IVec3::splat(CHUNK_SIZE),
		1,
		0,
		true,
		&mut points,
	);
	tracy_client::plot!("planet chunk emitted voxels", points.len() as f64);
	points_to_voxels(points)
}

pub(super) fn build_planet_lod_region(
	tile_index: usize,
	min_chunk: IVec3,
	size_chunks: IVec3,
	lod: f32,
) -> Option<Voxels> {
	let _zone = span!("planet build lod region");
	let tile = planet_tiles().get(tile_index)?;
	let step = 1i32 << lod.max(0.0).floor() as u32;
	let sample_offset = step / 2;
	let extent = (size_chunks * CHUNK_SIZE) / step;
	let origin = chunk_origin(min_chunk);
	let mut points = Vec::new();
	tracy_client::plot!("planet lod step", step as f64);
	tracy_client::plot!(
		"planet lod source chunks",
		(size_chunks.x * size_chunks.y * size_chunks.z) as f64
	);
	append_planet_samples(
		tile,
		origin,
		extent,
		step,
		sample_offset,
		false,
		&mut points,
	);
	tracy_client::plot!("planet lod emitted voxels", points.len() as f64);
	points_to_voxels(points)
}

fn points_to_voxels(points: Vec<(U16Vec3, Voxel)>) -> Option<Voxels> {
	let _zone = span!("planet points to voxels");
	tracy_client::plot!("planet points to voxels input", points.len() as f64);
	if points.is_empty() {
		None
	} else {
		let mut voxels = Voxels::new();
		voxels.add_voxels(&points);
		Some(voxels)
	}
}

fn append_planet_samples(
	tile: &PlanetTile,
	origin: IVec3,
	extent: IVec3,
	step: i32,
	sample_offset: i32,
	full_mass: bool,
	points: &mut Vec<(U16Vec3, Voxel)>,
) {
	let _zone = span!("planet append samples");
	let mass = if full_mass { 100 } else { 0 };
	let step_f = step as f32;
	let sample_base_z = origin.z as f32 + sample_offset as f32 + 0.5;

	let mut columns_tested = 0usize;
	let mut columns_in_shape = 0usize;
	let mut z_candidates = 0usize;
	let mut terrain_samples = 0usize;
	let start_points = points.len();

	tracy_client::plot!("planet sample extent x", extent.x as f64);
	tracy_client::plot!("planet sample extent y", extent.y as f64);
	tracy_client::plot!("planet sample extent z", extent.z as f64);

	for y in 0..extent.y {
		let sample_y = (origin.y + y * step + sample_offset) as f32 + 0.5;
		for x in 0..extent.x {
			columns_tested += 1;
			let sample_x = (origin.x + x * step + sample_offset) as f32 + 0.5;
			let Some((z0, z1)) = column_shape_z_range(
				tile,
				sample_x,
				sample_y,
				sample_base_z,
				extent.z,
				step_f,
				step_f * 1.25,
			) else {
				continue;
			};
			columns_in_shape += 1;
			z_candidates += (z1 - z0) as usize;

			let surface_planet_pos = sample_surface_planet_pos(tile, sample_x, sample_y);
			let terrain = terrain_sample(surface_planet_pos);
			terrain_samples += 1;
			let color = terrain_color(terrain, surface_planet_pos);
			let z1 = z1.min(column_terrain_top_z(
				sample_x,
				sample_y,
				sample_base_z,
				step_f,
				extent.z,
				terrain.height,
			));

			for z in z0..z1 {
				points.push((IVec3::new(x, y, z).as_u16vec3(), Voxel { color, mass }));
			}
		}
	}

	tracy_client::plot!("planet columns tested", columns_tested as f64);
	tracy_client::plot!("planet columns in shape", columns_in_shape as f64);
	tracy_client::plot!("planet z candidates", z_candidates as f64);
	tracy_client::plot!("planet terrain samples", terrain_samples as f64);
	tracy_client::plot!(
		"planet append emitted voxels",
		(points.len() - start_points) as f64
	);
}

fn column_shape_z_range(
	tile: &PlanetTile,
	sample_x: f32,
	sample_y: f32,
	sample_base_z: f32,
	extent_z: i32,
	step: f32,
	padding: f32,
) -> Option<(i32, i32)> {
	let mut min_sample_z =
		local_z_from_radial_altitude(sample_x, sample_y, -TILE_INWARD_DEPTH as f32)?;
	let mut max_sample_z =
		local_z_from_radial_altitude(sample_x, sample_y, TILE_OUTWARD_HEIGHT as f32)?;

	for h in &tile.halfspaces {
		let base = h.normal.x * sample_x + h.normal.y * sample_y + h.offset;
		let epsilon =
			TILE_SHAPE_EPSILON + padding * (h.normal.x.abs() + h.normal.y.abs() + h.normal.z.abs());
		if h.normal.z > 1e-6 {
			min_sample_z = min_sample_z.max((-epsilon - base) / h.normal.z);
		} else if h.normal.z < -1e-6 {
			max_sample_z = max_sample_z.min((-epsilon - base) / h.normal.z);
		} else if base < -epsilon {
			return None;
		}
	}

	if min_sample_z >= max_sample_z {
		return None;
	}

	let z0 = (((min_sample_z - sample_base_z) / step).ceil() as i32).clamp(0, extent_z);
	let z1 = (((max_sample_z - sample_base_z) / step).ceil() as i32).clamp(0, extent_z);
	(z0 < z1).then_some((z0, z1))
}

fn sample_surface_planet_pos(tile: &PlanetTile, x: f32, y: f32) -> Vec3 {
	let lateral = tile.axis_x * x + tile.axis_y * y;
	(lateral + tile.normal * PLANET_RADIUS).normalize_or_zero() * PLANET_RADIUS
}

fn local_z_from_radial_altitude(x: f32, y: f32, radial_altitude: f32) -> Option<f32> {
	let shell_radius = PLANET_RADIUS + radial_altitude;
	let lateral_len_sq = x * x + y * y;
	let remaining = shell_radius * shell_radius - lateral_len_sq;
	(remaining > 0.0).then_some(remaining.sqrt() - PLANET_RADIUS)
}

fn column_terrain_top_z(
	sample_x: f32,
	sample_y: f32,
	sample_base_z: f32,
	step: f32,
	extent_z: i32,
	terrain_height: f32,
) -> i32 {
	let Some(max_sample_z) = local_z_from_radial_altitude(sample_x, sample_y, terrain_height)
	else {
		return 0;
	};
	(((max_sample_z - sample_base_z) / step).ceil() as i32).clamp(0, extent_z)
}
