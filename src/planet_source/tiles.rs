use std::f32::consts::PI;
use std::sync::OnceLock;

use bevy::prelude::*;
use tracy_client::span;
use voxel_streaming::{CHUNK_SIZE, chunk_of, chunk_origin};

use super::config::{
	PLANET_RADIUS, PLANET_TILE_COUNT, TILE_BOUND_PADDING, TILE_INWARD_DEPTH, TILE_OUTWARD_HEIGHT,
	TILE_SHAPE_EPSILON, VORONOI_NEIGHBORS,
};

#[derive(Debug, Clone)]
pub(super) struct Halfspace {
	// Local tile coordinates are inside when normal.dot(local) + offset >= 0.
	pub(super) normal: Vec3,
	pub(super) offset: f32,
}

#[derive(Debug, Clone)]
pub(super) struct PlanetTile {
	pub(super) index: usize,
	pub(super) normal: Vec3,
	pub(super) origin: Vec3,
	pub(super) axis_x: Vec3,
	pub(super) axis_y: Vec3,
	pub(super) halfspaces: Vec<Halfspace>,
	pub(super) present_chunks: Vec<IVec3>,
	pub(super) present_areas: Vec<(IVec3, IVec3)>,
	pub(super) present_min: IVec3,
	pub(super) present_max_exclusive: IVec3,
}

pub(super) fn planet_tiles() -> &'static [PlanetTile] {
	static TILES: OnceLock<Vec<PlanetTile>> = OnceLock::new();
	TILES.get_or_init(build_planet_tiles).as_slice()
}

fn build_planet_tiles() -> Vec<PlanetTile> {
	let _zone = span!("planet build tile cache");
	tracy_client::plot!("planet tile count", PLANET_TILE_COUNT as f64);
	let normals: Vec<Vec3> = (0..PLANET_TILE_COUNT)
		.map(|index| fibonacci_sphere_point(index, PLANET_TILE_COUNT))
		.collect();

	let mut tiles = Vec::with_capacity(PLANET_TILE_COUNT);
	for (index, &normal) in normals.iter().enumerate() {
		let axis_x = if normal.x.abs() < 1e-6 && normal.z.abs() < 1e-6 {
			Vec3::X
		} else {
			Vec3::new(-normal.z, 0.0, normal.x).normalize()
		};
		let axis_y = normal.cross(axis_x).normalize();

		let mut neighbor_dots: Vec<(usize, f32)> = normals
			.iter()
			.enumerate()
			.filter(|&(other, _)| other != index)
			.map(|(other, &other_normal)| (other, normal.dot(other_normal)))
			.collect();
		if neighbor_dots.len() > VORONOI_NEIGHBORS {
			neighbor_dots.select_nth_unstable_by(VORONOI_NEIGHBORS, |a, b| b.1.total_cmp(&a.1));
			neighbor_dots.truncate(VORONOI_NEIGHBORS);
		}
		neighbor_dots.sort_by(|a, b| b.1.total_cmp(&a.1));

		// A spherical Voronoi edge between tile A and tile B is the plane where
		// dot(A, point_dir) == dot(B, point_dir). In a tile's local tangent
		// coordinates this is just a linear halfspace, so we can cache the real
		// convex cell once and use it for spawn presence, source costs and voxel
		// ownership. The first ~32 neighbors are plenty for a Fibonacci sphere;
		// farther sites cannot cut this local cell.
		let halfspaces: Vec<_> = neighbor_dots
			.iter()
			.take(VORONOI_NEIGHBORS)
			.map(|&(other, _)| voronoi_halfspace(normal, normals[other], axis_x, axis_y))
			.collect();
		let present_chunks = build_present_chunks(&halfspaces);
		let present_areas = compress_present_chunks(&present_chunks);
		let (present_min, present_max_exclusive) = chunk_bounds(&present_chunks);

		tiles.push(PlanetTile {
			index,
			normal,
			origin: normal * PLANET_RADIUS,
			axis_x,
			axis_y,
			halfspaces,
			present_chunks,
			present_areas,
			present_min,
			present_max_exclusive,
		});
	}

	tracy_client::plot!(
		"planet present chunks total",
		tiles
			.iter()
			.map(|tile| tile.present_chunks.len())
			.sum::<usize>() as f64
	);
	tracy_client::plot!(
		"planet present areas total",
		tiles
			.iter()
			.map(|tile| tile.present_areas.len())
			.sum::<usize>() as f64
	);
	tiles
}

fn fibonacci_sphere_point(index: usize, count: usize) -> Vec3 {
	let i = index as f32 + 0.5;
	let n = count as f32;
	let y = 1.0 - 2.0 * i / n;
	let h = PI * (1.0 + 5.0_f32.sqrt()) * i;
	let radius = (1.0 - y * y).max(0.0).sqrt();
	Vec3::new(h.cos() * radius, y, h.sin() * radius).normalize()
}

fn voronoi_halfspace(
	tile_normal: Vec3,
	neighbor_normal: Vec3,
	axis_x: Vec3,
	axis_y: Vec3,
) -> Halfspace {
	let diff = tile_normal - neighbor_normal;
	Halfspace {
		normal: Vec3::new(diff.dot(axis_x), diff.dot(axis_y), diff.dot(tile_normal)),
		offset: diff.dot(tile_normal * PLANET_RADIUS),
	}
}

fn build_present_chunks(halfspaces: &[Halfspace]) -> Vec<IVec3> {
	let (min_xy, max_xy) = voronoi_xy_bounds(halfspaces);
	let padded_min_xy = min_xy - Vec2::splat(TILE_BOUND_PADDING as f32);
	let padded_max_xy = max_xy + Vec2::splat(TILE_BOUND_PADDING as f32);
	let min_local_z =
		conservative_min_local_z(padded_min_xy, padded_max_xy, -TILE_INWARD_DEPTH as f32).floor()
			as i32
			- CHUNK_SIZE;
	let min_voxel = IVec3::new(
		padded_min_xy.x.floor() as i32,
		padded_min_xy.y.floor() as i32,
		min_local_z,
	);
	let max_voxel_exclusive = IVec3::new(
		padded_max_xy.x.ceil() as i32,
		padded_max_xy.y.ceil() as i32,
		TILE_OUTWARD_HEIGHT,
	);

	let min_chunk = chunk_of(min_voxel);
	let max_chunk = chunk_of(max_voxel_exclusive - IVec3::ONE);
	let mut chunks = Vec::new();
	for x in min_chunk.x..=max_chunk.x {
		for y in min_chunk.y..=max_chunk.y {
			for z in min_chunk.z..=max_chunk.z {
				let chunk = IVec3::new(x, y, z);
				if chunk_intersects_tile_shape(halfspaces, chunk) {
					chunks.push(chunk);
				}
			}
		}
	}
	chunks.sort_by_key(|c| (c.x, c.y, c.z));
	chunks.dedup();
	chunks
}

fn compress_present_chunks(chunks: &[IVec3]) -> Vec<(IVec3, IVec3)> {
	if chunks.is_empty() {
		return Vec::new();
	}

	let mut z_runs = Vec::new();
	let mut i = 0;
	while i < chunks.len() {
		let start = chunks[i];
		let mut z1 = start.z + 1;
		i += 1;
		while i < chunks.len()
			&& chunks[i].x == start.x
			&& chunks[i].y == start.y
			&& chunks[i].z == z1
		{
			z1 += 1;
			i += 1;
		}
		z_runs.push((IVec3::new(start.x, start.y, start.z), IVec3::new(1, 1, z1 - start.z)));
	}

	let mut y_strips = Vec::new();
	let mut i = 0;
	while i < z_runs.len() {
		let (start_min, start_size) = z_runs[i];
		let mut y1 = start_min.y + 1;
		i += 1;
		while i < z_runs.len()
			&& z_runs[i].0.x == start_min.x
			&& z_runs[i].0.y == y1
			&& z_runs[i].0.z == start_min.z
			&& z_runs[i].1.z == start_size.z
		{
			y1 += 1;
			i += 1;
		}
		y_strips.push((
			IVec3::new(start_min.x, start_min.y, start_min.z),
			IVec3::new(1, y1 - start_min.y, start_size.z),
		));
	}

	let mut areas = Vec::new();
	let mut i = 0;
	while i < y_strips.len() {
		let (start_min, start_size) = y_strips[i];
		let mut x1 = start_min.x + 1;
		i += 1;
		while i < y_strips.len()
			&& y_strips[i].0.x == x1
			&& y_strips[i].0.y == start_min.y
			&& y_strips[i].0.z == start_min.z
			&& y_strips[i].1.y == start_size.y
			&& y_strips[i].1.z == start_size.z
		{
			x1 += 1;
			i += 1;
		}
		areas.push((
			IVec3::new(start_min.x, start_min.y, start_min.z),
			IVec3::new(x1 - start_min.x, start_size.y, start_size.z),
		));
	}

	areas
}

fn chunk_bounds(chunks: &[IVec3]) -> (IVec3, IVec3) {
	let Some((&first, rest)) = chunks.split_first() else {
		return (IVec3::ZERO, IVec3::ZERO);
	};
	let (min, max) = rest.iter().fold((first, first), |(min, max), &chunk| {
		(min.min(chunk), max.max(chunk))
	});
	(min, max + IVec3::ONE)
}

fn voronoi_xy_bounds(halfspaces: &[Halfspace]) -> (Vec2, Vec2) {
	let mut min = Vec2::splat(f32::INFINITY);
	let mut max = Vec2::splat(f32::NEG_INFINITY);
	for z in [-TILE_INWARD_DEPTH as f32, TILE_OUTWARD_HEIGHT as f32] {
		let polygon = clipped_voronoi_polygon(halfspaces, z);
		for p in polygon {
			min = min.min(p);
			max = max.max(p);
		}
	}

	if !min.is_finite() || !max.is_finite() {
		// Extremely defensive fallback; this should never happen unless the
		// neighbor list is broken.
		(Vec2::splat(-512.0), Vec2::splat(512.0))
	} else {
		(min, max)
	}
}

fn conservative_min_local_z(min_xy: Vec2, max_xy: Vec2, radial_altitude: f32) -> f32 {
	let max_lateral_sq = [
		min_xy.length_squared(),
		Vec2::new(min_xy.x, max_xy.y).length_squared(),
		Vec2::new(max_xy.x, min_xy.y).length_squared(),
		max_xy.length_squared(),
	]
	.into_iter()
	.fold(0.0, f32::max);
	local_z_for_radial_altitude(max_lateral_sq, radial_altitude).unwrap_or(radial_altitude)
}

fn local_z_for_radial_altitude(lateral_len_sq: f32, radial_altitude: f32) -> Option<f32> {
	let shell_radius = PLANET_RADIUS + radial_altitude;
	let remaining = shell_radius * shell_radius - lateral_len_sq;
	(remaining > 0.0).then_some(remaining.sqrt() - PLANET_RADIUS)
}

fn clipped_voronoi_polygon(halfspaces: &[Halfspace], z: f32) -> Vec<Vec2> {
	let extent = PLANET_RADIUS * 0.25;
	let mut polygon = vec![
		Vec2::new(-extent, -extent),
		Vec2::new(extent, -extent),
		Vec2::new(extent, extent),
		Vec2::new(-extent, extent),
	];

	for halfspace in halfspaces {
		if polygon.is_empty() {
			break;
		}
		let mut clipped = Vec::new();
		let z_offset = halfspace.normal.z * z + halfspace.offset;
		for i in 0..polygon.len() {
			let a = polygon[i];
			let b = polygon[(i + 1) % polygon.len()];
			let va = halfspace.normal.x * a.x + halfspace.normal.y * a.y + z_offset;
			let vb = halfspace.normal.x * b.x + halfspace.normal.y * b.y + z_offset;
			let a_inside = va >= -TILE_SHAPE_EPSILON;
			let b_inside = vb >= -TILE_SHAPE_EPSILON;

			if a_inside && b_inside {
				clipped.push(b);
			} else if a_inside != b_inside {
				let t = (va / (va - vb)).clamp(0.0, 1.0);
				clipped.push(a.lerp(b, t));
				if b_inside {
					clipped.push(b);
				}
			}
		}
		polygon = clipped;
	}
	polygon
}

pub(super) fn tile_has_chunk(tile: &PlanetTile, chunk: IVec3) -> bool {
	tile.present_chunks
		.binary_search_by_key(&(chunk.x, chunk.y, chunk.z), |c| (c.x, c.y, c.z))
		.is_ok()
}

pub(super) fn tile_has_any_chunk_in_region(tile: &PlanetTile, min: IVec3, size: IVec3) -> bool {
	let max = min + size;
	if !aabb_intersects(min, max, tile.present_min, tile.present_max_exclusive) {
		return false;
	}
	if min.cmple(tile.present_min).all() && max.cmpge(tile.present_max_exclusive).all() {
		return !tile.present_chunks.is_empty();
	}
	tile.present_chunks.iter().any(|&chunk| {
		chunk.x >= min.x
			&& chunk.x < max.x
			&& chunk.y >= min.y
			&& chunk.y < max.y
			&& chunk.z >= min.z
			&& chunk.z < max.z
	})
}

fn aabb_intersects(a_min: IVec3, a_max: IVec3, b_min: IVec3, b_max: IVec3) -> bool {
	a_min.x < b_max.x
		&& a_max.x > b_min.x
		&& a_min.y < b_max.y
		&& a_max.y > b_min.y
		&& a_min.z < b_max.z
		&& a_max.z > b_min.z
}

fn chunk_intersects_tile_shape(halfspaces: &[Halfspace], chunk: IVec3) -> bool {
	let min = chunk_origin(chunk).as_vec3();
	let max = (chunk_origin(chunk) + IVec3::splat(CHUNK_SIZE)).as_vec3();
	if min.z >= TILE_OUTWARD_HEIGHT as f32 {
		return false;
	}

	halfspaces.iter().all(|h| {
		// If the furthest AABB vertex in a halfspace's direction is still
		// outside, the whole chunk is outside. This is conservative, so it may
		// keep a few edge chunks but will never crop a valid Voronoi cell.
		let p = Vec3::new(
			if h.normal.x >= 0.0 { max.x } else { min.x },
			if h.normal.y >= 0.0 { max.y } else { min.y },
			if h.normal.z >= 0.0 { max.z } else { min.z },
		);
		h.normal.dot(p) + h.offset >= -CHUNK_SIZE as f32
	})
}
