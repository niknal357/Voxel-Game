use bevy::prelude::*;
use tracy_client::span;
use voxel_data::voxels::{Voxel, Voxels};
use voxel_streaming::{CHUNK_SIZE, chunk_origin};

use super::config::{PLANET_RADIUS, TILE_INWARD_DEPTH, TILE_SHAPE_EPSILON};
use super::terrain::terrain_height;
use super::tiles::{Halfspace, PlanetTile, planet_tiles};

const HEIGHTMAP_CELLS: usize = CHUNK_SIZE as usize;
const HEIGHTMAP_VERTS: usize = HEIGHTMAP_CELLS + 1;
const TRIANGLE_EPSILON: f32 = 1e-4;

pub(super) fn build_planet_chunk(tile_index: usize, chunk: IVec3) -> Option<Voxels> {
	let _zone = span!("planet build chunk");
	let tile = planet_tiles().get(tile_index)?;
	build_planet_region(tile, chunk_origin(chunk), IVec3::splat(CHUNK_SIZE), 1, true)
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
	build_planet_region(
		tile,
		chunk_origin(min_chunk),
		size_chunks * CHUNK_SIZE,
		step,
		false,
	)
}

fn build_planet_region(
	tile: &PlanetTile,
	origin: IVec3,
	source_size: IVec3,
	step: i32,
	full_mass: bool,
) -> Option<Voxels> {
	let _zone = span!("planet build region from height patch");
	if step <= 0 || source_size.cmple(IVec3::ZERO).any() {
		return None;
	}

	let extent = source_size / step;
	if extent.cmple(IVec3::ZERO).any() || extent.cmpgt(IVec3::splat(CHUNK_SIZE)).any() {
		return None;
	}

	let step_f = step as f32;
	let center_local = origin.as_vec3() + source_size.as_vec3() * 0.5;
	let center_planet = local_to_planet(tile, center_local);
	let planet_normal = planet_normal_in_local(tile, center_planet.normalize_or_zero());
	let (column_axis, column_sign) = best_column_axis(planet_normal);
	let (u_axis, v_axis) = cross_axes(column_axis);

	let heightmap = build_heightmap_patch(tile, origin, source_size, step_f);
	let triangles = build_terrain_triangles(&heightmap, column_axis, u_axis, v_axis);
	if triangles.is_empty() {
		return None;
	}

	let buckets = bucket_triangles(&triangles, origin, extent, step_f, u_axis, v_axis);
	let center_local = planet_center_in_local(tile);
	let color = [128, 128, 128, 255];
	let mass = if full_mass { 100 } else { 0 };
	let voxel = Voxel { color, mass };
	let mut areas = Vec::new();
	let mut columns_tested = 0usize;
	let mut columns_with_surface = 0usize;
	let mut columns_emitted = 0usize;
	let mut triangle_tests = 0usize;

	let u_extent = axis_i32(extent, u_axis);
	let v_extent = axis_i32(extent, v_axis);
	for v in 0..v_extent {
		let sample_v = axis_i32(origin, v_axis) as f32 + (v as f32 + 0.5) * step_f;
		for u in 0..u_extent {
			columns_tested += 1;
			let sample_u = axis_i32(origin, u_axis) as f32 + (u as f32 + 0.5) * step_f;
			let bucket_index = (v * u_extent + u) as usize;
			let candidates = &buckets[bucket_index];
			triangle_tests += candidates.len();

			let Some(terrain_top_t) = terrain_top_for_column(
				&triangles,
				candidates,
				sample_u,
				sample_v,
				column_sign,
			) else {
				continue;
			};
			columns_with_surface += 1;

			let Some((range_min_t, range_max_t)) = clipped_column_range(
				tile,
				origin,
				source_size,
				step_f,
				column_axis,
				column_sign,
				u_axis,
				v_axis,
				sample_u,
				sample_v,
			) else {
				continue;
			};

			let p0 = column_base(column_axis, u_axis, v_axis, sample_u, sample_v);
			let dir = axis_dir(column_axis, column_sign);
			let inner_radius = PLANET_RADIUS - TILE_INWARD_DEPTH as f32;
			let bottom_t = sphere_exit_t(p0, dir, center_local, inner_radius).unwrap_or(range_min_t);
			let solid_min_t = range_min_t.max(bottom_t);
			let solid_max_t = range_max_t.min(terrain_top_t);

			let Some((a0, a1)) = output_axis_range(
				origin,
				extent,
				step_f,
				column_axis,
				column_sign,
				solid_min_t,
				solid_max_t,
			) else {
				continue;
			};

			let mut pos = IVec3::ZERO;
			set_axis_i32(&mut pos, column_axis, a0);
			set_axis_i32(&mut pos, u_axis, u);
			set_axis_i32(&mut pos, v_axis, v);

			let mut size = IVec3::ONE;
			set_axis_i32(&mut size, column_axis, a1 - a0);

			areas.push((pos.as_u16vec3(), size.as_u16vec3(), voxel));
			columns_emitted += 1;
		}
	}

	tracy_client::plot!("planet sample extent x", extent.x as f64);
	tracy_client::plot!("planet sample extent y", extent.y as f64);
	tracy_client::plot!("planet sample extent z", extent.z as f64);
	tracy_client::plot!("planet height samples", (HEIGHTMAP_VERTS * HEIGHTMAP_VERTS) as f64);
	tracy_client::plot!("planet terrain triangles", triangles.len() as f64);
	tracy_client::plot!("planet columns tested", columns_tested as f64);
	tracy_client::plot!("planet columns with surface", columns_with_surface as f64);
	tracy_client::plot!("planet columns emitted", columns_emitted as f64);
	tracy_client::plot!("planet triangle tests", triangle_tests as f64);
	tracy_client::plot!("planet append emitted areas", areas.len() as f64);

	if areas.is_empty() {
		None
	} else {
		let mut voxels = Voxels::new();
		voxels.add_areas(&areas);
		Some(voxels)
	}
}

fn build_heightmap_patch(
	tile: &PlanetTile,
	origin: IVec3,
	source_size: IVec3,
	step: f32,
) -> Vec<Vec3> {
	let _zone = span!("planet build height patch");
	let center_local = origin.as_vec3() + source_size.as_vec3() * 0.5;
	let mut patch_normal = local_to_planet(tile, center_local).normalize_or_zero();
	if patch_normal.length_squared() < 0.5 {
		patch_normal = tile.site_normal;
	}
	let patch_center = patch_normal * PLANET_RADIUS;
	let (patch_x, patch_y) = tangent_basis(patch_normal);
	let (mut min_uv, mut max_uv) = projected_surface_bounds(
		tile,
		origin.as_vec3(),
		(origin + source_size).as_vec3(),
		patch_center,
		patch_x,
		patch_y,
	);

	let padding = (source_size.as_vec3().length() * 0.03).max(step * 2.0 + 2.0);
	min_uv -= Vec2::splat(padding);
	max_uv += Vec2::splat(padding);
	if !min_uv.is_finite() || !max_uv.is_finite() || min_uv.cmpge(max_uv).any() {
		let fallback = source_size.as_vec3().length().max(CHUNK_SIZE as f32) * 0.5 + padding;
		min_uv = Vec2::splat(-fallback);
		max_uv = Vec2::splat(fallback);
	}

	let mut vertices = Vec::with_capacity(HEIGHTMAP_VERTS * HEIGHTMAP_VERTS);
	for y in 0..HEIGHTMAP_VERTS {
		let ty = y as f32 / HEIGHTMAP_CELLS as f32;
		let uv_y = min_uv.y.lerp(max_uv.y, ty);
		for x in 0..HEIGHTMAP_VERTS {
			let tx = x as f32 / HEIGHTMAP_CELLS as f32;
			let uv_x = min_uv.x.lerp(max_uv.x, tx);
			let surface_dir = (patch_center + patch_x * uv_x + patch_y * uv_y).normalize_or_zero();
			let surface_pos = surface_dir * PLANET_RADIUS;
			let height = terrain_height(surface_pos);
			vertices.push(planet_to_local(tile, surface_dir * (PLANET_RADIUS + height)));
		}
	}
	vertices
}

fn projected_surface_bounds(
	tile: &PlanetTile,
	min: Vec3,
	max: Vec3,
	patch_center: Vec3,
	patch_x: Vec3,
	patch_y: Vec3,
) -> (Vec2, Vec2) {
	let mut min_uv = Vec2::splat(f32::INFINITY);
	let mut max_uv = Vec2::splat(f32::NEG_INFINITY);
	for x in [min.x, max.x] {
		for y in [min.y, max.y] {
			for z in [min.z, max.z] {
				let planet = local_to_planet(tile, Vec3::new(x, y, z));
				let dir = planet.normalize_or_zero();
				if dir.length_squared() < 0.5 {
					continue;
				}
				let delta = dir * PLANET_RADIUS - patch_center;
				let uv = Vec2::new(delta.dot(patch_x), delta.dot(patch_y));
				min_uv = min_uv.min(uv);
				max_uv = max_uv.max(uv);
			}
		}
	}
	(min_uv, max_uv)
}

#[derive(Debug, Clone)]
struct TerrainTriangle {
	u: [f32; 3],
	v: [f32; 3],
	q: [f32; 3],
	area: f32,
}

fn build_terrain_triangles(
	vertices: &[Vec3],
	column_axis: usize,
	u_axis: usize,
	v_axis: usize,
) -> Vec<TerrainTriangle> {
	let _zone = span!("planet build terrain triangles");
	let mut triangles = Vec::with_capacity(HEIGHTMAP_CELLS * HEIGHTMAP_CELLS * 2);
	for y in 0..HEIGHTMAP_CELLS {
		for x in 0..HEIGHTMAP_CELLS {
			let i00 = heightmap_index(x, y);
			let i10 = heightmap_index(x + 1, y);
			let i01 = heightmap_index(x, y + 1);
			let i11 = heightmap_index(x + 1, y + 1);
			push_triangle(
				&mut triangles,
				[vertices[i00], vertices[i10], vertices[i11]],
				column_axis,
				u_axis,
				v_axis,
			);
			push_triangle(
				&mut triangles,
				[vertices[i00], vertices[i11], vertices[i01]],
				column_axis,
				u_axis,
				v_axis,
			);
		}
	}
	triangles
}

fn heightmap_index(x: usize, y: usize) -> usize {
	y * HEIGHTMAP_VERTS + x
}

fn push_triangle(
	triangles: &mut Vec<TerrainTriangle>,
	points: [Vec3; 3],
	column_axis: usize,
	u_axis: usize,
	v_axis: usize,
) {
	let u = points.map(|p| axis_f32(p, u_axis));
	let v = points.map(|p| axis_f32(p, v_axis));
	let q = points.map(|p| axis_f32(p, column_axis));
	let area = edge_cross_2d(u[1] - u[0], v[1] - v[0], u[2] - u[0], v[2] - v[0]);
	if area.abs() <= TRIANGLE_EPSILON {
		return;
	}
	triangles.push(TerrainTriangle { u, v, q, area });
}

fn bucket_triangles(
	triangles: &[TerrainTriangle],
	origin: IVec3,
	extent: IVec3,
	step: f32,
	u_axis: usize,
	v_axis: usize,
) -> Vec<Vec<usize>> {
	let _zone = span!("planet bucket terrain triangles");
	let u_extent = axis_i32(extent, u_axis);
	let v_extent = axis_i32(extent, v_axis);
	let mut buckets = vec![Vec::new(); (u_extent * v_extent) as usize];
	let origin_u = axis_i32(origin, u_axis) as f32;
	let origin_v = axis_i32(origin, v_axis) as f32;

	for (triangle_index, tri) in triangles.iter().enumerate() {
		let min_u = tri.u.into_iter().fold(f32::INFINITY, f32::min);
		let max_u = tri.u.into_iter().fold(f32::NEG_INFINITY, f32::max);
		let min_v = tri.v.into_iter().fold(f32::INFINITY, f32::min);
		let max_v = tri.v.into_iter().fold(f32::NEG_INFINITY, f32::max);
		let Some((u0, u1)) = bucket_range(min_u, max_u, origin_u, step, u_extent) else {
			continue;
		};
		let Some((v0, v1)) = bucket_range(min_v, max_v, origin_v, step, v_extent) else {
			continue;
		};

		for v in v0..=v1 {
			for u in u0..=u1 {
				buckets[(v * u_extent + u) as usize].push(triangle_index);
			}
		}
	}
	buckets
}

fn bucket_range(min: f32, max: f32, origin: f32, step: f32, extent: i32) -> Option<(i32, i32)> {
	if max < origin || min >= origin + extent as f32 * step {
		return None;
	}
	let i0 = ((min - origin) / step).floor() as i32;
	let i1 = ((max - origin) / step).floor() as i32;
	Some((i0.clamp(0, extent - 1), i1.clamp(0, extent - 1)))
}

fn terrain_top_for_column(
	triangles: &[TerrainTriangle],
	candidates: &[usize],
	u: f32,
	v: f32,
	column_sign: f32,
) -> Option<f32> {
	let mut top_t = f32::NEG_INFINITY;
	for &triangle_index in candidates {
		let tri = &triangles[triangle_index];
		let Some((w0, w1, w2)) = barycentric_2d(tri, u, v) else {
			continue;
		};
		let q = w0 * tri.q[0] + w1 * tri.q[1] + w2 * tri.q[2];
		top_t = top_t.max(column_sign * q);
	}
	top_t.is_finite().then_some(top_t)
}

fn barycentric_2d(tri: &TerrainTriangle, u: f32, v: f32) -> Option<(f32, f32, f32)> {
	let du = u - tri.u[0];
	let dv = v - tri.v[0];
	let w1 = edge_cross_2d(du, dv, tri.u[2] - tri.u[0], tri.v[2] - tri.v[0]) / tri.area;
	let w2 = edge_cross_2d(tri.u[1] - tri.u[0], tri.v[1] - tri.v[0], du, dv) / tri.area;
	let w0 = 1.0 - w1 - w2;
	(w0 >= -TRIANGLE_EPSILON && w1 >= -TRIANGLE_EPSILON && w2 >= -TRIANGLE_EPSILON)
		.then_some((w0, w1, w2))
}

fn clipped_column_range(
	tile: &PlanetTile,
	origin: IVec3,
	source_size: IVec3,
	step: f32,
	column_axis: usize,
	column_sign: f32,
	u_axis: usize,
	v_axis: usize,
	sample_u: f32,
	sample_v: f32,
) -> Option<(f32, f32)> {
	let origin_q = axis_i32(origin, column_axis) as f32;
	let max_q = origin_q + axis_i32(source_size, column_axis) as f32;
	let (mut min_t, mut max_t) = if column_sign > 0.0 {
		(origin_q, max_q)
	} else {
		(-max_q, -origin_q)
	};

	let p0 = column_base(column_axis, u_axis, v_axis, sample_u, sample_v);
	let dir = axis_dir(column_axis, column_sign);
	let padding = step * 1.25;
	clip_halfspaces(&mut min_t, &mut max_t, p0, dir, &tile.halfspaces, padding)?;
	(min_t < max_t).then_some((min_t, max_t))
}

fn clip_halfspaces(
	min_t: &mut f32,
	max_t: &mut f32,
	p0: Vec3,
	dir: Vec3,
	halfspaces: &[Halfspace],
	padding: f32,
) -> Option<()> {
	for h in halfspaces {
		let base = h.normal.dot(p0) + h.offset;
		let slope = h.normal.dot(dir);
		let epsilon = TILE_SHAPE_EPSILON
			+ padding * (h.normal.x.abs() + h.normal.y.abs() + h.normal.z.abs());
		if slope > 1e-6 {
			*min_t = (*min_t).max((-epsilon - base) / slope);
		} else if slope < -1e-6 {
			*max_t = (*max_t).min((-epsilon - base) / slope);
		} else if base < -epsilon {
			return None;
		}
		if *min_t >= *max_t {
			return None;
		}
	}
	Some(())
}

fn output_axis_range(
	origin: IVec3,
	extent: IVec3,
	step: f32,
	column_axis: usize,
	column_sign: f32,
	min_t: f32,
	max_t: f32,
) -> Option<(i32, i32)> {
	if min_t >= max_t {
		return None;
	}
	let (min_q, max_q) = if column_sign > 0.0 {
		(min_t, max_t)
	} else {
		(-max_t, -min_t)
	};
	let origin_q = axis_i32(origin, column_axis) as f32;
	let center0 = origin_q + step * 0.5;
	let i0 = ((min_q - center0) / step).ceil() as i32;
	let i1 = ((max_q - center0) / step).ceil() as i32;
	let extent_q = axis_i32(extent, column_axis);
	let i0 = i0.clamp(0, extent_q);
	let i1 = i1.clamp(0, extent_q);
	(i0 < i1).then_some((i0, i1))
}

fn sphere_exit_t(p0: Vec3, dir: Vec3, center: Vec3, radius: f32) -> Option<f32> {
	let m = p0 - center;
	let b = m.dot(dir);
	let c = m.length_squared() - radius * radius;
	let discriminant = b * b - c;
	(discriminant >= 0.0).then_some(-b + discriminant.sqrt())
}

fn best_column_axis(normal: Vec3) -> (usize, f32) {
	let x = normal.x.abs();
	let y = normal.y.abs();
	let z = normal.z.abs();
	if x >= y && x >= z {
		(0, normal.x.signum().max(0.0) * 2.0 - 1.0)
	} else if y >= z {
		(1, normal.y.signum().max(0.0) * 2.0 - 1.0)
	} else {
		(2, normal.z.signum().max(0.0) * 2.0 - 1.0)
	}
}

fn cross_axes(axis: usize) -> (usize, usize) {
	match axis {
		0 => (1, 2),
		1 => (0, 2),
		_ => (0, 1),
	}
}

fn column_base(column_axis: usize, u_axis: usize, v_axis: usize, sample_u: f32, sample_v: f32) -> Vec3 {
	let mut p = Vec3::ZERO;
	set_axis_f32(&mut p, column_axis, 0.0);
	set_axis_f32(&mut p, u_axis, sample_u);
	set_axis_f32(&mut p, v_axis, sample_v);
	p
}

fn axis_dir(axis: usize, sign: f32) -> Vec3 {
	match axis {
		0 => Vec3::X * sign,
		1 => Vec3::Y * sign,
		_ => Vec3::Z * sign,
	}
}

fn planet_normal_in_local(tile: &PlanetTile, normal: Vec3) -> Vec3 {
	Vec3::new(
		normal.dot(tile.axis_x),
		normal.dot(tile.axis_y),
		normal.dot(tile.axis_z),
	)
}

fn local_to_planet(tile: &PlanetTile, p: Vec3) -> Vec3 {
	tile.origin + tile.axis_x * p.x + tile.axis_y * p.y + tile.axis_z * p.z
}

fn planet_to_local(tile: &PlanetTile, p: Vec3) -> Vec3 {
	let d = p - tile.origin;
	Vec3::new(d.dot(tile.axis_x), d.dot(tile.axis_y), d.dot(tile.axis_z))
}

fn planet_center_in_local(tile: &PlanetTile) -> Vec3 {
	planet_to_local(tile, Vec3::ZERO)
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

fn edge_cross_2d(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
	ax * by - ay * bx
}

fn axis_f32(v: Vec3, axis: usize) -> f32 {
	match axis {
		0 => v.x,
		1 => v.y,
		_ => v.z,
	}
}

fn axis_i32(v: IVec3, axis: usize) -> i32 {
	match axis {
		0 => v.x,
		1 => v.y,
		_ => v.z,
	}
}

fn set_axis_f32(v: &mut Vec3, axis: usize, value: f32) {
	match axis {
		0 => v.x = value,
		1 => v.y = value,
		_ => v.z = value,
	}
}

fn set_axis_i32(v: &mut IVec3, axis: usize, value: i32) {
	match axis {
		0 => v.x = value,
		1 => v.y = value,
		_ => v.z = value,
	}
}
