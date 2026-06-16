use std::f32::consts::PI;
use std::sync::OnceLock;

use bevy::math::I16Vec3;
use bevy::prelude::*;
use voxel_data::grid::Grid;
use voxel_data::voxels::{Voxel, Voxels};
use voxel_edit::GridEdits;
use voxel_physics::{IsStatic, RigidBody, components::VoxelCollider};
use voxel_sources::{ChunkSource, GridKey, SourceHandle, VoxelSourcesAppExt};
use voxel_streaming::{CHUNK_SIZE, GridStreaming, chunk_of, chunk_origin};

const PLANET_GRID_BASE: u64 = 10000;
const PLANET_TILE_COUNT: usize = 1024;
const PLANET_RADIUS: f32 = 4096.0;
const PLANET_COST: u32 = 1;

// Each tangent grid is clipped by the spherical Voronoi cell around its
// Fibonacci point.  These are only radial limits; x/y bounds are inferred from
// the cached Voronoi halfspaces per tile.
const TILE_INWARD_DEPTH: i32 = 64;
const TILE_OUTWARD_HEIGHT: i32 = 128;
const TILE_BOUND_PADDING: i32 = CHUNK_SIZE * 2;
const VORONOI_NEIGHBORS: usize = 32;
const TERRAIN_HEIGHT: f32 = 80.0;
const TILE_SHAPE_EPSILON: f32 = 0.001;

#[derive(Clone)]
struct Halfspace {
    // Local tile coordinates are inside when normal.dot(local) + offset >= 0.
    normal: Vec3,
    offset: f32,
}

#[derive(Clone)]
struct PlanetTile {
    index: usize,
    normal: Vec3,
    origin: Vec3,
    axis_x: Vec3,
    axis_y: Vec3,
    halfspaces: Vec<Halfspace>,
    present_chunks: Vec<IVec3>,
    present_min: IVec3,
    present_max_exclusive: IVec3,
    tint: [u8; 3],
}

pub struct ProceduralPlanetPlugin;

impl Plugin for ProceduralPlanetPlugin {
    fn build(&self, app: &mut App) {
        // Force the Fibonacci sphere and neighbor lists to be built once on the
        // main thread instead of the first streaming worker that asks for a chunk.
        let _ = planet_tiles();
        app.register_source(ProceduralPlanetSource::default())
            .add_systems(Startup, spawn_planet);
    }
}

#[derive(Default)]
struct ProceduralPlanetSource {
    handle: OnceLock<SourceHandle>,
}

impl ChunkSource for ProceduralPlanetSource {
    fn init(&self, handle: SourceHandle) {
        let _ = self.handle.set(handle);
    }

    fn cost(&self, grid: GridKey, chunk: IVec3) -> Option<u32> {
        let tile = planet_tiles().get(tile_index(grid)?)?;
        tile_has_chunk(tile, chunk).then_some(PLANET_COST)
    }

    fn request_load(&self, grid: GridKey, chunk: IVec3) {
        let voxels = build_planet_chunk(grid, chunk);
        if let Some(handle) = self.handle.get() {
            handle.loaded(grid, chunk, voxels);
        }
    }

    fn cost_lod(&self, grid: GridKey, min: IVec3, size: IVec3, _lod: f32) -> Option<u32> {
        let tile = planet_tiles().get(tile_index(grid)?)?;
        tile_has_any_chunk_in_region(tile, min, size).then_some(PLANET_COST)
    }

    fn request_load_lod(&self, grid: GridKey, min: IVec3, size: IVec3, lod: f32) {
        let voxels = build_planet_lod_region(grid, min, size, lod);
        if let Some(handle) = self.handle.get() {
            handle.loaded_lod(grid, min, size, lod, voxels);
        }
    }
}

fn spawn_planet(mut commands: Commands) {
    let parent = commands
        .spawn((RigidBody, IsStatic, Transform::IDENTITY))
        .id();

    for tile in planet_tiles() {
        let key = grid_key(tile.index);
        let mut streaming = GridStreaming::default();
        for &chunk in &tile.present_chunks {
            streaming.presence_mut().mark_present(chunk);
        }

        let rotation = Quat::from_mat3(&Mat3::from_cols(tile.axis_x, tile.axis_y, tile.normal));
        let transform = Transform {
            translation: tile.origin,
            rotation,
            scale: Vec3::ONE,
        };

        let grid_entity = commands
            .spawn((
                transform,
                Grid::new(),
                VoxelCollider,
                GridEdits::default(),
                key,
                streaming,
            ))
            .id();
        commands.entity(parent).add_child(grid_entity);
    }
}

fn planet_tiles() -> &'static [PlanetTile] {
    static TILES: OnceLock<Vec<PlanetTile>> = OnceLock::new();
    TILES.get_or_init(build_planet_tiles).as_slice()
}

fn build_planet_tiles() -> Vec<PlanetTile> {
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
        let tint = tile_tint(index);

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
        // dot(A, point_dir) == dot(B, point_dir).  In a tile's local tangent
        // coordinates this is just a linear halfspace, so we can cache the real
        // convex cell once and use it for spawn presence, source costs and voxel
        // ownership.  The first ~32 neighbors are plenty for a Fibonacci sphere;
        // farther sites cannot cut this local cell.
        let halfspaces: Vec<_> = neighbor_dots
            .iter()
            .take(VORONOI_NEIGHBORS)
            .map(|&(other, _)| voronoi_halfspace(normal, normals[other], axis_x, axis_y))
            .collect();
        let present_chunks = build_present_chunks(&halfspaces);
        let (present_min, present_max_exclusive) = chunk_bounds(&present_chunks);

        tiles.push(PlanetTile {
            index,
            normal,
            origin: normal * PLANET_RADIUS,
            axis_x,
            axis_y,
            halfspaces,
            present_chunks,
            present_min,
            present_max_exclusive,
            tint,
        });
    }

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

fn grid_key(index: usize) -> GridKey {
    GridKey(PLANET_GRID_BASE + index as u64)
}

fn tile_index(grid: GridKey) -> Option<usize> {
    let index = grid.0.checked_sub(PLANET_GRID_BASE)? as usize;
    (index < PLANET_TILE_COUNT).then_some(index)
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
    let min_voxel = IVec3::new(
        min_xy.x.floor() as i32 - TILE_BOUND_PADDING,
        min_xy.y.floor() as i32 - TILE_BOUND_PADDING,
        -TILE_INWARD_DEPTH,
    );
    let max_voxel_exclusive = IVec3::new(
        max_xy.x.ceil() as i32 + TILE_BOUND_PADDING,
        max_xy.y.ceil() as i32 + TILE_BOUND_PADDING,
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

fn tile_has_chunk(tile: &PlanetTile, chunk: IVec3) -> bool {
    tile.present_chunks
        .binary_search_by_key(&(chunk.x, chunk.y, chunk.z), |c| (c.x, c.y, c.z))
        .is_ok()
}

fn tile_has_any_chunk_in_region(tile: &PlanetTile, min: IVec3, size: IVec3) -> bool {
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
    if max.z <= -TILE_INWARD_DEPTH as f32 || min.z >= TILE_OUTWARD_HEIGHT as f32 {
        return false;
    }

    halfspaces.iter().all(|h| {
        // If the furthest AABB vertex in a halfspace's direction is still
        // outside, the whole chunk is outside.  This is conservative, so it may
        // keep a few edge chunks but will never crop a valid Voronoi cell.
        let p = Vec3::new(
            if h.normal.x >= 0.0 { max.x } else { min.x },
            if h.normal.y >= 0.0 { max.y } else { min.y },
            if h.normal.z >= 0.0 { max.z } else { min.z },
        );
        h.normal.dot(p) + h.offset >= -CHUNK_SIZE as f32
    })
}

fn build_planet_chunk(grid: GridKey, chunk: IVec3) -> Option<Voxels> {
    let tile = &planet_tiles()[tile_index(grid)?];
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
    points_to_voxels(points)
}

fn build_planet_lod_region(
    grid: GridKey,
    min_chunk: IVec3,
    size_chunks: IVec3,
    lod: f32,
) -> Option<Voxels> {
    let tile = &planet_tiles()[tile_index(grid)?];
    let step = 1i32 << lod.max(0.0).floor() as u32;
    let sample_offset = step / 2;
    let extent = (size_chunks * CHUNK_SIZE) / step;
    let origin = chunk_origin(min_chunk);
    let mut points = Vec::new();
    append_planet_samples(
        tile,
        origin,
        extent,
        step,
        sample_offset,
        false,
        &mut points,
    );
    points_to_voxels(points)
}

fn points_to_voxels(points: Vec<(I16Vec3, Voxel)>) -> Option<Voxels> {
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
    points: &mut Vec<(I16Vec3, Voxel)>,
) {
    let mass = if full_mass { 100 } else { 0 };
    let step_f = step as f32;
    let sample_base_z = origin.z as f32 + sample_offset as f32 + 0.5;

    for y in 0..extent.y {
        let sample_y = (origin.y + y * step + sample_offset) as f32 + 0.5;
        for x in 0..extent.x {
            let sample_x = (origin.x + x * step + sample_offset) as f32 + 0.5;
            let Some((z0, z1)) =
                column_shape_z_range(tile, sample_x, sample_y, sample_base_z, extent.z, step_f)
            else {
                continue;
            };

            let lateral_len_sq = sample_x * sample_x + sample_y * sample_y;

            for z in z0..z1 {
                let sample_z = sample_base_z + z as f32 * step_f;
                let radial = PLANET_RADIUS + sample_z;
                let radius = (lateral_len_sq + radial * radial).sqrt();
                if radius <= 1e-5 {
                    continue;
                }

                // Keep terrain sampling exact per voxel.  Fixed tangent-grid x/y
                // columns are not radial columns: as z changes, the normalized
                // planet direction changes too, so latitude/longitude-dependent
                // terrain must be re-evaluated for each z sample.
                let unit = local_unit_to_planet(tile, sample_x, sample_y, radial, radius);
                let terrain = terrain_sample(unit);
                let altitude = radius - PLANET_RADIUS;
                if altitude > terrain.height {
                    continue;
                }

                points.push((
                    IVec3::new(x, y, z).as_i16vec3(),
                    Voxel {
                        color: terrain_color(tile, terrain, altitude),
                        mass,
                    },
                ));
            }
        }
    }
}

fn column_shape_z_range(
    tile: &PlanetTile,
    sample_x: f32,
    sample_y: f32,
    sample_base_z: f32,
    extent_z: i32,
    step: f32,
) -> Option<(i32, i32)> {
    let mut min_sample_z = -TILE_INWARD_DEPTH as f32;
    let mut max_sample_z = TILE_OUTWARD_HEIGHT as f32;

    for h in &tile.halfspaces {
        let base = h.normal.x * sample_x + h.normal.y * sample_y + h.offset;
        if h.normal.z > 1e-6 {
            min_sample_z = min_sample_z.max((-TILE_SHAPE_EPSILON - base) / h.normal.z);
        } else if h.normal.z < -1e-6 {
            max_sample_z = max_sample_z.min((-TILE_SHAPE_EPSILON - base) / h.normal.z);
        } else if base < -TILE_SHAPE_EPSILON {
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

fn local_unit_to_planet(tile: &PlanetTile, x: f32, y: f32, radial: f32, radius: f32) -> Vec3 {
    (tile.axis_x * x + tile.axis_y * y + tile.normal * radial) / radius
}

#[derive(Clone, Copy)]
struct TerrainSample {
    height: f32,
    shade: f32,
}

fn terrain_sample(unit: Vec3) -> TerrainSample {
    let height = terrain_height(unit);
    let slope = 1.0 - unit.dot(Vec3::Y).abs() * 0.15;
    let shade_raw = (0.78 + 0.22 * value_noise(unit * 45.0)).clamp(0.58, 1.0) * slope;
    TerrainSample {
        height,
        shade: quantize_float(shade_raw.clamp(0.55, 1.0), 10),
    }
}

fn terrain_height(unit: Vec3) -> f32 {
    let continents = fbm(unit * 2.1 + Vec3::new(17.0, -31.0, 8.0), 5);
    let hills = fbm(unit * 8.0 + Vec3::new(-4.0, 19.0, 52.0), 4);
    let ridges = (1.0 - fbm(unit * 14.0 + Vec3::new(91.0, 7.0, -23.0), 4).abs()).powi(2);
    12.0 + continents * 34.0 + hills * 14.0 + ridges * TERRAIN_HEIGHT
}

fn terrain_color(tile: &PlanetTile, column: TerrainSample, altitude: f32) -> [u8; 4] {
    // The GPU tree only has 254 palette entries per uploaded voxel tree.  Keep
    // the procedural planet material-driven: 4 materials * 10 shade bands = at
    // most 40 colors per tile/chunk instead of one unique color per voxel.
    let material = if column.height > 72.0 || altitude > column.height - 8.0 && column.height > 55.0
    {
        0
    } else if column.height > 38.0 {
        1
    } else if column.height < -4.0 {
        2
    } else {
        3
    };

    let base = match material {
        0 => Vec3::new(218.0, 225.0, 220.0),
        1 => Vec3::new(112.0, 105.0, 88.0),
        2 => Vec3::new(45.0, 96.0, 56.0),
        _ => Vec3::new(68.0, 140.0, 72.0),
    };

    let tint = Vec3::new(
        tile.tint[0] as f32,
        tile.tint[1] as f32,
        tile.tint[2] as f32,
    );
    let color = (base * 0.88 + tint * 0.12) * column.shade;
    [
        quantize_u8(color.x.clamp(0.0, 255.0) as u8, 32),
        quantize_u8(color.y.clamp(0.0, 255.0) as u8, 32),
        quantize_u8(color.z.clamp(0.0, 255.0) as u8, 32),
        255,
    ]
}

fn quantize_float(value: f32, levels: u8) -> f32 {
    let max_level = (levels - 1) as f32;
    (value * max_level).round() / max_level
}

fn quantize_u8(value: u8, step: u8) -> u8 {
    ((value as u16 + (step as u16 / 2)) / step as u16 * step as u16).min(255) as u8
}

fn tile_tint(index: usize) -> [u8; 3] {
    let hue = fract(index as f32 * 0.618_034);
    let c = hsv_to_rgb(hue, 0.25, 1.0);
    [c.x as u8, c.y as u8, c.z as u8]
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

fn fbm(mut p: Vec3, octaves: usize) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut norm = 0.0;
    for _ in 0..octaves {
        sum += value_noise(p) * amp;
        norm += amp;
        p *= 2.03;
        amp *= 0.5;
    }
    sum / norm
}

fn value_noise(p: Vec3) -> f32 {
    let i = p.floor();
    let f = p - i;
    let u = f * f * f * (f * (f * 6.0 - Vec3::splat(15.0)) + Vec3::splat(10.0));

    let ix = i.x as i32;
    let iy = i.y as i32;
    let iz = i.z as i32;

    let x00 = lerp(hash3(ix, iy, iz), hash3(ix + 1, iy, iz), u.x);
    let x10 = lerp(hash3(ix, iy + 1, iz), hash3(ix + 1, iy + 1, iz), u.x);
    let x01 = lerp(hash3(ix, iy, iz + 1), hash3(ix + 1, iy, iz + 1), u.x);
    let x11 = lerp(
        hash3(ix, iy + 1, iz + 1),
        hash3(ix + 1, iy + 1, iz + 1),
        u.x,
    );
    let y0 = lerp(x00, x10, u.y);
    let y1 = lerp(x01, x11, u.y);
    lerp(y0, y1, u.z)
}

fn hash3(x: i32, y: i32, z: i32) -> f32 {
    let mut n = x as u32;
    n = n.wrapping_mul(0x9E37_79B9) ^ (y as u32).wrapping_mul(0x85EB_CA6B);
    n ^= (z as u32).wrapping_mul(0xC2B2_AE35);
    n ^= n >> 16;
    n = n.wrapping_mul(0x7FEB_352D);
    n ^= n >> 15;
    n = n.wrapping_mul(0x846C_A68B);
    n ^= n >> 16;
    (n as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn fract(value: f32) -> f32 {
    value - value.floor()
}
