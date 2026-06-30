use std::sync::{Arc, OnceLock};

use bevy::math::U16Vec3;
use bevy::prelude::*;
use voxel_data::grid::{Grid, GridId};
use voxel_data::voxels::{Voxel, Voxels};
use voxel_edit::GridEdits;
use voxel_physics::RigidBody;
use voxel_physics::components::{VoxelCollider, VoxelMass};
use voxel_sources::{ChunkSource, SourceHandle, VoxelSourcesAppExt};
use voxel_streaming::{CHUNK_SIZE, GridStreaming, chunk_of, chunk_origin};

use crate::planet_source::PLANET_RADIUS;

const BALL_RADIUS: i32 = 16;
const BALL_COST: u32 = 1;
const BALL_MASS: u32 = 100;
const BALL_SPAWN_ALTITUDE: f32 = 180.0;
const BALL_FORWARD_OFFSET: f32 = -120.0;

pub(crate) struct BallPlugin;

impl Plugin for BallPlugin {
    fn build(&self, app: &mut App) {
        let grid = Arc::new(OnceLock::new());
        app.register_source(BallSource {
            grid: grid.clone(),
            handle: OnceLock::new(),
        })
        .insert_resource(BallGrid(grid))
        .add_systems(Startup, spawn_ball);
    }
}

#[derive(Resource, Clone)]
struct BallGrid(Arc<OnceLock<GridId>>);

struct BallSource {
    grid: Arc<OnceLock<GridId>>,
    handle: OnceLock<SourceHandle>,
}

impl BallSource {
    fn is_mine(&self, grid: GridId) -> bool {
        self.grid.get() == Some(&grid)
    }
}

impl ChunkSource for BallSource {
    fn init(&self, handle: SourceHandle) {
        let _ = self.handle.set(handle);
    }

    fn cost(&self, grid: GridId, chunk: IVec3) -> Option<u32> {
        (self.is_mine(grid) && chunk_intersects_ball(chunk)).then_some(BALL_COST)
    }

    fn request_load(&self, grid: GridId, chunk: IVec3, generation: u64) {
        let voxels = build_ball_chunk(chunk, true);
        if let Some(handle) = self.handle.get() {
            handle.loaded(grid, chunk, generation, voxels);
        }
    }

    fn cost_lod(&self, grid: GridId, min: IVec3, size: IVec3, _lod: f32) -> Option<u32> {
        (self.is_mine(grid) && chunk_region_intersects_ball(min, size)).then_some(BALL_COST)
    }

    fn request_load_lod(&self, grid: GridId, min: IVec3, size: IVec3, lod: f32, generation: u64) {
        let voxels = build_ball_lod_region(min, size, lod);
        if let Some(handle) = self.handle.get() {
            handle.loaded_lod(grid, min, size, lod, generation, voxels);
        }
    }
}

fn spawn_ball(mut commands: Commands, grid: Res<BallGrid>) {
    let spawn_position = Vec3::new(
        0.0,
        PLANET_RADIUS + BALL_SPAWN_ALTITUDE,
        BALL_FORWARD_OFFSET,
    );

    let body = commands
        .spawn((RigidBody, Transform::from_translation(spawn_position)))
        .id();

    let mut streaming = GridStreaming::default();
    for chunk in ball_present_chunks() {
        streaming.presence_mut().mark_present(chunk);
    }

    let grid_entity = commands
        .spawn((
            Transform::from_translation(Vec3::splat(-0.5)),
            Grid::new(),
            VoxelCollider,
            VoxelMass,
            GridEdits::default(),
            streaming,
        ))
        .id();

    let _ = grid.0.set(grid_entity);
    commands.entity(body).add_child(grid_entity);
}

fn ball_present_chunks() -> Vec<IVec3> {
    let min = chunk_of(IVec3::splat(-BALL_RADIUS));
    let max = chunk_of(IVec3::splat(BALL_RADIUS));
    let mut chunks = Vec::new();

    for x in min.x..=max.x {
        for y in min.y..=max.y {
            for z in min.z..=max.z {
                let chunk = IVec3::new(x, y, z);
                if chunk_intersects_ball(chunk) {
                    chunks.push(chunk);
                }
            }
        }
    }

    chunks
}

fn chunk_intersects_ball(chunk: IVec3) -> bool {
    chunk_region_intersects_ball(chunk, IVec3::ONE)
}

fn chunk_region_intersects_ball(min_chunk: IVec3, size_chunks: IVec3) -> bool {
    let min = chunk_origin(min_chunk).as_vec3();
    let max = chunk_origin(min_chunk + size_chunks).as_vec3();
    let closest = Vec3::ZERO.clamp(min, max);
    closest.length_squared() <= ball_radius_squared()
}

fn ball_radius_squared() -> f32 {
    (BALL_RADIUS as f32 + 0.5).powi(2)
}

fn ball_voxel(pos: IVec3, mass: u32) -> Option<Voxel> {
    if pos.as_vec3().length_squared() > ball_radius_squared() {
        return None;
    }

    let normal = pos.as_vec3().normalize_or_zero();
    Some(Voxel {
        color: [
            ((((normal.x * 0.5 + 0.5) * 255.0) / 16.0).round() * 16.0) as u8,
            ((((normal.y * 0.5 + 0.5) * 255.0) / 16.0).round() * 16.0) as u8,
            255,
            // ((normal.z * 0.5 + 0.5) * 255.0) as u8,
            255,
        ],
        mass,
    })
}

fn build_ball_chunk(chunk: IVec3, include_mass: bool) -> Option<Voxels> {
    let origin = chunk_origin(chunk);
    let mass = if include_mass { BALL_MASS } else { 0 };
    let mut points = Vec::new();

    for x in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let local = IVec3::new(x, y, z);
                let Some(voxel) = ball_voxel(origin + local, mass) else {
                    continue;
                };
                points.push((local.as_u16vec3(), voxel));
            }
        }
    }

    points_to_voxels(points)
}

fn build_ball_lod_region(min_chunk: IVec3, size_chunks: IVec3, lod: f32) -> Option<Voxels> {
    let step = 1i32 << lod.max(0.0).floor() as u32;
    let sample_offset = step / 2;
    let extent = (size_chunks * CHUNK_SIZE) / step;
    let origin = chunk_origin(min_chunk);
    let max_source = size_chunks * CHUNK_SIZE - IVec3::ONE;
    let mut points = Vec::new();

    for x in 0..extent.x {
        for y in 0..extent.y {
            for z in 0..extent.z {
                let coarse = IVec3::new(x, y, z);
                let sample = (coarse * step + IVec3::splat(sample_offset)).min(max_source);
                let Some(voxel) = ball_voxel(origin + sample, 0) else {
                    continue;
                };
                points.push((coarse.as_u16vec3(), voxel));
            }
        }
    }

    points_to_voxels(points)
}

fn points_to_voxels(points: Vec<(U16Vec3, Voxel)>) -> Option<Voxels> {
    if points.is_empty() {
        None
    } else {
        let mut voxels = Voxels::new();
        voxels.add_voxels(&points);
        Some(voxels)
    }
}
