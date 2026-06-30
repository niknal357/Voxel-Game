use std::sync::OnceLock;

use bevy::prelude::*;
use voxel_data::grid::Grid;
use voxel_data::voxels::{Voxel, Voxels};
use voxel_edit::GridEdits;
use voxel_sources::{ChunkSource, GridKey, SourceHandle, VoxelSourcesAppExt};
use voxel_streaming::{CHUNK_SIZE, GridStreaming, chunk_origin};

const SPHERE_GRID: GridKey = GridKey(1);
const SPHERE_RADIUS: i32 = 4096;
const SPHERE_COST: u32 = 1;
const SPHERE_COLOR: [u8; 4] = [80, 180, 255, 255];

pub struct ProceduralSpherePlugin;

impl Plugin for ProceduralSpherePlugin {
    fn build(&self, app: &mut App) {
        app.register_source(ProceduralSphereSource::default())
            .add_systems(Startup, spawn_sphere_grid);
    }
}

#[derive(Default)]
struct ProceduralSphereSource {
    handle: OnceLock<SourceHandle>,
}

impl ChunkSource for ProceduralSphereSource {
    fn init(&self, handle: SourceHandle) {
        let _ = self.handle.set(handle);
    }

    fn cost(&self, grid: GridKey, chunk: IVec3) -> Option<u32> {
        (grid == SPHERE_GRID && chunk_intersects_sphere(chunk)).then_some(SPHERE_COST)
    }

    fn request_load(&self, grid: GridKey, chunk: IVec3) {
        let voxels = build_sphere_chunk(chunk);
        if let Some(handle) = self.handle.get() {
            handle.loaded(grid, chunk, voxels);
        }
    }

    fn cost_lod(&self, grid: GridKey, min: IVec3, size: IVec3, _lod: f32) -> Option<u32> {
        (grid == SPHERE_GRID && chunk_region_intersects_sphere(min, size)).then_some(SPHERE_COST)
    }

    fn request_load_lod(&self, grid: GridKey, min: IVec3, size: IVec3, lod: f32) {
        let voxels = build_sphere_lod_region(min, size, lod);
        if let Some(handle) = self.handle.get() {
            handle.loaded_lod(grid, min, size, lod, voxels);
        }
    }
}

fn spawn_sphere_grid(mut commands: Commands) {
    let radius_chunks = SPHERE_RADIUS.div_euclid(CHUNK_SIZE) + 1;
    let min = IVec3::splat(-radius_chunks);
    let size = IVec3::splat(radius_chunks * 2 + 1);

    let mut streaming = GridStreaming::default();
    streaming.presence_mut().mark_present_area(min, size);

    commands.spawn((
        Transform::IDENTITY,
        Grid::new(),
        GridEdits::default(),
        SPHERE_GRID,
        streaming,
    ));
}

fn chunk_intersects_sphere(chunk: IVec3) -> bool {
    chunk_region_intersects_sphere(chunk, IVec3::ONE)
}

fn chunk_region_intersects_sphere(min_chunk: IVec3, size_chunks: IVec3) -> bool {
    let min = chunk_origin(min_chunk).as_vec3();
    let max = chunk_origin(min_chunk + size_chunks).as_vec3();
    Vec3::ZERO.clamp(min, max).length_squared() <= (SPHERE_RADIUS as f32).powi(2)
}

fn sphere_voxel(pos: IVec3) -> Option<Voxel> {
    if pos.as_vec3().length_squared() > (SPHERE_RADIUS as f32).powi(2) {
        return None;
    }

    Some(Voxel {
        color: SPHERE_COLOR,
        mass: 100,
    })
}

fn build_sphere_chunk(chunk: IVec3) -> Option<Voxels> {
    let origin = chunk_origin(chunk);
    let mut voxels = Voxels::new();

    for x in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let local = IVec3::new(x, y, z);
                if let Some(voxel) = sphere_voxel(origin + local) {
                    voxels.add_voxel(local.as_u16vec3(), voxel);
                }
            }
        }
    }

    (!voxels.is_empty()).then_some(voxels)
}

fn build_sphere_lod_region(min_chunk: IVec3, size_chunks: IVec3, lod: f32) -> Option<Voxels> {
    let step = 1i32 << lod.max(0.0).floor() as u32;
    let half_step = step / 2;
    let extent = (size_chunks * CHUNK_SIZE) / step;
    let origin = chunk_origin(min_chunk);
    let max_source = size_chunks * CHUNK_SIZE - IVec3::ONE;
    let mut voxels = Voxels::new();

    for x in 0..extent.x {
        for y in 0..extent.y {
            for z in 0..extent.z {
                let coarse = IVec3::new(x, y, z);
                let sample = (coarse * step + IVec3::splat(half_step)).min(max_source);
                if let Some(voxel) = sphere_voxel(origin + sample) {
                    voxels.add_voxel(coarse.as_u16vec3(), voxel);
                }
            }
        }
    }

    (!voxels.is_empty()).then_some(voxels)
}
