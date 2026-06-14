use std::sync::OnceLock;

use bevy::prelude::*;
use voxel_data::grid::Grid;
use voxel_data::voxels::{Voxel, Voxels};
use voxel_edit::GridEdits;
use voxel_sources::{ChunkSource, GridKey, SourceHandle, VoxelSourcesAppExt};
use voxel_streaming::{CHUNK_SIZE, GridStreaming, chunk_origin};

const PLANET_RADIUS: i32 = 5000 * 32; // 5000m * 32voxels/m
const PLANET_COST: u32 = 1;

pub struct ProceduralPlanetPlugin;

impl Plugin for ProceduralPlanetPlugin {
    fn build(&self, app: &mut App) {
        app.register_source(ProceduralPlanetSource::default())
            .add_systems(Startup, spawn_planet_grids);
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
        (grid == PLANET_GRID && chunk_intersects_planet(chunk)).then_some(PLANET_COST)
    }

    fn request_load(&self, grid: GridKey, chunk: IVec3) {
        let voxels = build_planet_chunk(chunk);
        if let Some(handle) = self.handle.get() {
            handle.loaded(grid, chunk, voxels);
        }
    }

    fn cost_lod(&self, grid: GridKey, min: IVec3, size: IVec3, _lod: f32) -> Option<u32> {
        (grid == PLANET_GRID && chunk_region_intersects_planet(min, size)).then_some(PLANET_COST)
    }

    fn request_load_lod(&self, grid: GridKey, min: IVec3, size: IVec3, lod: f32) {
        let voxels = build_planet_lod_region(min, size, lod);
        if let Some(handle) = self.handle.get() {
            handle.loaded_lod(grid, min, size, lod, voxels);
        }
    }
}


