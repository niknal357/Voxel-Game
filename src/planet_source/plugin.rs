use std::sync::OnceLock;

use bevy::prelude::*;
use tracy_client::span;
use voxel_data::grid::Grid;
use voxel_edit::GridEdits;
use voxel_physics::{IsStatic, RigidBody, components::VoxelCollider};
use voxel_sources::{ChunkSource, GridKey, SourceHandle, VoxelSourcesAppExt};
use voxel_streaming::GridStreaming;

use super::config::PLANET_COST;
use super::generation::{build_planet_chunk, build_planet_lod_region};
use super::tiles::{
    grid_key, planet_tiles, tile_has_any_chunk_in_region, tile_has_chunk, tile_index,
};

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
        let _zone = span!("planet source request_load chunk");
        let voxels = build_planet_chunk(grid, chunk);
        if let Some(handle) = self.handle.get() {
            let _zone = span!("planet source publish chunk");
            handle.loaded(grid, chunk, voxels);
        }
    }

    fn cost_lod(&self, grid: GridKey, min: IVec3, size: IVec3, _lod: f32) -> Option<u32> {
        let tile = planet_tiles().get(tile_index(grid)?)?;
        tile_has_any_chunk_in_region(tile, min, size).then_some(PLANET_COST)
    }

    fn request_load_lod(&self, grid: GridKey, min: IVec3, size: IVec3, lod: f32) {
        let _zone = span!("planet source request_load_lod");
        tracy_client::plot!("planet lod level", lod as f64);
        let voxels = build_planet_lod_region(grid, min, size, lod);
        if let Some(handle) = self.handle.get() {
            let _zone = span!("planet source publish lod");
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
