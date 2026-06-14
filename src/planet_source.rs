use std::sync::OnceLock;

use bevy::prelude::*;
use voxel_data::grid::Grid;
use voxel_data::voxels::{Voxel, Voxels};
use voxel_edit::GridEdits;
use voxel_sources::{ChunkSource, GridKey, SourceHandle, VoxelSourcesAppExt};
use voxel_streaming::{CHUNK_SIZE, GridStreaming, chunk_origin};
use voxel_physics::{components::VoxelCollider, RigidBody, IsStatic};

const PLANET_GRID_KEY: GridKey = GridKey(1000000);
const PLANET_RADIUS: i32 = 5000 * 32; // 5000m * 32voxels/m
const PLANET_COST: u32 = 1;

#[derive(Resource)]
struct PlanetTiles(Vec<(GridKey, Transform)>);

pub struct ProceduralPlanetPlugin;

impl Plugin for ProceduralPlanetPlugin {
    fn build(&self, app: &mut App) {
        app.register_source(ProceduralPlanetSource::default())
            .add_systems(Startup, spawn_planet);
    }
}

#[derive(Default)]
struct ProceduralPlanetSource {
    handle: OnceLock<SourceHandle>,
}

fn spawn_planet(commands: &mut Commands, planet_tiles: &PlanetTiles) {
    let parent = commands.spawn((
        RigidBody,
        IsStatic,
        Transform::IDENTITY
    )).id();

    for &(grid_key, transform) in &planet_tiles.0 {
        let child = commands.spawn((
            transform,
            Grid::new(),
            GridEdits::default(),
            grid_key,
            GridStreaming::default(),
            VoxelCollider,
        )).id();
        commands.entity(parent).add_child(child);
    }
}
