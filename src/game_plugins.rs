use bevy::app::PluginGroupBuilder;
use bevy::prelude::*;
use voxel_engine::{VoxelEngineMode, VoxelEnginePlugins};

use crate::camera_controller::FlyCameraPlugin;
use crate::gravity::GravityPlugin;
use crate::objects::{BallPlugin, SpaceshipPlugin};
use crate::planet_source::ProceduralPlanetPlugin;
use crate::world_interaction::WorldInteractionPlugin;
// use crate::sphere_source::ProceduralSpherePlugin;

pub struct GamePlugins;

impl PluginGroup for GamePlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add_group(VoxelEnginePlugins {
                mode: VoxelEngineMode::Host,
            })
            .add(GravityPlugin)
            // .add(ProceduralSpherePlugin)
            .add(ProceduralPlanetPlugin)
            .add(BallPlugin)
            .add(SpaceshipPlugin)
            .add(FlyCameraPlugin)
            .add(WorldInteractionPlugin)
    }
}
