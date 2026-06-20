use bevy::app::PluginGroupBuilder;
use bevy::prelude::*;
use camera_voxel_loader::CameraVoxelLoaderPlugin;
use voxel_edit::VoxelEditPlugin;
use voxel_physics::VoxelPhysicsPlugin;
use voxel_renderer::VoxelRendererPlugin;
use voxel_streaming::VoxelStreamingPlugin;

use crate::gravity::GravityPlugin;
use crate::objects::BallPlugin;
use crate::planet_source::ProceduralPlanetPlugin;
use crate::player_controller::PlayerPlugin;
use crate::world_interaction::WorldInteractionPlugin;
// use crate::sphere_source::ProceduralSpherePlugin;

pub struct GamePlugins;

impl PluginGroup for GamePlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(VoxelEditPlugin)
            .add(VoxelStreamingPlugin)
            .add(VoxelPhysicsPlugin)
            .add(GravityPlugin)
            // .add(ProceduralSpherePlugin)
            .add(ProceduralPlanetPlugin)
            .add(BallPlugin)
            .add(CameraVoxelLoaderPlugin)
            .add(VoxelRendererPlugin)
            .add(PlayerPlugin)
            .add(WorldInteractionPlugin)
    }
}
