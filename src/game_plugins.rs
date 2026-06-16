use bevy::app::PluginGroupBuilder;
use bevy::prelude::*;
use camera_voxel_loader::CameraVoxelLoaderPlugin;
use voxel_edit::VoxelEditPlugin;
use voxel_renderer::VoxelRendererPlugin;
use voxel_sources::VoxelSourcesPlugin;
use voxel_streaming::VoxelStreamingPlugin;

use crate::camera_controller::FlyCameraPlugin;
use crate::planet_source::ProceduralPlanetPlugin;
// use crate::sphere_source::ProceduralSpherePlugin;

pub struct GamePlugins;

impl PluginGroup for GamePlugins {
    fn build(self) -> PluginGroupBuilder {
        PluginGroupBuilder::start::<Self>()
            .add(VoxelEditPlugin)
            .add(VoxelStreamingPlugin)
            .add(VoxelSourcesPlugin)
            // .add(ProceduralSpherePlugin)
            .add(ProceduralPlanetPlugin)
            .add(CameraVoxelLoaderPlugin)
            .add(VoxelRendererPlugin)
            .add(FlyCameraPlugin)
    }
}
