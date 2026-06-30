use bevy::app::PluginGroupBuilder;
use bevy::prelude::*;
use camera_voxel_loader::{CameraVoxelLoaderPlugin, CameraVoxelRenderState};
use voxel_edit::VoxelEditPlugin;
use voxel_physics::VoxelPhysicsPlugin;
use voxel_renderer::{VoxelRendererPlugin, voxel_camera::VoxelCamera};
use voxel_streaming::VoxelStreamingPlugin;

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
            .add(VoxelEditPlugin)
            .add(VoxelStreamingPlugin)
            .add(VoxelPhysicsPlugin)
            .add(GravityPlugin)
            // .add(ProceduralSpherePlugin)
            .add(ProceduralPlanetPlugin)
            .add(BallPlugin)
            .add(SpaceshipPlugin)
            .add(CameraVoxelLoaderPlugin)
            .add(VoxelRendererPlugin)
            .add(VoxelCameraLinkPlugin)
            .add(FlyCameraPlugin)
            .add(WorldInteractionPlugin)
    }
}

struct VoxelCameraLinkPlugin;

impl Plugin for VoxelCameraLinkPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, ensure_voxel_camera_components);
    }
}

fn ensure_voxel_camera_components(
    mut commands: Commands,
    mut cameras: Query<(Entity, Option<&CameraVoxelRenderState>, Option<&mut VoxelCamera>), With<Camera3d>>,
) {
    for (entity, render_state, voxel_camera) in &mut cameras {
        match (render_state, voxel_camera) {
            (Some(render_state), Some(mut voxel_camera)) => {
                voxel_camera.subgrids_to_render = render_state.subgrids_to_render.clone();
                voxel_camera.lods_to_render = render_state.lods_to_render.clone();
            }
            (Some(render_state), None) => {
                commands.entity(entity).insert(VoxelCamera {
                    subgrids_to_render: render_state.subgrids_to_render.clone(),
                    lods_to_render: render_state.lods_to_render.clone(),
                });
            }
            (None, Some(mut voxel_camera)) => {
                voxel_camera.subgrids_to_render.clear();
                voxel_camera.lods_to_render.clear();
            }
            (None, None) => {
                commands.entity(entity).insert(VoxelCamera::default());
            }
        }
    }
}
