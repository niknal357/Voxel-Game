mod camera_controller;
mod game_plugins;
mod planet_source;

use std::time::Duration;

use bevy::prelude::*;
use bevy::window::{WindowPlugin, WindowResolution};
use camera_voxel_loader::{CameraVoxelLoaderDefaultSettings, CameraVoxelLoaderSettings};

use camera_controller::setup_camera;
use game_plugins::GamePlugins;

fn main() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to create Tokio runtime");
    let runtime_guard = runtime.enter();

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Voxel Game".into(),
            resolution: WindowResolution::new(1280, 720),
            ..default()
        }),
        ..default()
    }))
    .insert_resource(Time::<Virtual>::from_max_delta(Duration::from_millis(16)))
    .insert_resource(CameraVoxelLoaderDefaultSettings(
        CameraVoxelLoaderSettings {
            max_lod: 3,
            near_radius_chunks: 3,
            rings_per_lod: 2,
            requests_per_frame: 16,
            max_in_flight: 128,
        },
    ))
    .add_plugins(GamePlugins)
    .add_systems(Startup, setup_camera);

    app.run();

    drop(app);
    drop(runtime_guard);
    runtime.shutdown_background();
}
