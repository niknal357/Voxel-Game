mod camera_controller;
mod game_plugins;
mod gravity;
mod objects;
mod planet_source;
mod world_interaction;

use std::time::Duration;

use bevy::log::LogPlugin;
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
    app.add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Voxel Game".into(),
                    resolution: WindowResolution::new(1280, 720),
                    ..default()
                }),
                ..default()
            })
            .set(LogPlugin {
                custom_layer: tracy_layer,
                ..default()
            }),
    )
    .insert_resource(Time::<Virtual>::from_max_delta(Duration::from_millis(16)))
    .insert_resource(CameraVoxelLoaderDefaultSettings(
        CameraVoxelLoaderSettings {
            max_lod: 16,
            near_radius_chunks: 2,
            rings_per_lod: 2,
            requests_per_frame: 4,
            max_in_flight: 32,
        },
    ))
    .add_plugins(GamePlugins)
    .add_systems(Startup, setup_camera);

    #[cfg(feature = "tracy")]
    app.add_systems(Last, || tracing_tracy::client::frame_mark());

    app.run();

    drop(app);
    drop(runtime_guard);
    runtime.shutdown_background();
}

fn tracy_layer(_app: &mut App) -> Option<bevy::log::BoxedLayer> {
    #[cfg(feature = "tracy")]
    {
        tracing_tracy::client::Client::start();
        Some(Box::new(tracing_tracy::TracyLayer::default()))
    }
    #[cfg(not(feature = "tracy"))]
    None
}
