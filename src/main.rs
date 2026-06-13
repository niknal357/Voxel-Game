use std::sync::OnceLock;
use std::time::Duration;

use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::render::view::{Hdr, Msaa};
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow, WindowResolution};
use camera_voxel_loader::CameraVoxelLoaderPlugin;
use voxel_data::grid::Grid;
use voxel_data::voxels::{Voxel, Voxels};
use voxel_edit::{GridEdits, VoxelEditPlugin};
use voxel_renderer::VoxelRendererPlugin;
use voxel_sources::{ChunkSource, GridKey, SourceHandle, VoxelSourcesAppExt, VoxelSourcesPlugin};
use voxel_streaming::{CHUNK_SIZE, GridStreaming, VoxelStreamingPlugin, chunk_origin};

const SPHERE_GRID: GridKey = GridKey(1);
const SPHERE_RADIUS: i32 = 28;
const SPHERE_COST: u32 = 1;
const SPHERE_COLOR: [u8; 4] = [80, 180, 255, 255];

fn main() {
    // voxel-data starts Tokio tasks while its Bevy plugin is being built, so we
    // enter a runtime before adding the voxel engine plugins.
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
    .add_plugins((
        VoxelEditPlugin,
        VoxelStreamingPlugin,
        VoxelSourcesPlugin,
        ProceduralSpherePlugin,
        CameraVoxelLoaderPlugin,
        VoxelRendererPlugin,
        FlyCameraPlugin,
    ))
    .add_systems(Startup, setup_camera);

    app.run();

    drop(app);
    drop(runtime_guard);
    runtime.shutdown_background();
}

fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Hdr,
        Msaa::Off,
        Transform::from_xyz(0.0, 16.0, 120.0),
        FlyCamera::default(),
    ));
}

struct ProceduralSpherePlugin;

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
                    voxels.add_voxel(local.as_i16vec3(), voxel);
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
                    voxels.add_voxel(coarse.as_i16vec3(), voxel);
                }
            }
        }
    }

    (!voxels.is_empty()).then_some(voxels)
}

#[derive(Component)]
struct FlyCamera {
    speed: f32,
    rotation_speed: f32,
    mouse_sensitivity: f32,
    yaw: f32,
    pitch: f32,
}

impl Default for FlyCamera {
    fn default() -> Self {
        Self {
            speed: 30.0,
            rotation_speed: 1.5,
            mouse_sensitivity: 0.0015,
            yaw: 0.0,
            pitch: 0.0,
        }
    }
}

struct FlyCameraPlugin;

impl Plugin for FlyCameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (toggle_cursor_grab, fly_camera_system).chain());
    }
}

const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

fn fly_camera_system(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    cursor_options: Query<&CursorOptions, With<PrimaryWindow>>,
    mut cameras: Query<(&mut Transform, &mut FlyCamera)>,
) {
    let dt = time.delta_secs();
    let mouse_captured = cursor_options
        .single()
        .map(|cursor| cursor.grab_mode != CursorGrabMode::None)
        .unwrap_or(false);
    let mouse_delta = if mouse_captured {
        mouse_motion.delta
    } else {
        Vec2::ZERO
    };

    for (mut transform, mut camera) in &mut cameras {
        camera.yaw -= mouse_delta.x * camera.mouse_sensitivity;
        camera.pitch -= mouse_delta.y * camera.mouse_sensitivity;

        if keys.pressed(KeyCode::ArrowLeft) {
            camera.yaw += camera.rotation_speed * dt;
        }
        if keys.pressed(KeyCode::ArrowRight) {
            camera.yaw -= camera.rotation_speed * dt;
        }
        if keys.pressed(KeyCode::ArrowUp) {
            camera.pitch += camera.rotation_speed * dt;
        }
        if keys.pressed(KeyCode::ArrowDown) {
            camera.pitch -= camera.rotation_speed * dt;
        }

        camera.pitch = camera.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
        transform.rotation = Quat::from_axis_angle(Vec3::Y, camera.yaw)
            * Quat::from_axis_angle(Vec3::X, camera.pitch);

        let sprint = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
        let speed = camera.speed * if sprint { 8.0 } else { 1.0 };
        let mut direction = Vec3::ZERO;

        if keys.pressed(KeyCode::KeyW) {
            direction += transform.forward().as_vec3();
        }
        if keys.pressed(KeyCode::KeyS) {
            direction -= transform.forward().as_vec3();
        }
        if keys.pressed(KeyCode::KeyD) {
            direction += transform.right().as_vec3();
        }
        if keys.pressed(KeyCode::KeyA) {
            direction -= transform.right().as_vec3();
        }
        if keys.pressed(KeyCode::KeyE) {
            direction += transform.up().as_vec3();
        }
        if keys.pressed(KeyCode::KeyQ) {
            direction -= transform.up().as_vec3();
        }

        if direction != Vec3::ZERO {
            transform.translation += direction.normalize() * speed * dt;
        }
    }
}

fn toggle_cursor_grab(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut cursor_options: Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    let Ok(mut cursor) = cursor_options.single_mut() else {
        return;
    };

    if mouse_buttons.just_pressed(MouseButton::Left) {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
    } else if keys.just_pressed(KeyCode::Escape) {
        cursor.grab_mode = CursorGrabMode::None;
        cursor.visible = true;
    }
}
