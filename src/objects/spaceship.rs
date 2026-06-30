use std::sync::{Arc, OnceLock};

use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::math::U16Vec3;
use bevy::prelude::*;
use bevy::render::view::{Hdr, Msaa};
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use voxel_data::grid::{Grid, GridId};
use voxel_data::voxels::{Voxel, Voxels};
use voxel_data::world_query::VoxelWorldQueryParam;
use voxel_edit::GridEdits;
use voxel_physics::components::{VoxelCollider, VoxelMass};
use voxel_physics::{
    Accelerations, AngularVelocity, FreezePhysics, Impulses, Mass, PhysicsSet, RigidBody,
    RotationalInertia, Velocity,
};
use voxel_sources::{ChunkSource, SourceHandle, VoxelSourcesAppExt};
use voxel_streaming::{CHUNK_SIZE, GridStreaming, chunk_of, chunk_origin};

use crate::gravity::PlanetGravity;
use crate::planet_source::PLANET_RADIUS;

const VOXELS_PER_METER: f32 = 32.0;

const SHIP_COST: u32 = 1;
const SHIP_MASS_PER_VOXEL: u32 = 8;
const SHIP_BOUNDS_MIN: IVec3 = IVec3::new(-36, -10, -58);
const SHIP_BOUNDS_MAX: IVec3 = IVec3::new(36, 20, 52);
const SHIP_GRID_OFFSET: Vec3 = Vec3::splat(-0.5);

const SPAWN_ALTITUDE: f32 = 220.0;
const SPAWN_FORWARD_OFFSET: f32 = -160.0;

const MAIN_THRUST_ACCEL: f32 = 900.0;
const BOOST_THRUST_ACCEL: f32 = 2_200.0;
const STRAFE_THRUST_ACCEL: f32 = 700.0;

const KEY_TURN_RATE: f32 = 1.8;
const MOUSE_TURN_RATE_PER_PIXEL: f32 = 0.045;
const MAX_TARGET_TURN_RATE: f32 = 3.0;
const ANGULAR_ACCEL: f32 = 18.0;

const HOVER_ASSIST_RANGE: f32 = 192.0;
const HOVER_TARGET_ALTITUDE: f32 = 56.0;
const HOVER_MAX_ACCEL: f32 = 650.0;
const GROUND_RAY_CLEARANCE: f32 = 48.0;
const GROUND_RAY_MAX_DISTANCE: f32 = 768.0;

const COCKPIT_CAMERA_OFFSET: Vec3 = Vec3::new(0.0, 8.0, -58.0);
const ORBIT_FOCUS_OFFSET: Vec3 = Vec3::new(0.0, 4.0, -6.0);
const ORBIT_DEFAULT_DISTANCE: f32 = 170.0;
const ORBIT_MIN_DISTANCE: f32 = 60.0;
const ORBIT_MAX_DISTANCE: f32 = 420.0;
const ORBIT_PITCH_LIMIT: f32 = 1.25;

pub(crate) struct SpaceshipPlugin;

impl Plugin for SpaceshipPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<EguiPlugin>() {
            app.add_plugins(EguiPlugin::default());
        }

        let grid = Arc::new(OnceLock::new());
        app.register_source(SpaceshipSource {
            grid: grid.clone(),
            handle: OnceLock::new(),
        })
        .insert_resource(SpaceshipGrid(grid))
        .init_resource::<SpaceshipInput>()
        .init_resource::<SpaceshipAutopilot>()
        .init_resource::<SpaceshipCameraRig>()
        .init_resource::<SpaceshipTelemetry>()
        .add_systems(Startup, spawn_spaceship)
        .add_systems(Update, collect_spaceship_input)
        .add_systems(
            FixedUpdate,
            fly_spaceship_system
                .in_set(PhysicsSet::Apply)
                .run_if(|freeze: Res<FreezePhysics>| !freeze.0),
        )
        .add_systems(
            Update,
            (update_spaceship_camera, update_spaceship_telemetry).chain(),
        )
        .add_systems(EguiPrimaryContextPass, spaceship_hud);
    }
}

#[derive(Resource, Clone)]
struct SpaceshipGrid(Arc<OnceLock<GridId>>);

struct SpaceshipSource {
    grid: Arc<OnceLock<GridId>>,
    handle: OnceLock<SourceHandle>,
}

impl SpaceshipSource {
    fn is_mine(&self, grid: GridId) -> bool {
        self.grid.get() == Some(&grid)
    }
}

impl ChunkSource for SpaceshipSource {
    fn init(&self, handle: SourceHandle) {
        let _ = self.handle.set(handle);
    }

    fn cost(&self, grid: GridId, chunk: IVec3) -> Option<u32> {
        (self.is_mine(grid) && chunk_has_ship_voxels(chunk)).then_some(SHIP_COST)
    }

    fn request_load(&self, grid: GridId, chunk: IVec3, generation: u64) {
        let voxels = build_ship_chunk(chunk, true);
        if let Some(handle) = self.handle.get() {
            handle.loaded(grid, chunk, generation, voxels);
        }
    }

    fn cost_lod(&self, grid: GridId, min: IVec3, size: IVec3, _lod: f32) -> Option<u32> {
        (self.is_mine(grid) && chunk_region_intersects_ship_bounds(min, size)).then_some(SHIP_COST)
    }

    fn request_load_lod(&self, grid: GridId, min: IVec3, size: IVec3, lod: f32, generation: u64) {
        let voxels = build_ship_lod_region(min, size, lod);
        if let Some(handle) = self.handle.get() {
            handle.loaded_lod(grid, min, size, lod, generation, voxels);
        }
    }
}

#[derive(Component)]
struct Spaceship;

#[derive(Component)]
struct SpaceshipCamera;

#[derive(Component)]
struct ShipVoxelGrid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CameraMode {
    Cockpit,
    Orbit,
}

impl CameraMode {
    fn label(self) -> &'static str {
        match self {
            CameraMode::Cockpit => "cockpit",
            CameraMode::Orbit => "orbit cam",
        }
    }
}

#[derive(Resource, Debug, Clone)]
struct SpaceshipCameraRig {
    mode: CameraMode,
    orbit_yaw: f32,
    orbit_pitch: f32,
    orbit_distance: f32,
}

impl Default for SpaceshipCameraRig {
    fn default() -> Self {
        Self {
            mode: CameraMode::Cockpit,
            orbit_yaw: 0.0,
            orbit_pitch: 0.25,
            orbit_distance: ORBIT_DEFAULT_DISTANCE,
        }
    }
}

#[derive(Resource, Debug, Clone, Copy)]
struct SpaceshipInput {
    thrust: Vec3,
    look_delta: Vec2,
    keyboard_turn: Vec3,
    boost: bool,
    hover_assist: bool,
}

impl Default for SpaceshipInput {
    fn default() -> Self {
        Self {
            thrust: Vec3::ZERO,
            look_delta: Vec2::ZERO,
            keyboard_turn: Vec3::ZERO,
            boost: false,
            hover_assist: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpaceshipAutopilotMode {
    Manual,
}

#[derive(Resource, Debug, Clone, Copy)]
struct SpaceshipAutopilot {
    mode: SpaceshipAutopilotMode,
}

impl Default for SpaceshipAutopilot {
    fn default() -> Self {
        Self {
            mode: SpaceshipAutopilotMode::Manual,
        }
    }
}

impl SpaceshipInput {
    fn reset_frame_delta(&mut self) -> Vec2 {
        let delta = self.look_delta;
        self.look_delta = Vec2::ZERO;
        delta
    }
}

#[derive(Resource, Debug, Clone, Copy)]
struct SpaceshipTelemetry {
    speed: f32,
    vertical_speed: f32,
    horizontal_speed: f32,
    radial_altitude: f32,
    ground_altitude: Option<f32>,
    apoapsis_altitude: Option<f32>,
    periapsis_altitude: Option<f32>,
    circular_speed: f32,
    eccentricity: Option<f32>,
    hover_assist_active: bool,
    camera_mode: CameraMode,
}

impl Default for SpaceshipTelemetry {
    fn default() -> Self {
        Self {
            speed: 0.0,
            vertical_speed: 0.0,
            horizontal_speed: 0.0,
            radial_altitude: 0.0,
            ground_altitude: None,
            apoapsis_altitude: None,
            periapsis_altitude: None,
            circular_speed: 0.0,
            eccentricity: None,
            hover_assist_active: true,
            camera_mode: CameraMode::Cockpit,
        }
    }
}

fn spawn_spaceship(mut commands: Commands, grid: Res<SpaceshipGrid>) {
    let spawn_position = Vec3::new(0.0, PLANET_RADIUS + SPAWN_ALTITUDE, SPAWN_FORWARD_OFFSET);

    let body = commands
        .spawn((
            RigidBody,
            Spaceship,
            Transform::from_translation(spawn_position),
        ))
        .id();

    let mut streaming = GridStreaming::default();
    for chunk in ship_present_chunks() {
        streaming.presence_mut().mark_present(chunk);
    }

    let grid_entity = commands
        .spawn((
            Transform::from_translation(SHIP_GRID_OFFSET),
            Grid::new(),
            VoxelCollider,
            VoxelMass,
            ShipVoxelGrid,
            GridEdits::default(),
            streaming,
        ))
        .id();

    let _ = grid.0.set(grid_entity);
    commands.entity(body).add_child(grid_entity);

    commands.spawn((
        Camera3d::default(),
        Hdr,
        Msaa::Off,
        Transform::from_translation(spawn_position + COCKPIT_CAMERA_OFFSET),
        SpaceshipCamera,
    ));
}

fn collect_spaceship_input(
    keys: Res<ButtonInput<KeyCode>>,
    mouse_motion: Res<AccumulatedMouseMotion>,
    cursor_options: Query<&CursorOptions, With<PrimaryWindow>>,
    mut input: ResMut<SpaceshipInput>,
    mut camera_rig: ResMut<SpaceshipCameraRig>,
) {
    let mut thrust = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyD) {
        thrust.x += 1.0;
    }
    if keys.pressed(KeyCode::KeyA) {
        thrust.x -= 1.0;
    }
    if keys.pressed(KeyCode::Space) {
        thrust.y += 1.0;
    }
    if keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight) {
        thrust.y -= 1.0;
    }
    if keys.pressed(KeyCode::KeyW) {
        thrust.z += 1.0;
    }
    if keys.pressed(KeyCode::KeyS) {
        thrust.z -= 1.0;
    }
    input.thrust = thrust.clamp_length_max(1.0);
    input.boost = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);

    if keys.just_pressed(KeyCode::KeyH) {
        input.hover_assist = !input.hover_assist;
    }
    if keys.just_pressed(KeyCode::KeyC) {
        camera_rig.mode = match camera_rig.mode {
            CameraMode::Cockpit => CameraMode::Orbit,
            CameraMode::Orbit => CameraMode::Cockpit,
        };
    }
    if keys.just_pressed(KeyCode::KeyR) {
        camera_rig.orbit_yaw = 0.0;
        camera_rig.orbit_pitch = 0.25;
        camera_rig.orbit_distance = ORBIT_DEFAULT_DISTANCE;
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        camera_rig.orbit_distance = (camera_rig.orbit_distance + 20.0).min(ORBIT_MAX_DISTANCE);
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        camera_rig.orbit_distance = (camera_rig.orbit_distance - 20.0).max(ORBIT_MIN_DISTANCE);
    }

    let mut keyboard_turn = Vec3::ZERO;
    if keys.pressed(KeyCode::ArrowUp) {
        keyboard_turn.x += 1.0;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        keyboard_turn.x -= 1.0;
    }
    if keys.pressed(KeyCode::ArrowLeft) {
        keyboard_turn.y += 1.0;
    }
    if keys.pressed(KeyCode::ArrowRight) {
        keyboard_turn.y -= 1.0;
    }
    if keys.pressed(KeyCode::KeyQ) {
        keyboard_turn.z -= 1.0;
    }
    if keys.pressed(KeyCode::KeyE) {
        keyboard_turn.z += 1.0;
    }
    input.keyboard_turn = keyboard_turn.clamp_length_max(1.0);

    let mouse_captured = cursor_options
        .single()
        .map(|cursor| cursor.grab_mode != CursorGrabMode::None)
        .unwrap_or(false);
    let mouse_delta = if mouse_captured {
        mouse_motion.delta
    } else {
        Vec2::ZERO
    };

    match camera_rig.mode {
        CameraMode::Cockpit => {
            input.look_delta += mouse_delta;
        }
        CameraMode::Orbit => {
            camera_rig.orbit_yaw -= mouse_delta.x * 0.006;
            camera_rig.orbit_pitch = (camera_rig.orbit_pitch - mouse_delta.y * 0.006)
                .clamp(-ORBIT_PITCH_LIMIT, ORBIT_PITCH_LIMIT);
        }
    }
}

fn fly_spaceship_system(
    time: Res<Time>,
    gravity: Res<PlanetGravity>,
    mut input: ResMut<SpaceshipInput>,
    autopilot: Res<SpaceshipAutopilot>,
    voxel_world: VoxelWorldQueryParam<Without<ShipVoxelGrid>>,
    mut accelerations: ResMut<Accelerations>,
    mut impulses: ResMut<Impulses>,
    ships: Query<
        (
            Entity,
            &Transform,
            &Velocity,
            &AngularVelocity,
            &Mass,
            &RotationalInertia,
        ),
        With<Spaceship>,
    >,
) {
    let dt = time.delta_secs();
    if dt <= 0.0 {
        return;
    }

    let look_delta = input.reset_frame_delta();

    for (entity, transform, velocity, angular_velocity, mass, inertia) in &ships {
        let forward = transform.forward().as_vec3();
        let right = transform.right().as_vec3();
        let up = transform.up().as_vec3();

        let local = input.thrust;
        if matches!(autopilot.mode, SpaceshipAutopilotMode::Manual) {
            let thrust_accel = if input.boost {
                BOOST_THRUST_ACCEL
            } else {
                MAIN_THRUST_ACCEL
            };
            let accel = right * (local.x * STRAFE_THRUST_ACCEL)
                + up * (local.y * STRAFE_THRUST_ACCEL)
                + forward * (local.z * thrust_accel);
            if accel != Vec3::ZERO {
                accelerations.apply_central_acceleration(entity, accel);
            }
        }

        if input.hover_assist {
            if let Some(hover_accel) = hover_assist_acceleration(
                &voxel_world,
                gravity.center,
                transform.translation,
                velocity.0,
                local.y,
            ) {
                accelerations.apply_central_acceleration(entity, hover_accel);
            }
        }

        if mass.0 <= f32::EPSILON {
            continue;
        }
        let mouse_turn = Vec3::new(-look_delta.y, -look_delta.x, 0.0) * MOUSE_TURN_RATE_PER_PIXEL;
        let local_target_turn = (input.keyboard_turn * KEY_TURN_RATE + mouse_turn)
            .clamp_length_max(MAX_TARGET_TURN_RATE);
        let target_angular_velocity =
            right * local_target_turn.x + up * local_target_turn.y + forward * local_target_turn.z;
        let delta_w =
            (target_angular_velocity - angular_velocity.0).clamp_length_max(ANGULAR_ACCEL * dt);
        if delta_w.length_squared() > 1e-8 {
            let world_inertia = inertia
                .0
                .get_rotated(transform.rotation.as_dquat())
                .mat
                .as_mat3();
            impulses.apply_rotational_impulse(entity, world_inertia * delta_w);
        }
    }
}

fn update_spaceship_camera(
    camera_rig: Res<SpaceshipCameraRig>,
    ships: Query<&Transform, With<Spaceship>>,
    mut cameras: Query<&mut Transform, (With<SpaceshipCamera>, Without<Spaceship>)>,
) {
    let Ok(ship) = ships.single() else {
        return;
    };
    let Ok(mut camera) = cameras.single_mut() else {
        return;
    };

    match camera_rig.mode {
        CameraMode::Cockpit => {
            *camera = *ship * Transform::from_translation(COCKPIT_CAMERA_OFFSET);
        }
        CameraMode::Orbit => {
            let focus = ship.transform_point(ORBIT_FOCUS_OFFSET);
            let orbit_rotation = ship.rotation
                * Quat::from_rotation_y(camera_rig.orbit_yaw)
                * Quat::from_rotation_x(camera_rig.orbit_pitch);
            let offset = orbit_rotation * Vec3::new(0.0, 24.0, camera_rig.orbit_distance);
            *camera =
                Transform::from_translation(focus + offset).looking_at(focus, ship.up().as_vec3());
        }
    }
}

fn update_spaceship_telemetry(
    gravity: Res<PlanetGravity>,
    camera_rig: Res<SpaceshipCameraRig>,
    input: Res<SpaceshipInput>,
    voxel_world: VoxelWorldQueryParam<Without<ShipVoxelGrid>>,
    ships: Query<(&Transform, &Velocity), With<Spaceship>>,
    mut telemetry: ResMut<SpaceshipTelemetry>,
) {
    let Ok((transform, velocity)) = ships.single() else {
        return;
    };
    let position = transform.translation;
    let r_vec = position - gravity.center;
    let radius = r_vec.length();
    if radius <= f32::EPSILON {
        return;
    }

    let up = r_vec / radius;
    let radial_velocity = velocity.0.dot(up);
    let horizontal_velocity = velocity.0 - up * radial_velocity;
    let mu = gravity.acceleration * gravity.reference_distance.powi(2);
    let circular_speed = (mu / radius).sqrt();
    let radial_altitude = radius - PLANET_RADIUS;
    let ground_altitude = ground_altitude(&voxel_world, gravity.center, position);

    let (apoapsis_altitude, periapsis_altitude, eccentricity) =
        orbit_elements(position, velocity.0, gravity.center, mu);

    *telemetry = SpaceshipTelemetry {
        speed: velocity.0.length(),
        vertical_speed: radial_velocity,
        horizontal_speed: horizontal_velocity.length(),
        radial_altitude,
        ground_altitude,
        apoapsis_altitude,
        periapsis_altitude,
        circular_speed,
        eccentricity,
        hover_assist_active: input.hover_assist,
        camera_mode: camera_rig.mode,
    };
}

fn spaceship_hud(
    mut contexts: EguiContexts,
    telemetry: Res<SpaceshipTelemetry>,
    autopilot: Res<SpaceshipAutopilot>,
) -> Result {
    let ctx = contexts.ctx_mut()?;

    egui::Window::new("Ship")
        .default_pos([8.0, 8.0])
        .default_size([245.0, 180.0])
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(format!("Camera: {} (C)", telemetry.camera_mode.label()));
            ui.label(format!("Autopilot: {:?}", autopilot.mode));
            ui.label(format!(
                "Hover assist: {} (H)",
                if telemetry.hover_assist_active {
                    "on"
                } else {
                    "off"
                }
            ));
            ui.separator();
            ui.label(format!("Speed: {}", format_speed(telemetry.speed)));
            ui.label(format!(
                "Vertical: {}",
                format_speed_signed(telemetry.vertical_speed)
            ));
            ui.label(format!(
                "Horizontal: {}",
                format_speed(telemetry.horizontal_speed)
            ));
            ui.separator();
            ui.label(format!(
                "Altitude: {}",
                format_distance(telemetry.radial_altitude)
            ));
            if let Some(ground_altitude) = telemetry.ground_altitude {
                ui.label(format!("Ground: {}", format_distance(ground_altitude)));
            } else {
                ui.label("Ground: --");
            }
            ui.separator();
            ui.label(format!(
                "Circular speed: {}",
                format_speed(telemetry.circular_speed)
            ));
            match (
                telemetry.periapsis_altitude,
                telemetry.apoapsis_altitude,
                telemetry.eccentricity,
            ) {
                (Some(periapsis), Some(apoapsis), Some(eccentricity)) => {
                    let orbit_state = if periapsis > 0.0 {
                        "bound orbit"
                    } else {
                        "intersecting"
                    };
                    ui.label(format!("Trajectory: {orbit_state}"));
                    ui.label(format!("Pe: {}", format_distance(periapsis)));
                    ui.label(format!("Ap: {}", format_distance(apoapsis)));
                    ui.label(format!("e: {:.3}", eccentricity));
                }
                _ => {
                    ui.label("Trajectory: escape");
                }
            }
            ui.separator();
            ui.label("WASD/Space/Ctrl thrust, Shift boost");
            ui.label("Mouse or arrows turn, Q/E roll");
            ui.label("Orbit cam: mouse orbit, [/] zoom, R reset");
        });

    Ok(())
}

fn hover_assist_acceleration(
    voxel_world: &VoxelWorldQueryParam<Without<ShipVoxelGrid>>,
    gravity_center: Vec3,
    position: Vec3,
    velocity: Vec3,
    vertical_input: f32,
) -> Option<Vec3> {
    let down = (gravity_center - position).normalize_or_zero();
    if down == Vec3::ZERO {
        return None;
    }
    let up = -down;
    let altitude = ground_altitude(voxel_world, gravity_center, position)
        .unwrap_or_else(|| (position - gravity_center).length() - PLANET_RADIUS);
    if !(0.0..HOVER_ASSIST_RANGE).contains(&altitude) {
        return None;
    }

    let strength = 1.0 - altitude / HOVER_ASSIST_RANGE;
    let pilot_override = if vertical_input.abs() > 0.1 {
        0.30
    } else {
        1.0
    };
    let vertical_speed = velocity.dot(up);
    let altitude_error = HOVER_TARGET_ALTITUDE - altitude;
    let accel_mag = (altitude_error * 6.0 - vertical_speed * 4.0)
        .clamp(-HOVER_MAX_ACCEL, HOVER_MAX_ACCEL)
        * strength
        * pilot_override;

    Some(up * accel_mag)
}

fn ground_altitude(
    voxel_world: &VoxelWorldQueryParam<Without<ShipVoxelGrid>>,
    gravity_center: Vec3,
    position: Vec3,
) -> Option<f32> {
    let down = (gravity_center - position).normalize_or_zero();
    if down == Vec3::ZERO {
        return None;
    }
    let origin = position + down * GROUND_RAY_CLEARANCE;
    voxel_world
        .raycast(origin, down, Some(GROUND_RAY_MAX_DISTANCE))
        .map(|hit| hit.distance + GROUND_RAY_CLEARANCE)
}

fn orbit_elements(
    position: Vec3,
    velocity: Vec3,
    gravity_center: Vec3,
    mu: f32,
) -> (Option<f32>, Option<f32>, Option<f32>) {
    let r_vec = position - gravity_center;
    let r = r_vec.length();
    if r <= f32::EPSILON || mu <= f32::EPSILON {
        return (None, None, None);
    }

    let v2 = velocity.length_squared();
    let energy = 0.5 * v2 - mu / r;
    if energy >= 0.0 {
        return (None, None, None);
    }

    let h = r_vec.cross(velocity);
    let e_vec = velocity.cross(h) / mu - r_vec / r;
    let eccentricity = e_vec.length();
    let semi_major_axis = -mu / (2.0 * energy);
    let periapsis = semi_major_axis * (1.0 - eccentricity) - PLANET_RADIUS;
    let apoapsis = semi_major_axis * (1.0 + eccentricity) - PLANET_RADIUS;
    (Some(apoapsis), Some(periapsis), Some(eccentricity))
}

fn format_speed(voxels_per_second: f32) -> String {
    format!("{:.1} m/s", voxels_per_second / VOXELS_PER_METER)
}

fn format_speed_signed(voxels_per_second: f32) -> String {
    format!("{:+.1} m/s", voxels_per_second / VOXELS_PER_METER)
}

fn format_distance(voxels: f32) -> String {
    let meters = voxels / VOXELS_PER_METER;
    if meters.abs() >= 1_000.0 {
        format!("{:.2} km", meters / 1_000.0)
    } else {
        format!("{:.1} m", meters)
    }
}

fn ship_present_chunks() -> Vec<IVec3> {
    let min = chunk_of(SHIP_BOUNDS_MIN);
    let max = chunk_of(SHIP_BOUNDS_MAX);
    let mut chunks = Vec::new();

    for x in min.x..=max.x {
        for y in min.y..=max.y {
            for z in min.z..=max.z {
                let chunk = IVec3::new(x, y, z);
                if chunk_has_ship_voxels(chunk) {
                    chunks.push(chunk);
                }
            }
        }
    }

    chunks
}

fn chunk_has_ship_voxels(chunk: IVec3) -> bool {
    let origin = chunk_origin(chunk);
    for x in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                if ship_voxel(origin + IVec3::new(x, y, z), SHIP_MASS_PER_VOXEL).is_some() {
                    return true;
                }
            }
        }
    }
    false
}

fn chunk_region_intersects_ship_bounds(min_chunk: IVec3, size_chunks: IVec3) -> bool {
    let min = chunk_origin(min_chunk);
    let max = chunk_origin(min_chunk + size_chunks) - IVec3::ONE;
    min.cmple(SHIP_BOUNDS_MAX).all() && max.cmpge(SHIP_BOUNDS_MIN).all()
}

fn build_ship_chunk(chunk: IVec3, include_mass: bool) -> Option<Voxels> {
    let origin = chunk_origin(chunk);
    let mass = if include_mass { SHIP_MASS_PER_VOXEL } else { 0 };
    let mut points = Vec::new();

    for x in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let local = IVec3::new(x, y, z);
                let Some(voxel) = ship_voxel(origin + local, mass) else {
                    continue;
                };
                points.push((local.as_u16vec3(), voxel));
            }
        }
    }

    points_to_voxels(points)
}

fn build_ship_lod_region(min_chunk: IVec3, size_chunks: IVec3, lod: f32) -> Option<Voxels> {
    let step = 1i32 << lod.max(0.0).floor() as u32;
    let sample_offset = step / 2;
    let extent = (size_chunks * CHUNK_SIZE) / step;
    let origin = chunk_origin(min_chunk);
    let max_source = size_chunks * CHUNK_SIZE - IVec3::ONE;
    let mut points = Vec::new();

    for x in 0..extent.x {
        for y in 0..extent.y {
            for z in 0..extent.z {
                let coarse = IVec3::new(x, y, z);
                let sample = (coarse * step + IVec3::splat(sample_offset)).min(max_source);
                let Some(voxel) = ship_voxel(origin + sample, 0) else {
                    continue;
                };
                points.push((coarse.as_u16vec3(), voxel));
            }
        }
    }

    points_to_voxels(points)
}

fn points_to_voxels(points: Vec<(U16Vec3, Voxel)>) -> Option<Voxels> {
    if points.is_empty() {
        None
    } else {
        let mut voxels = Voxels::new();
        voxels.add_voxels(&points);
        Some(voxels)
    }
}

fn ship_voxel(pos: IVec3, mass: u32) -> Option<Voxel> {
    if pos.cmplt(SHIP_BOUNDS_MIN).any() || pos.cmpgt(SHIP_BOUNDS_MAX).any() {
        return None;
    }

    if let Some(color) = cockpit_voxel(pos) {
        return Some(Voxel { color, mass });
    }
    if let Some(color) = fuselage_voxel(pos) {
        return Some(Voxel { color, mass });
    }
    if let Some(color) = wing_voxel(pos) {
        return Some(Voxel { color, mass });
    }
    if let Some(color) = tail_voxel(pos) {
        return Some(Voxel { color, mass });
    }
    if let Some(color) = thruster_voxel(pos) {
        return Some(Voxel { color, mass });
    }

    None
}

fn fuselage_voxel(pos: IVec3) -> Option<[u8; 4]> {
    let z = pos.z as f32;
    if !(-56.0..=42.0).contains(&z) {
        return None;
    }

    let t = ((z + 56.0) / 98.0).clamp(0.0, 1.0);
    let nose_taper = smoothstep(0.0, 0.22, t);
    let tail_taper = 1.0 - 0.35 * smoothstep(0.72, 1.0, t);
    let rx = (4.0 + 9.5 * nose_taper) * tail_taper;
    let ry = (3.0 + 6.0 * nose_taper) * tail_taper;
    let x = pos.x as f32;
    let y = pos.y as f32;
    let shell = (x / rx).powi(2) + ((y - 1.0) / ry).powi(2);
    if shell > 1.0 {
        return None;
    }

    let stripe = pos.y <= -4 && pos.z > -42 && pos.z < 30;
    Some(if stripe {
        [64, 104, 192, 255]
    } else if pos.z < -48 {
        [210, 214, 218, 255]
    } else {
        [176, 184, 192, 255]
    })
}

fn cockpit_voxel(pos: IVec3) -> Option<[u8; 4]> {
    let z = pos.z as f32;
    if !(-38.0..=-10.0).contains(&z) || pos.y < 3 {
        return None;
    }

    let t = ((z + 38.0) / 28.0).clamp(0.0, 1.0);
    let rx = 3.0 + 5.0 * (1.0 - (t - 0.5).abs() * 1.6).max(0.0);
    let ry = 3.0 + 2.0 * (1.0 - (t - 0.5).abs() * 1.6).max(0.0);
    let x = pos.x as f32;
    let y = (pos.y - 6) as f32;
    if (x / rx).powi(2) + (y / ry).powi(2) <= 1.0 {
        Some([60, 180, 255, 210])
    } else {
        None
    }
}

fn wing_voxel(pos: IVec3) -> Option<[u8; 4]> {
    if !(pos.y >= -3 && pos.y <= 1 && pos.z >= -12 && pos.z <= 32) {
        return None;
    }
    let z_t = ((pos.z + 12) as f32 / 44.0).clamp(0.0, 1.0);
    let half_span = 14.0 + 24.0 * (1.0 - (z_t - 0.45).abs() * 1.45).max(0.0);
    let body_clearance = 7.0 + 3.0 * (1.0 - z_t);
    let abs_x = pos.x.abs() as f32;
    if abs_x >= body_clearance && abs_x <= half_span {
        let edge = half_span - abs_x;
        Some(if edge < 2.0 {
            [80, 92, 106, 255]
        } else {
            [138, 146, 158, 255]
        })
    } else {
        None
    }
}

fn tail_voxel(pos: IVec3) -> Option<[u8; 4]> {
    if pos.z < 24 || pos.z > 46 {
        return None;
    }

    let vertical_fin = pos.x.abs() <= 2 && pos.y >= 4 && pos.y <= 18 && pos.z >= 28;
    let horizontal_fin = pos.y >= -1 && pos.y <= 4 && pos.x.abs() >= 9 && pos.x.abs() <= 24;
    if vertical_fin || horizontal_fin {
        Some([112, 122, 138, 255])
    } else {
        None
    }
}

fn thruster_voxel(pos: IVec3) -> Option<[u8; 4]> {
    if !(pos.z >= 38 && pos.z <= 52) {
        return None;
    }

    let pods = [IVec3::new(-7, -1, 45), IVec3::new(7, -1, 45)];
    for pod in pods {
        let rel = pos - pod;
        if rel.z >= -7 && rel.z <= 5 && (rel.x * rel.x + rel.y * rel.y) <= 16 {
            return Some(if pos.z >= 48 {
                [80, 190, 255, 255]
            } else {
                [46, 50, 58, 255]
            });
        }
    }

    None
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
