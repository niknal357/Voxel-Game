use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use voxel_data::world_query::VoxelWorldQueryParam;
use voxel_physics::{FreezePhysics, PhysicsSet};

use crate::gravity::PlanetGravity;
use crate::planet_source::PLANET_RADIUS;

pub(crate) struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_player)
            .add_systems(
                Update,
                (
                    toggle_cursor_grab,
                    player_look_system,
                    camera_follow_player_system,
                )
                    .chain(),
            )
            .add_systems(
                FixedUpdate,
                player_movement_system
                    .in_set(PhysicsSet::Apply)
                    .run_if(|freeze: Res<FreezePhysics>| !freeze.0),
            );
    }
}

#[derive(Component)]
pub(crate) struct Player;

#[derive(Component, Debug, Clone)]
pub(crate) struct PlayerController {
    walk_speed: f32,
    sprint_speed: f32,
    jump_speed: f32,
    mouse_sensitivity: f32,
    eye_height: f32,
    step_height: f32,
    ground_snap: f32,
    camera_vertical_smoothing: f32,
    camera_position: Option<Vec3>,
    flat_forward: Vec3,
    pitch: f32,
    vertical_velocity: f32,
    grounded: bool,
}

impl Default for PlayerController {
    fn default() -> Self {
        Self {
            walk_speed: 90.0,
            sprint_speed: 180.0,
            jump_speed: 85.0,
            mouse_sensitivity: 0.0015,
            eye_height: 32.0,
            step_height: 10.0,
            ground_snap: 8.0,
            camera_vertical_smoothing: 12.0,
            camera_position: None,
            flat_forward: Vec3::Z,
            pitch: 0.0,
            vertical_velocity: 0.0,
            grounded: false,
        }
    }
}

const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.1;
const FALL_ACCEL_MULTIPLIER: f32 = 1.0;

fn spawn_player(mut commands: Commands) {
    commands.spawn((
        Player,
        PlayerController::default(),
        Transform::from_xyz(0.0, PLANET_RADIUS + 180.0, 120.0),
    ));
}

fn player_look_system(
    mouse_motion: Res<AccumulatedMouseMotion>,
    cursor_options: Query<&CursorOptions, With<PrimaryWindow>>,
    gravity: Res<PlanetGravity>,
    mut players: Query<(&Transform, &mut PlayerController), With<Player>>,
) {
    let mouse_captured = cursor_options
        .single()
        .map(|cursor| cursor.grab_mode != CursorGrabMode::None)
        .unwrap_or(false);
    if !mouse_captured {
        return;
    }

    for (transform, mut controller) in &mut players {
        let up = planet_up(transform.translation, gravity.center);
        let flat_forward = tangent_forward(controller.flat_forward, up);
        let yaw_delta = -mouse_motion.delta.x * controller.mouse_sensitivity;

        controller.flat_forward =
            (Quat::from_axis_angle(up, yaw_delta) * flat_forward).normalize_or(flat_forward);
        controller.pitch -= mouse_motion.delta.y * controller.mouse_sensitivity;
        controller.pitch = controller.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }
}

fn player_movement_system(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    gravity: Res<PlanetGravity>,
    voxel_world: VoxelWorldQueryParam,
    mut players: Query<(&mut Transform, &mut PlayerController), With<Player>>,
) {
    let dt = time.delta_secs();

    for (mut transform, mut controller) in &mut players {
        let up = planet_up(transform.translation, gravity.center);
        let flat_forward = tangent_forward(controller.flat_forward, up);
        let flat_right = up.cross(flat_forward).normalize_or(Vec3::X);
        controller.flat_forward = flat_forward;

        let mut wish_dir = Vec3::ZERO;
        if keys.pressed(KeyCode::KeyW) {
            wish_dir += flat_forward;
        }
        if keys.pressed(KeyCode::KeyS) {
            wish_dir -= flat_forward;
        }
        if keys.pressed(KeyCode::KeyD) {
            wish_dir -= flat_right;
        }
        if keys.pressed(KeyCode::KeyA) {
            wish_dir += flat_right;
        }

        if keys.just_pressed(KeyCode::Space) && controller.grounded {
            controller.vertical_velocity = controller.jump_speed;
            controller.grounded = false;
        }

        let sprint = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
        let speed = if sprint {
            controller.sprint_speed
        } else {
            controller.walk_speed
        };
        let horizontal_motion = wish_dir.normalize_or_zero() * speed * dt;

        let gravity_accel = gravity_at(transform.translation, &gravity) * FALL_ACCEL_MULTIPLIER;
        controller.vertical_velocity -= gravity_accel * dt;

        let mut next_position =
            transform.translation + horizontal_motion + up * controller.vertical_velocity * dt;
        let next_up = planet_up(next_position, gravity.center);

        let ray_origin = next_position + next_up * controller.step_height;
        let ray_length = controller.eye_height + controller.step_height + controller.ground_snap;
        let ground_hit = voxel_world.raycast(ray_origin, -next_up, Some(ray_length));

        controller.grounded = false;
        if let Some(hit) = ground_hit {
            let eye_distance_from_ground = hit.distance - controller.step_height;
            let should_snap_to_ground = controller.vertical_velocity <= 0.0
                && eye_distance_from_ground <= controller.eye_height + controller.ground_snap;
            let is_inside_ground = eye_distance_from_ground < controller.eye_height;

            if should_snap_to_ground || is_inside_ground {
                next_position = hit.world_position + next_up * controller.eye_height;
                controller.vertical_velocity = controller.vertical_velocity.max(0.0);
                controller.grounded = should_snap_to_ground;
            }
        }

        transform.translation = next_position;
    }
}

fn camera_follow_player_system(
    time: Res<Time>,
    gravity: Res<PlanetGravity>,
    mut players: Query<(&Transform, &mut PlayerController), With<Player>>,
    mut cameras: Query<&mut Transform, (With<Camera3d>, Without<Player>)>,
) {
    let Ok((player_transform, mut controller)) = players.single_mut() else {
        return;
    };

    let up = planet_up(player_transform.translation, gravity.center);
    let flat_forward = tangent_forward(controller.flat_forward, up);
    let flat_right = up.cross(flat_forward).normalize_or(Vec3::X);
    let look_dir = (Quat::from_axis_angle(flat_right, -controller.pitch) * flat_forward)
        .normalize_or(flat_forward);

    let target_position = player_transform.translation;
    let visual_position = smoothed_camera_position(
        controller.camera_position,
        target_position,
        up,
        controller.camera_vertical_smoothing,
        time.delta_secs(),
    );
    controller.camera_position = Some(visual_position);

    for mut camera_transform in &mut cameras {
        camera_transform.translation = visual_position;
        camera_transform.look_to(look_dir, up);
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

fn planet_up(position: Vec3, center: Vec3) -> Vec3 {
    (position - center).normalize_or(Vec3::Y)
}

fn gravity_at(position: Vec3, gravity: &PlanetGravity) -> f32 {
    let to_center = gravity.center - position;
    let distance_squared = to_center.length_squared();
    if distance_squared <= f32::EPSILON {
        gravity.acceleration
    } else {
        gravity.acceleration * gravity.reference_distance.powi(2) / distance_squared
    }
}

fn smoothed_camera_position(
    current: Option<Vec3>,
    target: Vec3,
    up: Vec3,
    smoothing: f32,
    dt: f32,
) -> Vec3 {
    let Some(current) = current else {
        return target;
    };

    let to_target = target - current;
    if to_target.length_squared() > 250.0 * 250.0 {
        return target;
    }

    let vertical_delta = up * to_target.dot(up);
    let horizontal_delta = to_target - vertical_delta;
    let vertical_alpha = 1.0 - (-smoothing * dt).exp();

    current + horizontal_delta + vertical_delta * vertical_alpha
}

fn tangent_forward(forward: Vec3, up: Vec3) -> Vec3 {
    let projected = forward - up * forward.dot(up);
    if projected.length_squared() > 1e-6 {
        projected.normalize()
    } else {
        fallback_tangent_forward(up)
    }
}

fn fallback_tangent_forward(up: Vec3) -> Vec3 {
    let reference = if up.dot(Vec3::Y).abs() < 0.95 {
        Vec3::Y
    } else {
        Vec3::Z
    };
    up.cross(reference).normalize_or(Vec3::X)
}
