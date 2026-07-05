use bevy::input::ButtonInput;
use bevy::prelude::*;
use bevy::transform::components::{GlobalTransform, Transform};

use voxel_data::world_query::VoxelWorldQueryParam;
use voxel_physics::{CenterOfMass, FreezePhysics, Impulses, IsStatic, Mass, PhysicsSet, Velocity};

pub(crate) struct WorldInteractionPlugin;

impl Plugin for WorldInteractionPlugin {
	fn build(&self, app: &mut App) {
		app.init_resource::<HeldBody>()
			.add_systems(Update, pickup_toggle_system)
			.add_systems(
				FixedUpdate,
				hold_held_body_system
					.in_set(PhysicsSet::Apply)
					.run_if(|freeze: Res<FreezePhysics>| !freeze.0),
			);
	}
}

#[derive(Resource, Default, Debug, Clone, Copy)]
pub(crate) struct HeldBody(pub Option<Entity>);

const HOLD_DISTANCE: f32 = 80.0;
const MAX_GRAB_ACCEL: f32 = 8_000.0;

fn camera_ray(
	cameras: &Query<(&Camera, &GlobalTransform), With<Camera3d>>,
) -> Option<(Vec3, Vec3)> {
	let (_, camera_global_transform) = cameras.iter().find(|(camera, _)| camera.is_active)?;
	let transform = camera_global_transform.compute_transform();
	Some((transform.translation, transform.forward().as_vec3()))
}

fn pickup_toggle_system(
	keys: Res<ButtonInput<KeyCode>>,
	cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
	voxel_world: VoxelWorldQueryParam,
	parents: Query<&ChildOf>,
	bodies: Query<Has<IsStatic>, With<voxel_physics::RigidBody>>,
	mut held: ResMut<HeldBody>,
) {
	if !keys.just_pressed(KeyCode::KeyF) {
		return;
	}

	if held.0.is_some() {
		held.0 = None;
		return;
	}

	let Some((origin, dir)) = camera_ray(&cameras) else {
		return;
	};
	let Some(hit) = voxel_world.raycast(origin, dir, None) else {
		return;
	};
	let Ok(child_of) = parents.get(hit.grid) else {
		return;
	};

	let body = child_of.parent();
	let Ok(is_static) = bodies.get(body) else {
		return;
	};
	if is_static {
		return;
	}

	held.0 = Some(body);
}

fn hold_held_body_system(
	held: Res<HeldBody>,
	time: Res<Time>,
	cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
	bodies: Query<
		(&Transform, &Velocity, &Mass, &CenterOfMass),
		(With<voxel_physics::RigidBody>, Without<IsStatic>),
	>,
	mut impulses: ResMut<Impulses>,
) {
	let Some(body_entity) = held.0 else {
		return;
	};
	let Ok((transform, velocity, mass, center_of_mass)) = bodies.get(body_entity) else {
		return;
	};
	let Some((origin, forward)) = camera_ray(&cameras) else {
		return;
	};

	let target = origin + forward * HOLD_DISTANCE;
	let body_center_of_mass_world = *transform * center_of_mass.0;
	let offset = target - body_center_of_mass_world;
	if offset.length_squared() < 1e-6 {
		return;
	}

	let dir = offset.normalize();
	let velocity_in_dir = velocity.0.dot(dir);
	let delta_v = dir * (offset.length() * 4.0 - velocity_in_dir * 0.5)
		- (velocity.0 - dir * velocity_in_dir);
	let delta_v = delta_v.clamp_length_max(MAX_GRAB_ACCEL * time.delta_secs());

	impulses.apply_central_impulse(body_entity, mass.0 * delta_v);
}
